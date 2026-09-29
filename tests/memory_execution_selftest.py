"""Offline adapter regressions: real adapters, fake transports/SDK boundaries."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.public_context import build_full_context  # noqa: E402 - FullContext 的真实格式化依赖
config = types.ModuleType('config')
config.chat_json = Mock(side_effect=AssertionError('unexpected external model call'))
config.MODEL = 'offline'
runner_stub = types.ModuleType('eval.multi_system')
runner_stub.header = lambda sid, date: f'[{sid} {date}] '
runner_stub.build_embed_memory = Mock(side_effect=AssertionError('unexpected embedding'))
runner_stub.build_full_context = lambda docs, budget: ('\n'.join(d[2] for d in docs), False, 0)
requests_stub = types.ModuleType('requests')
requests_stub.Session = Mock(side_effect=AssertionError('unexpected HTTP session'))
httpx_stub = types.ModuleType('httpx')
httpx_stub.Timeout = lambda *args, **kwargs: ('offline-timeout',)
with patch.dict(sys.modules, {
    'config': config, 'eval.multi_system': runner_stub,
    'eval.memory_interface': types.SimpleNamespace(EmbedMemory=object, _chunk=Mock(), build_embed_memory=Mock()),
    'eval.embed_cache': types.SimpleNamespace(cached_embed=Mock()),
    'requests': requests_stub, 'httpx': httpx_stub,
}):
    from eval.memory_systems.base import MemoryExecutionError
    from eval.memory_systems.execution import SDKGuard, validate_json_completion
    from eval.memory_systems import memos_adapter, mem0_adapter, amem_adapter, zep_adapter
    from eval.memory_systems import iterative, simplemem, fullcontext

SESSION = {'session_id': 1, 'date': '2026-01-01', 'docs': ['alpha', 'beta']}


class Response:
    def __init__(self, value=None, status=200, parse_error=None, text=None):
        self.value, self.status_code, self.parse_error = value, status, parse_error
        self.text = text if text is not None else json.dumps(value)
        self.content = b'' if value is None and parse_error is None and text is None else self.text.encode()
    def raise_for_status(self):
        if self.status_code >= 400:
            e = RuntimeError('provider details must not be exposed')
            e.response = self
            raise e
    def json(self):
        if self.parse_error:
            raise self.parse_error
        return self.value


def memos(responses):
    session = Mock(headers={})
    session.post.side_effect = responses
    with patch.object(memos_adapter.requests, 'Session', return_value=session):
        obj = memos_adapter.MemOSAdapter(base_url='http://offline.invalid',
            finalize_attempts=2, finalize_interval=0)
    return obj


def mem0(memory):
    obj = mem0_adapter.Mem0Adapter.__new__(mem0_adapter.Mem0Adapter)
    obj.memory, obj._guard = memory, SDKGuard()
    obj._user_id, obj._last_context, obj.top_k = 'offline-user', '', 3
    return obj


def amem(system):
    obj = amem_adapter.AMemAdapter.__new__(amem_adapter.AMemAdapter)
    obj._sys, obj._guard = system, SDKGuard()
    obj._last_context, obj.top_k = '', 3
    return obj


def zep(responses=(), mode='ce', profile='auto_legacy'):
    obj = zep_adapter.ZepAdapter.__new__(zep_adapter.ZepAdapter)
    obj._mode, obj._session_id, obj._base = mode, 'offline-session', 'http://offline.invalid'
    obj._session, obj.client = Mock(), Mock()
    obj._session.post.side_effect = list(responses)
    obj.top_k, obj.ingest_wait = 3, 0
    obj._last_context, obj._last_message_uuid, obj._has_ingested = '', None, False
    obj.api_profile, obj._ingest_batch, obj._expected_messages = profile, 'offline-batch', {}
    obj.finalize_attempts, obj.finalize_interval = 2, 0
    return obj


class ExecutionTest(unittest.TestCase):
    def setUp(self):
        config.chat_json.reset_mock(side_effect=True)
        config.chat_json.side_effect = AssertionError('unexpected external model call')

    def failure(self, stage, call, code=None):
        with self.assertRaises(MemoryExecutionError) as ctx:
            call()
        error = ctx.exception
        self.assertEqual(error.stage, stage)
        if code:
            self.assertEqual(error.code, code)
        self.assertNotIn('provider details', str(error))
        self.assertNotIn('provider details', json.dumps(error.as_dict()))
        return error

    def test_memos_ingest_timeout_cannot_claim_docs(self):
        obj = memos([TimeoutError('provider details')])
        e = self.failure('ingest', lambda: obj.ingest_session(SESSION))
        self.assertEqual(e.details['completed_docs'], 0)
        self.assertEqual(e.details['requested_docs'], 2)
        self.assertEqual(obj._conv_ids, [])

    def test_memos_rejected_and_malformed_ingest(self):
        for response in [Response({}, status=500), Response({'code': 8}),
                         Response({'code': 0, 'data': {'success': False}}),
                         Response([], parse_error=ValueError('bad JSON'))]:
            with self.subTest(value=response.value):
                obj = memos([response])
                self.failure('ingest', lambda: obj.ingest_session(SESSION))

    def test_memos_retrieve_errors_are_not_context(self):
        for response in [TimeoutError(), Response({}, status=503),
                         Response({'code': 7}), Response({'code': 0, 'data': {}}),
                         Response({'code': 0, 'data': {'memory_detail_list': [None]}}),
                         Response([], parse_error=ValueError('bad JSON'))]:
            with self.subTest(value=repr(response)):
                obj = memos([response]); obj._last_context = 'previous context'
                self.failure('retrieve', lambda: obj.retrieve('question'))
                self.assertEqual(obj.get_retrieved_context(), '')

    def test_memos_valid_empty_is_success(self):
        obj = memos([Response({'code': 0, 'data': {'memory_detail_list': []}})])
        self.assertEqual(obj.retrieve('question'), '(无检索结果)')

    def test_memos_async_is_only_accepted_until_task_completes(self):
        obj = memos([Response({'code': 0, 'data': {'success': True, 'task_id': 'task', 'status': 'running'}}),
                     Response({'code': 0, 'data': {'task_id': 'task', 'status': 'running'}}),
                     Response({'code': 0, 'data': {'task_id': 'task', 'status': 'completed'}})])
        receipt = obj.ingest_session(SESSION)
        self.assertEqual(receipt['completion'], 'accepted')
        obj.finalize_ingest()
        self.assertFalse(obj._pending_tasks)
        self.assertTrue(obj._session.post.call_args.args[0].endswith('/get/status'))

    def test_memos_finalization_failures(self):
        for state, code in [('failed', 'processing_failed'), ('running', 'completion_timeout')]:
            obj = memos([Response({'code': 0, 'data': {'task_id': 'task', 'status': 'running'}}),
                         Response({'code': 0, 'data': {'task_id': 'task', 'status': state}}),
                         Response({'code': 0, 'data': {'task_id': 'task', 'status': state}})])
            obj.ingest_session(SESSION)
            self.failure('finalize', obj.finalize_ingest, code)
        obj = memos([Response({'code': 0, 'data': {'success': True}})])
        self.assertEqual(obj.ingest_session(SESSION)['completion'], 'accepted')
        self.failure('finalize', obj.finalize_ingest, 'completion_unverified')

    def test_mem0_partial_ingest_reports_confirmed_prefix(self):
        obj = mem0(types.SimpleNamespace(add=Mock(side_effect=[{'results': []}, TimeoutError()])))
        e = self.failure('ingest', lambda: obj.ingest_session(SESSION))
        self.assertEqual(e.details['completed_docs'], 1)
        self.assertEqual(obj.memory.add.call_count, 2)

    def test_mem0_malformed_add_and_valid_no_new_facts(self):
        obj = mem0(types.SimpleNamespace(add=Mock(return_value=None)))
        self.failure('ingest', lambda: obj.ingest_session(SESSION), 'invalid_response')
        obj.memory.add.return_value = {'results': []}
        self.assertEqual(obj.ingest_session(SESSION)['n_docs'], 2)

    def test_mem0_retrieve_empty_failure_and_malformed(self):
        obj = mem0(types.SimpleNamespace(search=Mock(return_value={'results': []})))
        self.assertEqual(obj.retrieve('question'), '(无检索结果)')
        obj.memory.search.return_value = {}
        self.failure('retrieve', lambda: obj.retrieve('question'), 'invalid_response')
        obj.memory.search.side_effect = TimeoutError()
        self.failure('retrieve', lambda: obj.retrieve('question'))

    def test_mem0_sdk_swallowed_llm_and_write_failures_are_retained(self):
        for component, name in [('llm', 'generate_response'), ('vector_store', 'insert')]:
            dep = types.SimpleNamespace(**{name: Mock(side_effect=TimeoutError())})
            def add(*args, **kwargs):
                try:
                    getattr(dep, name)()
                except Exception:
                    pass
                return {'results': []}
            obj = mem0(types.SimpleNamespace(add=add))
            obj._guard.watch(dep, name)
            self.failure('ingest', lambda: obj.ingest_session(SESSION))

    def test_sdk_bad_json_cannot_turn_into_zero_extracted_memories(self):
        llm = types.SimpleNamespace(generate_response=Mock(return_value='{bad json'))
        def add(*args, **kwargs):
            try:
                llm.generate_response(response_format={'type': 'json_object'})
            except Exception:
                pass
            return {'results': []}
        obj = mem0(types.SimpleNamespace(add=add))
        obj._guard.watch(llm, 'generate_response', validate_json_completion)
        e = self.failure('ingest', lambda: obj.ingest_session(SESSION), 'dependency_completion_unverified')
        self.assertEqual(e.details['failures'][0]['code'], 'invalid_json')

    def test_mem0_constructor_installs_sdk_observation(self):
        llm = types.SimpleNamespace(generate_response=Mock(return_value='{bad json'), client=None)
        def add(*args, **kwargs):
            try:
                llm.generate_response(response_format={'type': 'json_object'})
            except Exception:
                pass
            return {'results': []}
        sdk = types.SimpleNamespace(llm=llm, add=add)
        module = types.SimpleNamespace(Memory=types.SimpleNamespace(from_config=Mock(return_value=sdk)))
        with patch.dict(sys.modules, {'mem0': module}), patch.dict('os.environ', {}, clear=True):
            obj = mem0_adapter.Mem0Adapter()
            e = self.failure('ingest', lambda: obj.ingest_session(SESSION), 'dependency_completion_unverified')
            self.assertEqual(e.details['failures'][0]['code'], 'invalid_json')

    def test_sdk_successful_exact_retry_is_recovery(self):
        dep = types.SimpleNamespace(search=Mock(side_effect=[TimeoutError(), {'results': ['ok']}]))
        guard = SDKGuard(); guard.watch(dep, 'search')
        def recovering_sdk():
            try:
                return dep.search(query='q', options={'k': 2})
            except Exception:
                return dep.search(query='q', options={'k': 2})
        self.assertEqual(guard.run('retrieve', recovering_sdk), {'results': ['ok']})

    def test_sdk_success_on_different_query_does_not_clear_failure(self):
        dep = types.SimpleNamespace(search=Mock(side_effect=[TimeoutError(), []]))
        guard = SDKGuard(); guard.watch(dep, 'search')
        def partial_sdk():
            try:
                dep.search(query='first')
            except Exception:
                pass
            return dep.search(query='second')
        self.failure('retrieve', lambda: guard.run('retrieve', partial_sdk),
                     'dependency_completion_unverified')

    def test_amem_partial_ingest_and_valid_empty(self):
        system = types.SimpleNamespace(add_note=Mock(side_effect=['note-1', TimeoutError()]),
                                       search_agentic=Mock(return_value=[]))
        obj = amem(system)
        e = self.failure('ingest', lambda: obj.ingest_session(SESSION))
        self.assertEqual(e.details['completed_docs'], 1)
        self.assertEqual(obj.retrieve('question'), '(无检索结果)')

    def test_amem_sdk_swallowed_retrieval_is_not_empty_success(self):
        retriever = types.SimpleNamespace(search=Mock(side_effect=TimeoutError()))
        def search_agentic(*args, **kwargs):
            try:
                retriever.search('question')
            except Exception:
                return []
        obj = amem(types.SimpleNamespace(retriever=retriever, search_agentic=search_agentic))
        self.failure('retrieve', lambda: obj.retrieve('question'))

    def test_amem_sdk_malformed_partial_search_is_not_empty_success(self):
        retriever = types.SimpleNamespace(search=Mock(return_value={'ids': [['id-1']], 'metadatas': [[]]}))
        def search_agentic(*args, **kwargs):
            try:
                retriever.search('question')
            except Exception:
                return []
        obj = amem(types.SimpleNamespace(retriever=retriever, search_agentic=search_agentic))
        e = self.failure('retrieve', lambda: obj.retrieve('question'), 'dependency_completion_unverified')
        self.assertEqual(e.details['failures'][0]['code'], 'invalid_response')

    def test_amem_llm_exhaustion_never_returns_empty_json(self):
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
            create=Mock(side_effect=TimeoutError('provider details')))))
        llm = types.SimpleNamespace(client=client, model='offline')
        system = types.SimpleNamespace(llm_controller=types.SimpleNamespace(llm=llm))
        amem_adapter._harden_llm(system)
        with patch.object(amem_adapter.time, 'sleep'):
            self.failure('ingest', lambda: llm.get_completion('prompt'), 'llm_failed')
        self.assertEqual(client.chat.completions.create.call_count, 3)

    def test_amem_sdk_swallowed_analysis_failure_is_not_successful_note(self):
        llm = types.SimpleNamespace(get_completion=Mock(side_effect=TimeoutError()))
        def add_note(*args, **kwargs):
            try:
                llm.get_completion('prompt', response_format={'type': 'json_object'})
            except Exception:
                pass
            return 'stored-with-defaults'
        obj = amem(types.SimpleNamespace(llm_controller=types.SimpleNamespace(llm=llm), add_note=add_note))
        self.failure('ingest', lambda: obj.ingest_session(SESSION))

    def test_zep_ce_initialization_checks_http(self):
        session = Mock(headers={}); session.post.return_value = Response({}, status=500)
        with patch.dict('os.environ', {'ZEP_API_KEY': ''}), patch.object(zep_adapter._req, 'Session', return_value=session):
            self.failure('init', zep_adapter.ZepAdapter)
        self.assertEqual(session.post.call_count, 1)

    def test_zep_versioned_ce_routes_and_plain_text_write_ack(self):
        session = Mock(headers={})
        session.post.side_effect = [Response({}), Response({}), Response(text='OK'), Response(text='OK')]
        with patch.dict('os.environ', {'ZEP_API_KEY': 'unused-cloud-key'}), \
             patch.object(zep_adapter._req, 'Session', return_value=session):
            obj = zep_adapter.ZepAdapter(api_profile='ce_0_27_2', ingest_wait=0)
        self.assertEqual(obj._mode, 'ce')
        self.assertTrue(session.post.call_args_list[0].args[0].endswith('/api/v1/user'))
        receipt = obj.ingest_session(SESSION)
        self.assertEqual(receipt['completion'], 'accepted')
        self.assertEqual(receipt['completion_scope'], 'message_embeddings')
        self.assertEqual(len(obj._expected_messages), 4)
        sent = session.post.call_args_list[2].kwargs['json']['messages']
        self.assertIn('eval_message_id', sent[0]['metadata'])

    def test_zep_versioned_ce_rejects_wrong_success_body(self):
        for response in [Response(text='NOT OK'), Response({'success': True})]:
            obj = zep([response], profile='ce_0_27_2')
            self.failure('ingest', lambda: obj.ingest_session(SESSION), 'invalid_response')
            self.assertEqual(obj._expected_messages, {})

    def indexed_zep(self):
        obj = zep(profile='ce_0_27_2')
        written = []
        def write(*args, **kwargs):
            written.extend(kwargs['json']['messages'])
            return Response(text='OK')
        obj._session.post.side_effect = write
        obj.ingest_session(SESSION)
        rows = [{'message': {**message, 'uuid': f'uuid-{i}'}} for i, message in enumerate(written)]
        return obj, rows

    def test_zep_versioned_ce_verifies_all_indexed_messages(self):
        obj, rows = self.indexed_zep()
        obj._session.post.reset_mock()
        obj._session.post.side_effect = [Response(rows[:2]), Response(rows)]
        ready = obj.finalize_ingest()
        self.assertEqual(ready['scope'], 'message_embeddings')
        self.assertEqual(ready['verified_messages'], 4)
        self.assertEqual(ready['background_tasks'], 'not_verified')
        call = obj._session.post.call_args
        self.assertEqual(call.kwargs['json']['text'], '')
        self.assertEqual(call.kwargs['json']['search_scope'], 'messages')
        self.assertIn('offline-batch', call.kwargs['json']['metadata']['where']['jsonpath'])
        self.assertEqual(call.kwargs['params']['limit'], 5)

    def test_zep_versioned_ce_missing_indexed_message_times_out(self):
        obj, rows = self.indexed_zep()
        obj._session.post.side_effect = None
        obj._session.post.return_value = Response(rows[:-1])
        error = self.failure('finalize', obj.finalize_ingest, 'completion_timeout')
        self.assertEqual(error.details['verified_messages'], 3)

    def test_zep_versioned_ce_rejects_stale_duplicate_and_changed_messages(self):
        from copy import deepcopy
        for mutation in ['batch', 'content', 'role', 'duplicate']:
            obj, rows = self.indexed_zep()
            changed = deepcopy(rows)
            if mutation == 'batch':
                changed[0]['message']['metadata']['eval_ingest_id'] = 'old-batch'
            elif mutation == 'content':
                changed[0]['message']['content'] = 'altered'
            elif mutation == 'role':
                changed[0]['message']['role'] = 'assistant'
            else:
                changed.append(deepcopy(changed[0]))
            obj._session.post.side_effect = None
            obj._session.post.return_value = Response(changed)
            self.failure('finalize', obj.finalize_ingest, 'invalid_response')

    def test_zep_versioned_search_limit_is_query_parameter(self):
        obj = zep([Response([])], profile='ce_0_27_2')
        self.assertEqual(obj.retrieve('question', top_k=7), '(无检索结果)')
        call = obj._session.post.call_args
        self.assertEqual(call.kwargs['params'], {'limit': 7})
        self.assertNotIn('limit', call.kwargs['json'])

    def test_zep_partial_chunk_write(self):
        obj = zep([Response([{'uuid': 'first'}]), Response({}, status=500)])
        with patch.object(zep_adapter, '_split_text', return_value=['part1', 'part2']):
            e = self.failure('ingest', lambda: obj.ingest_session(SESSION))
        self.assertEqual(e.details['completed_docs'], 0)
        self.assertEqual(e.details['accepted_chunks'], 1)

    def test_zep_ce_empty_and_malformed_or_failed_search(self):
        obj = zep([Response([])])
        self.assertEqual(obj.retrieve('question'), '(无检索结果)')
        for response in [TimeoutError(), Response({}, status=503), Response({}), Response([{}])]:
            obj = zep([response]); self.failure('retrieve', lambda: obj.retrieve('question'))

    def test_zep_cloud_empty_and_failure(self):
        obj = zep(mode='cloud')
        obj.client.memory.search.return_value = types.SimpleNamespace(results=[])
        self.assertEqual(obj.retrieve('question'), '(无检索结果)')
        obj.client.memory.search.return_value = types.SimpleNamespace(results=None)
        self.assertEqual(obj.retrieve('question'), '(无检索结果)')
        obj.client.memory.get.return_value = types.SimpleNamespace(facts=None)
        self.assertEqual(obj.get_memory_snapshot()['n_facts'], 0)
        obj.client.memory.search.side_effect = TimeoutError()
        self.failure('retrieve', lambda: obj.retrieve('question'))

    def test_zep_cloud_partial_ingest(self):
        obj = zep(mode='cloud')
        obj.client.memory.add.side_effect = [types.SimpleNamespace(messages=[]), TimeoutError()]
        with patch.dict(sys.modules, {'zep_cloud': types.SimpleNamespace(Message=lambda **kw: kw)}):
            e = self.failure('ingest', lambda: obj.ingest_session(SESSION))
        self.assertEqual(e.details['completed_docs'], 1)

    def test_zep_summary_is_not_a_whole_ingest_completion_proof(self):
        obj = zep([Response([{'uuid': 'last'}]), Response([{'uuid': 'last'}])])
        self.assertEqual(obj.ingest_session(SESSION)['completion'], 'accepted')
        obj._session.get.return_value = Response({'summary': {'content': 'old', 'recent_message_uuid': 'old'}})
        self.failure('finalize', obj.finalize_ingest, 'completion_unverified')
        obj._session.get.return_value = Response({'summary': {'content': 'new', 'recent_message_uuid': 'last'}})
        self.failure('finalize', obj.finalize_ingest, 'completion_unverified')

    def test_zep_finalization_no_cursor_or_http_failure(self):
        obj = zep([Response(None), Response(None)])
        obj.ingest_session(SESSION)
        obj._session.get.return_value = Response({})
        self.failure('finalize', obj.finalize_ingest, 'completion_unverified')
        obj._last_message_uuid = 'last'
        obj._session.get.return_value = Response({}, status=500)
        self.failure('finalize', obj.finalize_ingest)

    def test_iterative_bridge_error_is_not_one_hop_success(self):
        memory = types.SimpleNamespace(retrieve=Mock(return_value=['first hop']))
        obj = iterative.Iterative(embed_mem=memory)
        config.chat_json.side_effect = TimeoutError()
        self.failure('retrieve', lambda: obj.retrieve('question'))
        self.assertEqual(memory.retrieve.call_count, 1)
        self.assertEqual(obj.get_retrieved_context(), '')

    def test_iterative_no_bridge_is_valid_but_malformed_json_is_not(self):
        memory = types.SimpleNamespace(retrieve=Mock(return_value=[]))
        obj = iterative.Iterative(embed_mem=memory)
        config.chat_json.side_effect = None
        config.chat_json.return_value = {'bridge': '', 'subquestion': ''}
        self.assertEqual(obj.retrieve('question'), '(无检索结果)')
        config.chat_json.return_value = {}
        self.failure('retrieve', lambda: obj.retrieve('question'), 'invalid_response')

    def test_iterative_parallel_diagnostics_stay_with_their_question(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        barrier = Barrier(2)
        memory = types.SimpleNamespace(retrieve=lambda q, top_k: [q])
        obj = iterative.Iterative(embed_mem=memory)
        def bridge(messages, **kwargs):
            question = 'alpha' if '最终问题】alpha' in messages[-1]['content'] else 'beta'
            return {'bridge': question, 'subquestion': question + '-hop2'}
        def worker(question):
            context = obj.retrieve(question)
            barrier.wait(timeout=5)
            return context, obj.get_retrieved_context(), obj.get_diagnostics()
        config.chat_json.side_effect = bridge
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(worker, ['alpha', 'beta']))
        for question, (context, saved_context, diag) in zip(['alpha', 'beta'], results):
            self.assertEqual(diag['bridge'], question)
            self.assertEqual(context, saved_context)
            self.assertIn(question + '-hop2', context)
        self.assertEqual(obj.get_diagnostics(), {})
        obj.reset()
        self.assertEqual(obj.get_retrieved_context(), '')

    def test_instance_evaluation_config_binds_baseline_knobs(self):
        with patch.dict(sys.modules, {'config': config, 'numpy': types.ModuleType('numpy')}):
            spec = importlib.util.spec_from_file_location('offline_config_memory', ROOT/'eval/memory_interface.py')
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        memory = module.EmbedMemory(model='declared-model', chunk=True, chunk_chars=123)
        cfg = simplemem.SimpleMem(top_k=2, embed_mem=memory).evaluation_config()
        self.assertEqual(cfg['configuration_status'], 'declared')
        self.assertEqual(cfg['embedding']['model'], 'declared-model')
        self.assertEqual(cfg['embedding']['chunk_chars'], 123)
        self.assertEqual(cfg['embedding']['encoder_hf_id'], 'BAAI/bge-small-zh-v1.5')
        self.assertNotEqual(fullcontext.FullContext(99).evaluation_config(),
                            fullcontext.FullContext(100).evaluation_config())
        self.assertNotEqual(iterative.Iterative(embed_mem=memory, max_tokens=99).evaluation_config(),
                            iterative.Iterative(embed_mem=memory, max_tokens=100).evaluation_config())
        unknown = simplemem.SimpleMem(embed_mem=types.SimpleNamespace()).evaluation_config()
        self.assertEqual(unknown['configuration_status'], 'undeclared')
        json.dumps(cfg)

    def test_instance_evaluation_config_binds_external_knobs_without_ids_or_secrets(self):
        obj = zep(profile='ce_0_27_2')
        first = obj.evaluation_config()
        obj._session_id, obj._ingest_batch = 'different-random-id', 'different-batch'
        self.assertEqual(first, obj.evaluation_config())
        obj.api_profile = 'ce_legacy'
        self.assertNotEqual(first, obj.evaluation_config())
        remote = memos([])
        remote.base_url = 'https://user:secret-token@offline.invalid/private'
        remote._user_id = 'random-private-user'
        encoded = json.dumps(remote.evaluation_config())
        self.assertNotIn('secret-token', encoded)
        self.assertNotIn('random-private-user', encoded)
        with patch.object(amem_adapter, '_build_amem', return_value=types.SimpleNamespace()):
            a = amem_adapter.AMemAdapter(llm_model='model-a')
            b = amem_adapter.AMemAdapter(llm_model='model-b')
        self.assertNotEqual(a.evaluation_config(), b.evaluation_config())
        json.dumps(first)

    def test_baseline_finalization_failure_and_fullcontext_compatibility(self):
        obj = simplemem.SimpleMem()
        self.assertEqual(obj.ingest_session(SESSION)['n_docs'], 2)
        with patch.object(simplemem, 'build_embed_memory', side_effect=TimeoutError()):
            self.failure('finalize', obj.finalize_ingest)
        full = fullcontext.FullContext()
        full.ingest_session(SESSION); full.finalize_ingest()
        # FullContext 现在直接使用 eval.public_context.build_full_context（不再经 legacy
        # driver 转发），因此对真实格式化结果断言，而不是 driver stub 的裸拼接。
        expected, _, _ = build_full_context(
            [(SESSION['session_id'], SESSION['date'], doc) for doc in SESSION['docs']],
            budget=full.budget,
        )
        self.assertEqual(full.retrieve('question'), expected)

    def test_local_embedding_partial_batch_is_atomic(self):
        with patch.dict(sys.modules, {'config': config, 'numpy': types.ModuleType('numpy')}):
            spec = importlib.util.spec_from_file_location('offline_memory_interface', ROOT/'eval/memory_interface.py')
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        obj = module.EmbedMemory(chunk=False)
        with patch.object(module, 'embed_texts', return_value=[]), patch.dict(sys.modules, {
            'eval.memory_systems.base': types.SimpleNamespace(MemoryExecutionError=MemoryExecutionError),
        }):
            self.failure('ingest', lambda: obj.ingest('one document'), 'embedding_count_mismatch')
        self.assertEqual(obj._docs, [])


if __name__ == '__main__':
    with patch('socket.socket', side_effect=AssertionError('network disabled')):
        unittest.main(verbosity=2)
