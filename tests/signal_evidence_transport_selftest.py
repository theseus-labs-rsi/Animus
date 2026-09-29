"""Complete author task transport and durable draft slots; no model quality claim."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import config
from pipeline import corpus_contract as cc, evidence_transport as wire, render as renderer
from pipeline.prompts import render
from corpus_semantic_fidelity_selftest import Trace, world, output


class TransportTrace(Trace):
    def __init__(self, *, fail_review=False, fail_author=False, status="supported", pfile=None):
        super().__init__(status=status)
        self.fail_review, self.author_inputs, self.review_inputs = fail_review, [], []
        self.fail_author = fail_author
        if pfile is not None:
            self.pfile = pfile

    def chat_json(self, step, messages, **parameters):
        actual = deepcopy(messages)
        if step in {"render.signal", "corpus.review"}:
            payload = json.loads(messages[-1]["content"])
            payload = wire.unpack(payload)
            if step == "render.signal":
                self.author_inputs.append(deepcopy(payload))
                self.calls.append((step, actual, deepcopy(parameters)))
                if self.fail_author:
                    raise RuntimeError("Fixed offline author interruption")
                return {"docs": [{"title": "记录", "content": self.bodies[0]}]}
            self.review_inputs.append(deepcopy(payload))
            self.calls.append((step, actual, deepcopy(parameters)))
            if self.fail_review:
                raise RuntimeError("Fixed offline review interruption")
            return output(payload, status=self.status)
        return super().chat_json(step, messages, **parameters)


class SignalTransportTests(unittest.TestCase):
    def setUp(self):
        for obj, name in ((socket.socket, "connect"), (config, "chat"), (config, "chat_json")):
            guard = patch.object(obj, name, side_effect=AssertionError("Provider/network forbidden"))
            guard.start(); self.addCleanup(guard.stop)
        guard = patch.object(config, "pmap", side_effect=lambda fn, xs, **kw: [fn(x) for x in xs])
        guard.start(); self.addCleanup(guard.stop)

    def test_template_preserves_original_json_text_and_literal_dollar_sequences(self):
        variables = {"facts": [{"value": "$future", "marker": {"$evidence": "original"}}],
                     "text": "Original $s and $$ must remain literal"}
        messages, binding = wire.template_request("Original author rules", "$$ $text\n$facts", variables,
            json_variables=["facts"], step="author", parameters={})
        original = "$ " + variables["text"] + "\n" + json.dumps(variables["facts"], ensure_ascii=False)
        self.assertEqual(messages[-1]["content"], original)
        shared, shared_binding = wire.template_request("Original author rules", "$$ $text\n$facts", variables,
            json_variables=["facts"], step="author", parameters={}, protocol=wire.SHARED)
        logical = wire.unpack(json.loads(shared[-1]["content"]))
        self.assertEqual(wire.template_text(logical), original)
        self.assertEqual(binding["logical_input_hash"], shared_binding["logical_input_hash"])
        self.assertNotEqual(binding["wire_messages_hash"], shared_binding["wire_messages_hash"])

    def test_signal_inline_messages_and_parameters_equal_original_construction(self):
        facts = [{"entity": "甲", "field": "阶段", "value": "$开始", "date": "2025-01-06"}]
        events = [{"id": "e1", "session": 0, "effects": []}]
        context = {"facts": deepcopy(facts), "events": deepcopy(events),
                   "source_assertions": [{"source": "完整来源"}]}
        for quality in (False, True):
            for attempt in (0, 3):
                kwargs = dict(quality=quality, session_label=1, time_unit="期", date="2025-01-06",
                    facts=facts, events=events, story_context="\n原完整历史", hint="\n原负面反馈",
                    plan_text="\n原辅助安排", context=context, attempt=attempt)
                messages, binding, params = renderer._signal_request("Original system", **kwargs)
                text = render("corpus.quality_user" if quality else "corpus.user",
                    s=1, time_unit="期", date="2025-01-06", facts=json.dumps(facts, ensure_ascii=False),
                    events=json.dumps(events, ensure_ascii=False), story_context=kwargs["story_context"], hint=kwargs["hint"])
                text += kwargs["plan_text"]
                if quality:
                    text += "\n【生成与审阅共享的截至时点事实；不得把未知业务状态写成已发生】\n"
                    text += json.dumps(context, ensure_ascii=False) + "\n" + cc.SOURCE_ASSERTION_INSTRUCTIONS
                self.assertEqual(messages, [{"role": "system", "content": "Original system"}, {"role": "user", "content": text}])
                expected = {"temperature": .6 if attempt == 0 else .2, "max_tokens": renderer.SIGNAL_MAX_TOKENS}
                if quality:
                    expected["response_format"] = {"type": "json_object"}
                self.assertEqual(params, expected)
                shared, other_binding, other_params = renderer._signal_request("Original system", **kwargs,
                    evidence_transport=wire.SHARED)
                self.assertEqual(wire.template_text(wire.unpack(json.loads(shared[-1]["content"]))), text)
                self.assertEqual(other_params, params)
                self.assertEqual(other_binding["logical_input_hash"], binding["logical_input_hash"])

    def render(self, tracer, *, generation=False, protocol=wire.SHARED):
        wp = {"quality_contract": {"corpus_review": True}}
        if generation:
            wp["generation_contract"] = {"material_first": True}
        corpus = {"sessions": []}
        with patch.object(renderer, "_plan_signal", return_value=None):
            renderer.render_corpus(wp, world(), 0, tracer, corpus, set(), lambda: None,
                log=lambda *_: None, evidence_transport=protocol)
        return corpus

    def test_author_and_reviewer_share_complete_scope_without_changing_blind_reader(self):
        inline = Trace()
        original = self.render(inline, protocol=wire.INLINE)
        shared = TransportTrace()
        result = self.render(shared)
        self.assertEqual([row[0] for row in shared.calls], [row[0] for row in inline.calls])
        self.assertEqual(wire.template_text(shared.author_inputs[0]), inline.calls[0][1][-1]["content"])
        self.assertEqual(shared.review_inputs[0], json.loads(inline.calls[-1][1][-1]["content"]))
        self.assertEqual(shared.calls[1], inline.calls[1])
        self.assertEqual(result["sessions"][0]["docs"][0]["content"], original["sessions"][0]["docs"][0]["content"])
        self.assertEqual(cc.validate_corpus(world(), result)["status"], "passed")

    def test_negative_opinion_consumes_original_four_drafts_without_extra_retries(self):
        shared = TransportTrace(status="ambiguous")
        with self.assertRaisesRegex(RuntimeError, "4 轮"):
            self.render(shared)
        self.assertEqual(len(shared.author_inputs), 4)
        self.assertEqual(len(shared.review_inputs), 4)
        self.assertTrue(shared.author_inputs[1]["variables"]["hint"])
        self.assertTrue(all(p["CANON"] == shared.review_inputs[0]["CANON"] for p in shared.review_inputs))

    def test_saved_author_reply_is_not_redispatched_and_tampered_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            pfile = Path(td) / "prompts.jsonl"
            first = TransportTrace(fail_review=True, pfile=pfile)
            with self.assertRaisesRegex(RuntimeError, "Fixed offline"):
                self.render(first, generation=True)
            checkpoint, = (Path(td) / "05_signal_checkpoints").glob("*.json")
            saved = json.loads(checkpoint.read_text(encoding="utf8"))
            self.assertEqual(len(saved["progress"]["drafts"]), 1)
            second = TransportTrace(pfile=pfile)
            with self.assertRaisesRegex(RuntimeError, "Fixed offline"):
                self.render(second, generation=True)
            self.assertEqual(len(second.author_inputs), 0)
            self.assertEqual(len(second.review_inputs), 0)
            saved["progress"]["drafts"][0]["author_transport_binding"]["logical_input_hash"] = "wrong"
            saved["progress_hash"] = cc.fingerprint(saved["progress"])
            checkpoint.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf8")
            third = TransportTrace(pfile=pfile)
            with self.assertRaisesRegex(ValueError, "transport/input"):
                self.render(third, generation=True)
            self.assertEqual(third.calls, [])

    def test_lost_author_response_consumes_original_slot_on_resume(self):
        with tempfile.TemporaryDirectory() as td:
            pfile = Path(td) / "prompts.jsonl"
            first = TransportTrace(fail_author=True, pfile=pfile)
            with self.assertRaisesRegex(RuntimeError, "Fixed offline author"):
                self.render(first, generation=True)
            checkpoint, = (Path(td) / "05_signal_checkpoints").glob("*.json")
            saved = json.loads(checkpoint.read_text(encoding="utf8"))
            self.assertTrue(saved["progress"]["drafts"][0]["admitted"])
            self.assertNotIn("author_output", saved["progress"]["drafts"][0])
            second = TransportTrace(status="ambiguous", pfile=pfile)
            with self.assertRaisesRegex(RuntimeError, "4 轮"):
                self.render(second, generation=True)
            self.assertEqual(len(first.author_inputs) + len(second.author_inputs), 4)
            self.assertEqual(len(second.review_inputs), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
