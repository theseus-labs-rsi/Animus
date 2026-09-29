"""
config.py —— 复用 self_instruct_demo 的 API 思路:OpenAI 兼容客户端 + .env。
本项目(memory_bench_factory)的所有 LLM 调用都走这里。

注意:.env 里的 MODEL 是 deepseek 推理模型(会先吐一段隐藏 reasoning_content,
再给正文 content)。所以:
  - max_tokens 要给足,否则 token 预算被 reasoning 吃光、正文为空;
  - 解析 JSON 时要剥掉模型可能加的 ```json 代码块包裹。
"""
import os
import re
import sys
import json
import threading
import time
import uuid
import hashlib
import asyncio
import math
from contextvars import copy_context
from pathlib import Path
from llm_trace import emit, redact

from dotenv import load_dotenv
import openai as _openai
import httpx

# 显式指向本文件同目录的 .env,这样不管从哪个 cwd 跑都能找到。
load_dotenv(Path(__file__).parent / ".env")

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL = os.getenv("MODEL")

if not API_KEY:
    sys.exit("[配置错误] 未找到 OPENAI_API_KEY,请检查 .env。")
if not MODEL:
    sys.exit("[配置错误] 未找到 MODEL,请检查 .env。")

# 盲判别只做短 JSON 抽取，允许固定使用不展开 reasoning 的模型；不设置时
# 沿用主模型，保持任意 OpenAI 兼容端点可直接运行。
DISCRIMINATOR_MODEL = os.getenv("DISCRIMINATOR_MODEL") or MODEL

# 大型关系/事件 JSON 需要稳定吐正文；只允许显式固定路由，不做运行时模型 fallback。
STRUCTURE_MODEL = os.getenv("STRUCTURE_MODEL") or MODEL
REVIEWER_MODEL = os.getenv("REVIEWER_MODEL") or MODEL
JUDGE_MODEL = os.getenv("JUDGE_MODEL") or REVIEWER_MODEL

# ★socket 级超时(防死连接,不防慢代理):read 缺省 580s,代理持续发数据就不触发。
#   总响应时间由 chat() 的 DEADLINE_S 截止;这里只管 connect/write/pool 快速失败。
HTTP_READ_TIMEOUT_S = float(os.getenv("LLM_HTTP_READ_TIMEOUT_S", "580"))
# Each physical call owns an AsyncOpenAI/HTTP client and closes it before
# returning. A timed-out synchronous daemon thread cannot cancel its socket.

# ★全局 LLM 在飞并发上限:无论上层 pmap 怎么嵌套/并行,真正打到 API 的请求数 ≤ LLM_CONCURRENCY。
#   这一个数 = "API 能同时扛多少不抽风"的旋钮(env: LLM_CONCURRENCY,默认 8)。
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "8"))
_LLM_SEM = threading.BoundedSemaphore(LLM_CONCURRENCY)
DEADLINE_S = int(os.getenv("LLM_DEADLINE_S", "600"))
MIN_COMPLETION_TOKENS = int(os.getenv("LLM_MIN_COMPLETION_TOKENS", "0"))
# Optional ordinary-call override; explicit transport profiles retain precedence.
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "").strip()


class CompletionOutputError(ValueError):
    """The provider returned no usable completion, or stopped at its token limit."""

    def __init__(self, message, *, kind="empty_completion", token_cap=None):
        super().__init__(message)
        self.kind, self.token_cap = kind, token_cap
        self.retryable = kind == "output_truncated"


class CallDeadlineError(TimeoutError):
    """The local request was cancelled and cleaned up; remote cancellation is unknown."""

    kind, retryable = "total_deadline", False

    def __init__(self, message, *, phase, deadline_s):
        super().__init__(message)
        self.phase, self.deadline_s = phase, deadline_s
        self.local_cleanup_confirmed = True
        self.remote_cancellation = "unknown"


class ChatJSONError(ValueError):
    """A failed logical request with machine-readable evidence for its caller."""

    def __init__(self, message, *, cause, attempts, token_cap):
        super().__init__(message)
        self.kind = chat_error_kind(cause)
        self.retryable = False  # This logical request has exhausted its allowed policy.
        self.attempts, self.token_cap = attempts, token_cap
        for field in ("phase", "call_id", "deadline_s", "input_chars", "local_cleanup_confirmed", "remote_cancellation",
                      "usage_status", "response_received"):
            setattr(self, field, getattr(cause, field, None))


class CompletionRefusalError(ValueError):
    """An explicit provider refusal; repeating the same request cannot resolve it."""

    kind, retryable = "provider_refusal", False


def chat_error_kind(exc):
    if hasattr(exc, "kind"):
        return exc.kind
    cause = exc.__cause__
    if isinstance(exc, httpx.ReadTimeout) or isinstance(cause, httpx.ReadTimeout):
        return "http_read_timeout"
    if isinstance(exc, httpx.WriteTimeout) or isinstance(cause, httpx.WriteTimeout):
        return "http_write_timeout"
    for cls, kind in ((httpx.ConnectTimeout, "http_connect_timeout"), (httpx.PoolTimeout, "http_pool_timeout"),
                      (httpx.ConnectError, "http_connect_error"), (httpx.ReadError, "http_read_error"),
                      (httpx.WriteError, "http_write_error"), (httpx.RemoteProtocolError, "http_protocol_error")):
        if isinstance(exc, cls) or isinstance(cause, cls):
            return kind
    if isinstance(exc, TimeoutError):
        return "unclassified_timeout"
    if isinstance(exc, json.JSONDecodeError):
        return "json_syntax"
    status = getattr(exc, "status_code", None)
    return f"http_status_{status}" if type(status) is int else type(exc).__name__


def is_retryable_chat_error(exc):
    """Recognize transient completion/transport failures, without retrying policy or local errors."""
    if isinstance(exc, (CompletionRefusalError, CallDeadlineError, ChatJSONError)):
        return False
    if isinstance(exc, CompletionOutputError):
        return exc.kind == "output_truncated"
    if isinstance(exc, (TimeoutError, httpx.ReadTimeout, httpx.WriteTimeout)):
        return False  # Unknown remote progress; repeating the same workload is unsafe.
    if isinstance(exc, (ConnectionError, httpx.ConnectTimeout, httpx.PoolTimeout,
                        httpx.NetworkError, httpx.RemoteProtocolError)):
        return True
    # Tests and compatible clients may expose only OpenAI(), without SDK exception types.
    connection_type = getattr(_openai, "APIConnectionError", None)
    if isinstance(connection_type, type) and isinstance(exc, connection_type):
        cause = exc.__cause__
        if isinstance(cause, (httpx.ReadTimeout, httpx.WriteTimeout)):
            return False
        return not isinstance(exc, getattr(_openai, "APITimeoutError", ())) or isinstance(
            cause, (httpx.ConnectTimeout, httpx.PoolTimeout))
    status_type = getattr(_openai, "APIStatusError", None)
    if isinstance(status_type, type) and isinstance(exc, status_type):
        body = getattr(exc, "body", None)
        details = body.get("error", body) if isinstance(body, dict) else {}
        code = getattr(exc, "code", None) or (details.get("code") if isinstance(details, dict) else None)
        if isinstance(code, str) and code in {"insufficient_quota", "billing_hard_limit_reached", "billing_not_active",
                    "usage_limit_reached", "monthly_budget_exceeded", "content_policy_violation"}:
            return False
        status = getattr(exc, "status_code", None)
        return type(status) is int and (status in (408, 429) or 500 <= status <= 599)
    return False


def _completion_text(response):
    """提取正文；空正文时保留服务端停止原因与 token 证据。"""
    choices = getattr(response, "choices", None) or []
    if not choices:
        usage = getattr(response, "usage", None)
        raise CompletionOutputError(
            "LLM 返回空 choices:"
            f"completion_tokens={getattr(usage, 'completion_tokens', None)}")
    choice = choices[0]
    if getattr(choice.message, "refusal", None) or getattr(choice, "finish_reason", None) == "content_filter":
        raise CompletionRefusalError("LLM 明确拒绝作答或响应被内容策略过滤")
    text = choice.message.content or ""
    if text.strip() and getattr(choice, "finish_reason", None) != "length":
        return text
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    raise CompletionOutputError(
        "LLM 返回空正文或被截断:"
        f"finish_reason={getattr(choice, 'finish_reason', None)},"
        f"completion_tokens={getattr(usage, 'completion_tokens', None)},"
        f"reasoning_tokens={getattr(details, 'reasoning_tokens', None)}",
        kind="output_truncated" if getattr(choice, "finish_reason", None) == "length" else "empty_completion")


def _trace_secrets():
    return tuple(value for key, value in os.environ.items()
                 if key.endswith(("API_KEY", "API_TOKEN", "ACCESS_TOKEN", "PASSWORD")))


def _response_record(response):
    usage = getattr(response, "usage", None)
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump(mode="json")
    elif usage is not None:
        usage = {key: getattr(usage, key, None) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    return {"id": getattr(response, "id", None), "model": getattr(response, "model", None),
            "usage": usage,
            "choices": [{"index": getattr(c, "index", i),
                         "content": getattr(c.message, "content", None),
                         "refusal": getattr(c.message, "refusal", None),
                         "finish_reason": getattr(c, "finish_reason", None)}
                        for i, c in enumerate(getattr(response, "choices", None) or [])]}


def _perform_request(*, model, messages, parameters, http_timeout, deadline_s, lifecycle,
                     explicit_profile=False):
    """Cancellable HTTP lifecycle. SDK retries are always disabled.

    Observe response headers and raw HTTP body bytes, not generated model tokens.
    This retains the non-SSE Chat Completions wire contract and complete usage.
    The synchronous entrypoint is used from factory worker threads; it must not
    run inside another asyncio loop.
    """
    async def run():
        state = {"phase": "awaiting_http_headers", "body_bytes": 0, "cleanup_confirmed": False}

        class ObservedStream(httpx.AsyncByteStream):
            def __init__(self, stream):
                self.stream, self.last_progress = stream, 0.0

            async def __aiter__(self):
                async for chunk in self.stream:
                    if chunk:
                        first = state["body_bytes"] == 0
                        state["phase"] = "reading_http_body"
                        state["body_bytes"] += len(chunk)
                        now = time.monotonic()
                        if first or now - self.last_progress >= 10:
                            lifecycle("http_body_first_bytes" if first else "http_body_progress",
                                      received_bytes=state["body_bytes"], observation="http_bytes_not_model_tokens")
                            self.last_progress = now
                    yield chunk

            async def aclose(self):
                await self.stream.aclose()

        async def headers(response):
            state["phase"] = "awaiting_http_body"
            lifecycle("http_response_headers", status_code=response.status_code,
                      observation="http_headers_not_first_token")
            response.stream = ObservedStream(response.stream)

        async def request():
            http_client = httpx.AsyncClient(timeout=http_timeout, event_hooks={"response": [headers]})
            failure = None
            try:
                sdk = _openai.AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, max_retries=0,
                                          timeout=http_timeout, http_client=http_client)
                return await sdk.chat.completions.create(model=model, messages=messages, **parameters)
            except asyncio.CancelledError:
                lifecycle("request_cancellation_requested", phase=state["phase"],
                          remote_cancellation="unknown")
                raise
            except Exception as exc:
                failure = exc
                exc.phase = state["phase"]
                exc.kind = chat_error_kind(exc)
                raise
            finally:
                # wait_for waits for this finally block before it raises timeout.
                # The semaphore in chat remains held throughout local cleanup.
                await http_client.aclose()
                state["cleanup_confirmed"] = http_client.is_closed
                if failure is not None:
                    failure.local_cleanup_confirmed = state["cleanup_confirmed"]
                    failure.remote_cancellation = "unknown"
                lifecycle("request_client_closed", local_cleanup_confirmed=state["cleanup_confirmed"],
                          received_bytes=state["body_bytes"])

        try:
            return await asyncio.wait_for(request(), timeout=deadline_s)
        except asyncio.TimeoutError as exc:
            lifecycle("request_deadline", phase=state["phase"], deadline_s=deadline_s,
                      local_cleanup_confirmed=state["cleanup_confirmed"], remote_cancellation="unknown")
            error = CallDeadlineError(f"LLM 调用超总截止 {deadline_s}s，已关闭本地连接",
                                      phase=state["phase"], deadline_s=deadline_s)
            error.local_cleanup_confirmed = state["cleanup_confirmed"]
            raise error from exc

    return asyncio.run(run())


def chat(messages, temperature=0.7, top_p=1.0, max_tokens=4096, model=None, *, response_format=None,
         transport=None):
    """限并发/总时间；可选显式 transport，默认保持旧请求参数。"""
    effective_model = model or MODEL
    deadline_s, resolved_transport = DEADLINE_S, None
    http_timeout = httpx.Timeout(HTTP_READ_TIMEOUT_S, connect=15.0, write=30.0, pool=15.0)
    sdk_options = {"max_retries": 0, "timeout": {"read": HTTP_READ_TIMEOUT_S,
                   "connect": 15.0, "write": 30.0, "pool": 15.0}}
    if transport is None:
        # Keep legacy dependencies and exact wire behavior independent of profiles.
        request_parameters = {"temperature": temperature, "max_tokens": max(max_tokens, MIN_COMPLETION_TOKENS)}
        if top_p is not None:
            request_parameters["top_p"] = top_p
        if response_format is not None:
            request_parameters["response_format"] = response_format
        if REASONING_EFFORT:
            from llm_transport import RUN_MODEL_CAPABILITIES
            capability = RUN_MODEL_CAPABILITIES.get(effective_model)
            allowed = capability["reasoning_efforts"] if capability is not None else ()
            if REASONING_EFFORT not in allowed:
                raise ValueError(f"LLM_REASONING_EFFORT={REASONING_EFFORT!r} is unsupported for "
                                 f"model {effective_model!r}; supported values: {allowed}")
            request_parameters["reasoning_effort"] = REASONING_EFFORT
    else:
        from llm_transport import validate_transport, resolve_profile, build_request_parameters
        transport = validate_transport(transport)
        resolved_transport = resolve_profile(transport, effective_model)
        request_parameters = build_request_parameters(model=effective_model, max_tokens=max_tokens,
            temperature=temperature, top_p=top_p, response_format=response_format,
            min_completion_tokens=MIN_COMPLETION_TOKENS, transport=transport)
        profile = resolved_transport["profile"]
        if profile is not None:
            if "http_timeout_seconds" in profile:
                http_timeout = profile["http_timeout_seconds"]
                sdk_options = {"max_retries": 0, "timeout_seconds": http_timeout}
            deadline_s = profile.get("deadline_seconds", DEADLINE_S)
    call_id = uuid.uuid4().hex
    started = time.monotonic()
    secrets = _trace_secrets()
    cap = request_parameters.get("max_completion_tokens", request_parameters.get("max_tokens"))
    input_chars = sum(len(str(message.get("content", ""))) for message in messages)
    acquired = _LLM_SEM.acquire(timeout=deadline_s)
    if not acquired:
        error = CallDeadlineError("等待本地 LLM 并发位超时；未发送请求", phase="local_queue", deadline_s=deadline_s)
        error.kind, error.remote_cancellation = "local_queue_deadline", "not_dispatched"
        raise error
    queue_ms = round((time.monotonic() - started) * 1000)
    remaining_deadline_s = max(0.001, deadline_s - (time.monotonic() - started))
    dispatched = time.monotonic()
    def lifecycle(event, **fields):
        emit(event, secrets=secrets, call_id=call_id,
             elapsed_ms=round((time.monotonic() - dispatched) * 1000), **fields)
    response_record = None
    try:
        emit("request", secrets=secrets, call_id=call_id, messages=messages,
             model=effective_model, endpoint_hash=hashlib.sha256((BASE_URL or "").encode()).hexdigest(),
             logical_parameters={"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
                                 "response_format": response_format},
             parameters={**request_parameters, "deadline_s": deadline_s}, queue_wait_ms=queue_ms,
             request_deadline_remaining_s=remaining_deadline_s,
             transport=resolved_transport, sdk_options=sdk_options,
             lifecycle="cancellable_async_http/v1", input_chars=input_chars, actual_output_cap=cap)
        response = _perform_request(model=effective_model, messages=messages, parameters=request_parameters,
            http_timeout=http_timeout, deadline_s=remaining_deadline_s, lifecycle=lifecycle,
            explicit_profile=bool(resolved_transport and resolved_transport["profile"] is not None))
        response_record = _response_record(response)
        emit("response", secrets=secrets, call_id=call_id, response=response_record,
             elapsed_ms=round((time.monotonic() - started) * 1000))
        return _completion_text(response)
    except Exception as exc:
        exc.call_id, exc.token_cap, exc.input_chars = call_id, cap, input_chars
        exc.kind = chat_error_kind(exc)
        exc.response_received = response_record is not None
        usage = (response_record or {}).get("usage") or {}
        exc.usage_status = "reported" if all(type(usage.get(k)) is int and usage[k] >= 0
            for k in ("prompt_tokens", "completion_tokens")) else "unknown"
        emit("call_error", secrets=secrets, call_id=call_id, error_type=type(exc).__name__,
             error=str(exc), kind=getattr(exc, "kind", type(exc).__name__),
             retryable=is_retryable_chat_error(exc), actual_output_cap=cap,
             phase=getattr(exc, "phase", None), usage_status=exc.usage_status,
             local_cleanup_confirmed=getattr(exc, "local_cleanup_confirmed", None),
             remote_cancellation=getattr(exc, "remote_cancellation", None),
             elapsed_ms=round((time.monotonic() - started) * 1000))
        raise
    finally:
        _LLM_SEM.release()


def _strip_code_fence(text):
    """剥掉 ```json ... ``` 这类代码块包裹,只留里面的内容。"""
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text


def chat_json(messages, temperature=0.7, max_tokens=4096, retries=3,
              retry_delay_base=3.0, model=None, strict_json=False, *, top_p=1.0,
              response_format=None, transport=None, retry_parse_feedback=True,
              max_output_tokens=None):
    """调用 LLM 并把回复解析成 JSON 对象。

    暂时连接故障和 JSON 语法错误可重试。截断仅在显式 max_output_tokens
    允许提高实际额度时重试；总截止、读超时、空正文不重复相同工作。
    明确拒绝、鉴权、参数、预算及本地审计错误立即终止，不重复发送请求。
    推理模型偶尔会在 JSON 外面带点话或裹代码块,做"剥壳 + 抓第一个 {..}/[..]" 容错。
    ``strict_json=True`` 时只接受去掉代码围栏后的完整 JSON，不抓子串、不修语法；
    与 ``retries=1`` 合用可形成单次、无修复的硬协议。
    ``top_p=None`` 显式省略该传输参数；不按模型名推测参数兼容性。
    ``response_format`` 仅显式提供时透传；JSON 模式不替代严格解析/语义验收。
    ``transport`` 仅逐调用传入；显式配置错误在重试循环前拒绝，不修改全局配置。
    JSON 解析失败时，下一次请求附上本次输出和解析位置，让同一模型重新完整作答。
    反馈仅占用原有有限重试次数；不修改调用方 messages，不放宽严格解析或语义验收。
    """
    import time as _time
    if type(retries) is not int or retries < 1:
        raise ValueError("retries must be a positive integer upper bound")
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if max_output_tokens is not None and (type(max_output_tokens) is not int or max_output_tokens < max_tokens):
        raise ValueError("max_output_tokens must be an integer at least max_tokens")
    if not math.isfinite(retry_delay_base) or retry_delay_base < 0:
        raise ValueError("retry_delay_base must be finite and nonnegative")
    transport_kwargs = {}
    if transport is not None:
        from llm_transport import validate_transport, build_request_parameters
        transport = validate_transport(transport)
        build_request_parameters(model=model or MODEL, max_tokens=max_tokens,
            temperature=temperature, top_p=top_p, response_format=response_format,
            min_completion_tokens=MIN_COMPLETION_TOKENS, transport=transport)
        transport_kwargs["transport"] = transport
    last_err = None
    raw = ""
    request_messages = messages
    current_cap = max_tokens
    attempts = 0
    for attempt in range(retries):
        attempts = attempt + 1
        raw = ""
        attempt_id = uuid.uuid4().hex
        emit("json_attempt", secrets=_trace_secrets(), attempt_id=attempt_id,
             attempt=attempt + 1, max_attempts=retries, strict_json=strict_json,
             requested_output_cap=current_cap, maximum_output_cap=max_output_tokens)
        try:
            raw = chat(request_messages, temperature=temperature, max_tokens=current_cap,
                       model=model, top_p=top_p, response_format=response_format, **transport_kwargs)
            cleaned = _strip_code_fence(raw)
            try:
                parsed = json.loads(cleaned)
                emit("json_result", secrets=_trace_secrets(), attempt_id=attempt_id, parse_mode="complete", parsed=parsed)
                return parsed
            except json.JSONDecodeError as e:
                last_err = e
                if strict_json:
                    raise
                m = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
                if m:
                    try:
                        parsed = json.loads(m.group(1))
                        emit("json_result", secrets=_trace_secrets(), attempt_id=attempt_id, parse_mode="substring", parsed=parsed)
                        return parsed
                    except json.JSONDecodeError as e2:
                        last_err = e2
                # ★ json_repair 兜底:治 reasoning 模型偶发的少逗号/多逗号等 JSON 语法手滑
                try:
                    from json_repair import repair_json
                    obj = repair_json(cleaned, return_objects=True)
                except Exception as e3:
                    # The optional parser cannot replace the original syntax failure.
                    # Keep trace writes outside the repair try so audit failures stop.
                    emit("json_repair_error", secrets=_trace_secrets(), attempt_id=attempt_id,
                         error_type=type(e3).__name__, error=str(e3))
                    raise last_err from e3
                if obj not in (None, "", {}, []):
                    emit("json_result", secrets=_trace_secrets(), attempt_id=attempt_id, parse_mode="repaired", parsed=obj)
                    return obj
                raise last_err
        except Exception as e:
            last_err = e
            retryable = isinstance(e, json.JSONDecodeError) or is_retryable_chat_error(e)
            next_cap = current_cap
            retry_action = "json_feedback" if isinstance(e, json.JSONDecodeError) else "transient_backoff"
            if isinstance(e, CompletionOutputError):
                actual_cap = e.token_cap or current_cap
                next_cap = min(actual_cap * 2, max_output_tokens or actual_cap)
                retryable = e.kind == "output_truncated" and next_cap > actual_cap
                retry_action = "increase_output_cap" if retryable else "stop_output_failure"
            emit("json_error", secrets=_trace_secrets(), attempt_id=attempt_id,
                 error_type=type(e).__name__, error=str(e), raw_output=raw, retryable=retryable,
                 kind=getattr(e, "kind", "json_syntax" if isinstance(e, json.JSONDecodeError) else type(e).__name__),
                 retry_action=retry_action if retryable else "stop", next_output_cap=next_cap if retryable else None)
            if not retryable:
                break
            if attempt < retries - 1:
                current_cap = next_cap
                if retry_parse_feedback and isinstance(e, json.JSONDecodeError):
                    # Retain the original task and only the latest failed output.
                    # A large malformed reply must not grow the conversation on each retry.
                    limit = 8192
                    excerpt = raw if len(raw) <= limit else (
                        raw[:limit // 2] + "\n[中间内容已截去]\n" + raw[-limit // 2:])
                    feedback = (
                        "上一次回复无法解析为完整 JSON。解析器报告："
                        f"{e.msg}；第 {e.lineno} 行，第 {e.colno} 列，字符位置 {e.pos}。"
                        + ("上条助手消息仅保留原输出的头尾片段。" if len(raw) > limit else "")
                        + "请继续遵守原始任务及所有字段和业务要求，重新输出完整、合法的 JSON。"
                        "不要仅输出修改片段，不要添加解释或代码围栏，不要为修复格式编造事实。"
                    )
                    request_messages = list(messages) + [
                        {"role": "assistant", "content": excerpt},
                        {"role": "user", "content": feedback},
                    ]
                    emit("json_retry_feedback", secrets=_trace_secrets(), attempt_id=attempt_id,
                         next_attempt=attempt + 2, parse_error=str(e),
                         raw_output_chars=len(raw), excerpt_truncated=len(raw) > limit)
                delay = retry_delay_base * (2 ** attempt)
                print(f"[chat_json] 第 {attempt+1}/{retries} 次失败 "
                      f"({type(e).__name__}: {redact(str(e), _trace_secrets())[:80]}), {delay:.1f}s 后重试...")
                _time.sleep(delay)
                continue
    raise ChatJSONError(redact(
        f"chat_json {attempts} 次尝试后仍失败。最后错误: {last_err}\n"
        f"--- 原始输出前 800 字 ---\n{raw[:800]}", _trace_secrets()),
        cause=last_err, attempts=attempts, token_cap=getattr(last_err, "token_cap", None) or current_cap) from last_err


from concurrent.futures import ThreadPoolExecutor


def pmap(fn, items, workers: int = 6):
    """把【彼此独立的 LLM 调用】并发跑(网络 I/O 密集 → 线程池即可,GIL 在等网络时释放)。
    保序返回 list;workers 控并发上限(别太大,免得把抽风的 API 挤爆)。fn 内异常会向上抛。"""
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(items)))) as ex:
        pending = [ex.submit(copy_context().run, fn, item) for item in items]
        return [future.result() for future in pending]
