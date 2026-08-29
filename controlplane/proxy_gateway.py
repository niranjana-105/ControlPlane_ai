"""
ControlPlane.ai - FastAPI Reverse Proxy with SSE Token Interceptor
Implements /v1/chat/completions streaming endpoint with in-flight governance.
Supports both Built-in Simulator Scenarios and Real Live Upstream APIs (OpenAI, Groq, Ollama).
"""

import time
import uuid
import hashlib
import asyncio
import json
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from controlplane.config import DEFAULT_PROFILES, PolicyProfileType, PolicyProfile
from controlplane.ingress import IngressGate, IngressVerdict
from controlplane.cache import get_cache
from controlplane.session_state import get_or_create_session
from controlplane.nli_engine import NLIEngine
from controlplane.pii_redactor import PIIRedactor
from controlplane.bias_detector import BiasDetector
from controlplane.action_engine import resolve_actions, ActionType
from controlplane.simulator import ScenarioType, stream_response
from controlplane.telemetry import get_telemetry
from controlplane.live_upstream import LiveLLMClient


app = FastAPI(
    title="ControlPlane.ai — Enterprise AI Governance Proxy",
    description="Sub-20ms in-flight governance layer for LLM streaming responses.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ingress_gate = IngressGate()
_cache = get_cache()
_bias_detector = BiasDetector()
_telemetry = get_telemetry()


def _get_policy(profile_name: Optional[str]) -> PolicyProfile:
    try:
        profile_type = PolicyProfileType(profile_name or "Customer Support Bot")
    except ValueError:
        profile_type = PolicyProfileType.CUSTOMER_SUPPORT
    return DEFAULT_PROFILES[profile_type]


def _get_pii_redactor(policy: PolicyProfile) -> PIIRedactor:
    return PIIRedactor(sensitive_entities=policy.sensitive_entities)


async def _governed_stream(
    prompt: str,
    scenario: ScenarioType,
    policy: PolicyProfile,
    session_id: str,
    request_id: str,
    live_client: Optional[LiveLLMClient] = None,
    live_model: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    t_total_start = time.perf_counter()
    pii_redactor = _get_pii_redactor(policy)
    session_mgr = get_or_create_session(session_id)

    # 1. Tier 0: Ingress Gate
    ingress_result = _ingress_gate.evaluate(prompt)

    if ingress_result.verdict == IngressVerdict.BLOCK:
        block_event = {
            "event": "governance_block",
            "data": {
                "action": "HARD_BLOCK",
                "reason": ingress_result.blocked_reason,
                "latency_ms": ingress_result.latency_ms,
            }
        }
        yield f"data: {json.dumps(block_event)}\n\n"
        yield "data: [DONE]\n\n"
        return

    # 2. Cache Lookup
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    if policy.enable_semantic_cache:
        cache_result = _cache.lookup(prompt)
        if cache_result.hit:
            cache_event = {
                "event": "cache_hit",
                "data": {
                    "tier": cache_result.tier,
                    "similarity": cache_result.similarity_score,
                    "latency_ms": cache_result.latency_ms,
                }
            }
            yield f"data: {json.dumps(cache_event)}\n\n"
            words = (cache_result.response or "").split()
            for word in words:
                chunk = {"choices": [{"delta": {"content": word + " "}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.02)
            yield "data: [DONE]\n\n"
            return

    # 3. Tier 1 + 2: Egress Interception
    window_buffer = []
    full_response_tokens = []
    WINDOW_SIZE = 15

    session_eval = session_mgr.evaluate(policy.session_escalation_threshold)
    nli = NLIEngine(
        contradiction_threshold=policy.nli_contradiction_threshold,
        entropy_threshold=policy.entropy_uncertainty_threshold,
        enable_async_judge=True,
    )

    # Token source: Live LLM or Simulator
    if live_client:
        async for token_text in live_client.stream_chat(messages=[{"role": "user", "content": prompt}], model=live_model):
            for word in token_text.split():
                window_buffer.append(word)
                full_response_tokens.append(word)

                if len(window_buffer) >= WINDOW_SIZE:
                    window_text = " ".join(window_buffer)
                    action_result = await _process_window(
                        window_text, nli, pii_redactor, _bias_detector,
                        session_eval, policy, request_id, session_id
                    )
                    governed_text = action_result.transformed_text if action_result else window_text

                    gov_chunk = {
                        "choices": [{"delta": {"content": governed_text + " "}, "finish_reason": None}],
                        "_governance": {
                            "action": action_result.action_type.value if action_result else "PASSTHROUGH",
                            "flags": action_result.triggered_flags if action_result else [],
                        }
                    }
                    yield f"data: {json.dumps(gov_chunk)}\n\n"
                    window_buffer = []
                else:
                    pass_chunk = {"choices": [{"delta": {"content": word + " "}, "finish_reason": None}]}
                    yield f"data: {json.dumps(pass_chunk)}\n\n"
    else:
        for chunk in stream_response(scenario=scenario, token_delay_ms=0):
            meta = chunk.get("_meta", {})
            if meta.get("type") == "footer":
                break
            if meta.get("type") != "token":
                continue

            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if not content:
                continue

            word = content.strip()
            window_buffer.append(word)
            full_response_tokens.append(word)

            if len(window_buffer) >= WINDOW_SIZE:
                window_text = " ".join(window_buffer)
                action_result = await _process_window(
                    window_text, nli, pii_redactor, _bias_detector,
                    session_eval, policy, request_id, session_id
                )
                governed_text = action_result.transformed_text if action_result else window_text

                gov_chunk = {
                    "choices": [{"delta": {"content": governed_text + " "}, "finish_reason": None}],
                    "_governance": {
                        "action": action_result.action_type.value if action_result else "PASSTHROUGH",
                        "flags": action_result.triggered_flags if action_result else [],
                    }
                }
                yield f"data: {json.dumps(gov_chunk)}\n\n"
                window_buffer = []
            else:
                pass_chunk = {"choices": [{"delta": {"content": content}, "finish_reason": None}]}
                yield f"data: {json.dumps(pass_chunk)}\n\n"

            await asyncio.sleep(0)

    # Final flush
    if window_buffer:
        window_text = " ".join(window_buffer)
        action_result = await _process_window(
            window_text, nli, pii_redactor, _bias_detector,
            session_eval, policy, request_id, session_id
        )
        governed_text = action_result.transformed_text if action_result else window_text
        gov_chunk = {
            "choices": [{"delta": {"content": governed_text}, "finish_reason": "stop"}],
            "_governance": {
                "action": action_result.action_type.value if action_result else "PASSTHROUGH",
                "flags": action_result.triggered_flags if action_result else [],
            }
        }
        yield f"data: {json.dumps(gov_chunk)}\n\n"

    if policy.enable_semantic_cache and full_response_tokens:
        _cache.store(prompt, " ".join(full_response_tokens))

    total_latency_ms = (time.perf_counter() - t_total_start) * 1000
    summary = {
        "event": "governance_summary",
        "data": {
            "request_id": request_id,
            "session_id": session_id,
            "total_latency_ms": round(total_latency_ms, 2),
            "policy": policy.name.value,
        }
    }
    yield f"data: {json.dumps(summary)}\n\n"
    yield "data: [DONE]\n\n"


async def _process_window(
    window_text, nli, pii_redactor, bias_detector, session_eval, policy, request_id, session_id
):
    loop = asyncio.get_event_loop()
    nli_task = loop.run_in_executor(None, lambda: nli.evaluate(window_text, session_id=session_id))
    pii_task = loop.run_in_executor(None, lambda: pii_redactor.redact(window_text))
    bias_task = loop.run_in_executor(None, lambda: bias_detector.evaluate(window_text))

    nli_res, pii_res, bias_res = await asyncio.gather(nli_task, pii_task, bias_task)
    return resolve_actions(window_text, nli_res, pii_res, bias_res, session_eval, policy)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    prompt = messages[-1].get("content", "") if messages else ""
    profile_name = body.get("policy_profile", None)
    scenario_name = body.get("scenario", "clean")
    session_id = body.get("session_id", str(uuid.uuid4())[:8])
    request_id = str(uuid.uuid4())
    
    # Live API credentials if provided
    api_key = body.get("api_key") or request.headers.get("Authorization", "").replace("Bearer ", "")
    base_url = body.get("base_url", "https://api.openai.com/v1")
    model = body.get("model", "gpt-4o-mini")

    policy = _get_policy(profile_name)

    live_client = None
    if api_key or "localhost" in base_url or "127.0.0.1" in base_url:
        live_client = LiveLLMClient(api_key=api_key, base_url=base_url, default_model=model)

    try:
        scenario = ScenarioType(scenario_name)
    except ValueError:
        scenario = ScenarioType.CLEAN

    return StreamingResponse(
        _governed_stream(prompt, scenario, policy, session_id, request_id, live_client=live_client, live_model=model),
        media_type="text/event-stream",
        headers={
            "X-Request-ID": request_id,
            "X-Governance-Policy": policy.name.value,
            "X-Jurisdiction": policy.jurisdiction.value,
            "Cache-Control": "no-cache",
        }
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0", "service": "ControlPlane.ai"}
