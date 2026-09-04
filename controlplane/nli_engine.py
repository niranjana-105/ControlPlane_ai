"""
ControlPlane.ai - NLI Contradiction Engine + Async AI-as-Judge
15-token sliding window NLI, entropy-based uncertainty detection,
and non-blocking background AI-as-Judge evaluation.
"""

import re
import time
import math
import asyncio
import hashlib
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from collections import Counter


@dataclass
class NLIResult:
    is_contradiction: bool
    needs_hedging: bool
    hard_block: bool
    score: float              # Contradiction confidence [0.0-1.0]
    entropy: float            # Shannon entropy of response window
    fallback_text: str        # Grounded replacement if contradiction
    hedge_suffix: str         # Epistemic hedge string if unverified
    latency_ms: float
    judge_dispatched: bool = False
    claims_extracted: List[str] = field(default_factory=list)
    judge_job_id: Optional[str] = None   # job_id for the real AI judge (retrievable from ai_judge.py)


# ---------------------------------------------------------------------------
# Entropy-Based Uncertainty Detector
# ---------------------------------------------------------------------------

def _compute_shannon_entropy(text: str) -> float:
    """
    Compute word-level Shannon entropy as uncertainty proxy.
    High entropy (>1.8) => likely unverified/ungrounded claim.
    """
    words = re.findall(r"\b[a-z]{2,}\b", text.lower())
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return round(entropy, 4)


# ---------------------------------------------------------------------------
# Lightweight NLI Contradiction Heuristics (no model inference)
# ---------------------------------------------------------------------------

# Contradiction indicator phrases
_CONTRADICTION_SIGNALS = [
    r"\b(however|but|although|nevertheless|contrary|contradicts?|conflicts?|disagrees?)\b",
    r"\bnot\s+(true|correct|accurate|valid|right|supported)\b",
    r"\b(false|incorrect|wrong|mistaken|erroneous)\b",
    r"\b(actually|in\s+fact|the\s+truth\s+is|to\s+be\s+clear)\b",
    r"\bdon\'t\s+(believe|think|agree)\b",
]
_CONTRADICTION_RE = re.compile("|".join(_CONTRADICTION_SIGNALS), re.I)

# Quantitative claim patterns (often hallucinated)
_FACTUAL_CLAIM_RE = re.compile(
    r"\b\d+(?:\.\d+)?[\s]*(percent|%|million|billion|times|years|days|km|miles|kg|dollars)\b", re.I
)

# Hedge phrases for epistemic hedging output
_HEDGE_PHRASES = [
    " [Note: This claim has not been verified against enterprise knowledge base.]",
    " [Caution: Unverified assertion — please cross-reference with authoritative source.]",
    " [Advisory: Confidence level uncertain — human review recommended.]",
]


def _extract_quantitative_claims(text: str) -> List[str]:
    """Extract factual/quantitative claims that may need grounding."""
    return _FACTUAL_CLAIM_RE.findall(text)


def _score_contradiction(text: str, premise: Optional[str] = None) -> float:
    """
    Heuristic contradiction score [0.0-1.0].
    Combines lexical signal density + negation pattern count + quantitative claim density.
    """
    signals = _CONTRADICTION_RE.findall(text)
    signal_density = min(len(signals) / max(len(text.split()), 1) * 10, 1.0)
    claims = _FACTUAL_CLAIM_RE.findall(text)
    claim_boost = min(len(claims) * 0.08, 0.25) if len(signals) > 0 else 0.0

    # Negation boost if premise is provided
    negation_boost = 0.0
    if premise:
        premise_keywords = set(re.findall(r"\b[a-z]{4,}\b", premise.lower()))
        response_keywords = set(re.findall(r"\b[a-z]{4,}\b", text.lower()))
        overlap = len(premise_keywords & response_keywords)
        negation_count = len(re.findall(r"\bnot?\b|\bno\b|\bnever\b", text, re.I))
        negation_boost = min(negation_count * 0.15 * (overlap / max(len(premise_keywords), 1)), 0.4)

    return round(min(signal_density + claim_boost + negation_boost, 1.0), 4)


# ---------------------------------------------------------------------------
# Background AI-as-Judge Dispatcher
# ---------------------------------------------------------------------------

class AsyncJudgeDispatcher:
    """
    Real AI-as-Judge dispatcher backed by Groq LLM.
    Sends completed responses for deep LLM evaluation without blocking streaming tokens.
    The background thread calls the actual Groq API and stores a structured JudgeVerdict.
    Falls back to heuristic scores if the API is unavailable.
    """

    def __init__(self):
        self._pending: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def dispatch(
        self,
        response_text: str,
        session_id: str,
        on_complete: Optional[Callable[[str, Dict], None]] = None,
    ) -> str:
        """Fire-and-forget: dispatches a real Groq LLM evaluation in a background daemon thread."""
        job_id = hashlib.sha256(f"{session_id}{time.time_ns()}".encode()).hexdigest()[:12]

        def _background_evaluate():
            # --- REAL AI JUDGE: calls Groq LLM instead of simulating ---
            try:
                from controlplane.ai_judge import get_ai_judge
                judge = get_ai_judge()
                verdict = judge._call_groq_judge(response_text, job_id, session_id)
                result = {
                    "job_id": verdict.job_id,
                    "session_id": verdict.session_id,
                    "hallucination_score": verdict.hallucination_score,
                    "factual_consistency": max(0.0, 1.0 - verdict.contradiction_score),
                    "contradiction_score": verdict.contradiction_score,
                    "bias_score": verdict.bias_score,
                    "reasoning": verdict.reasoning,
                    "model_used": verdict.model_used,
                    "completed_at": verdict.completed_at,
                    "success": verdict.success,
                    "error": verdict.error,
                }
            except Exception as e:
                # Graceful fallback to heuristic scores if judge is unavailable
                result = {
                    "job_id": job_id,
                    "session_id": session_id,
                    "hallucination_score": self._heuristic_hallucination_score(response_text),
                    "factual_consistency": self._heuristic_factual_score(response_text),
                    "contradiction_score": 0.0,
                    "bias_score": 0.0,
                    "reasoning": f"Fallback heuristic (judge error: {str(e)[:60]})",
                    "model_used": "heuristic_fallback",
                    "completed_at": time.time(),
                    "success": False,
                    "error": str(e),
                }
            with self._lock:
                self._pending[job_id] = result
            if on_complete:
                on_complete(job_id, result)

        thread = threading.Thread(target=_background_evaluate, daemon=True, name=f"ai-judge-{job_id[:6]}")
        thread.start()
        return job_id

    @staticmethod
    def _heuristic_hallucination_score(text: str) -> float:
        """Fallback heuristic hallucination score when LLM judge is unavailable."""
        claims = _FACTUAL_CLAIM_RE.findall(text)
        return min(len(claims) * 0.15, 1.0)

    @staticmethod
    def _heuristic_factual_score(text: str) -> float:
        """Fallback heuristic factual consistency when LLM judge is unavailable."""
        entropy = _compute_shannon_entropy(text)
        return max(0.0, 1.0 - (entropy / 5.0))

    def get_result(self, job_id: str) -> Optional[Dict]:
        with self._lock:
            return self._pending.get(job_id)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)


# Singleton dispatcher
_judge_dispatcher = AsyncJudgeDispatcher()


# ---------------------------------------------------------------------------
# NLI Engine
# ---------------------------------------------------------------------------

class NLIEngine:
    """
    15-token sliding-window NLI contradiction engine.
    Runs concurrently in the egress interceptor pipeline.
    Dispatches AI-as-Judge asynchronously when claims are detected.
    """

    def __init__(
        self,
        contradiction_threshold: float = 0.65,
        entropy_threshold: float = 1.8,
        enable_async_judge: bool = True,
    ):
        self._contradiction_threshold = contradiction_threshold
        self._entropy_threshold = entropy_threshold
        self._async_judge = enable_async_judge
        self._dispatcher = _judge_dispatcher

    def evaluate(
        self,
        window_text: str,
        premise: Optional[str] = None,
        session_id: str = "default",
        policy_name: str = "default",
    ) -> NLIResult:
        t0 = time.perf_counter()

        # 1. Contradiction scoring
        contradiction_score = _score_contradiction(window_text, premise)
        is_contradiction = contradiction_score >= self._contradiction_threshold

        # 2. Shannon entropy for hallucination/unverified claim detection
        claims = _extract_quantitative_claims(window_text)
        entropy = _compute_shannon_entropy(window_text)
        
        # Hedging triggers on high ungrounded speculative entropy or quantitative unverified assertions
        needs_hedging = (entropy >= self._entropy_threshold and bool(claims)) or (entropy >= 8.2)

        # 3. Hard block for extreme contradiction (>0.90) with factual claims
        hard_block = contradiction_score >= 0.90 and len(claims) > 2

        # 4. Generate fallback text (contradiction replacement)
        if is_contradiction:
            fallback_text = (
                "[GOVERNANCE INTERCEPTED: This response contains contradictory information. "
                "Please consult the authoritative enterprise knowledge base for verified information.]"
            )
        else:
            fallback_text = window_text

        # 5. Hedge suffix
        hedge_idx = min(int(entropy * 10) % len(_HEDGE_PHRASES), len(_HEDGE_PHRASES) - 1)
        hedge_suffix = _HEDGE_PHRASES[hedge_idx] if needs_hedging else ""

        # 6. Dispatch real AI-as-Judge for deep LLM evaluation (non-blocking, zero stream overhead)
        judge_dispatched = False
        judge_job_id: Optional[str] = None
        if self._async_judge and (is_contradiction or needs_hedging or claims):
            judge_job_id = self._dispatcher.dispatch(window_text, session_id)
            judge_dispatched = True

        latency_ms = (time.perf_counter() - t0) * 1000

        return NLIResult(
            is_contradiction=is_contradiction,
            needs_hedging=needs_hedging,
            hard_block=hard_block,
            score=contradiction_score,
            entropy=entropy,
            fallback_text=fallback_text,
            hedge_suffix=hedge_suffix,
            latency_ms=round(latency_ms, 3),
            judge_dispatched=judge_dispatched,
            claims_extracted=claims,
            judge_job_id=judge_job_id,
        )

    def update_thresholds(self, contradiction: float, entropy: float) -> None:
        self._contradiction_threshold = contradiction
        self._entropy_threshold = entropy


def get_judge_dispatcher() -> AsyncJudgeDispatcher:
    return _judge_dispatcher
