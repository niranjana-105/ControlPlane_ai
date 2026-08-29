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
    Combines lexical signal density + negation pattern count.
    """
    signals = _CONTRADICTION_RE.findall(text)
    signal_density = min(len(signals) / max(len(text.split()), 1) * 10, 1.0)

    # Negation boost if premise is provided
    negation_boost = 0.0
    if premise:
        premise_keywords = set(re.findall(r"\b[a-z]{4,}\b", premise.lower()))
        response_keywords = set(re.findall(r"\b[a-z]{4,}\b", text.lower()))
        overlap = len(premise_keywords & response_keywords)
        negation_count = len(re.findall(r"\bnot?\b|\bno\b|\bnever\b", text, re.I))
        negation_boost = min(negation_count * 0.15 * (overlap / max(len(premise_keywords), 1)), 0.4)

    return round(min(signal_density + negation_boost, 1.0), 4)


# ---------------------------------------------------------------------------
# Background AI-as-Judge Dispatcher
# ---------------------------------------------------------------------------

class AsyncJudgeDispatcher:
    """
    Non-blocking background AI-as-Judge dispatcher.
    Sends completed responses for deep evaluation without blocking streaming tokens.
    Stores results in a thread-safe callback registry.
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
        """Fire-and-forget: dispatch a background deep evaluation."""
        job_id = hashlib.sha256(f"{session_id}{time.time_ns()}".encode()).hexdigest()[:12]

        def _background_evaluate():
            time.sleep(0.05)  # Simulate async network call to judge LLM
            result = {
                "job_id": job_id,
                "session_id": session_id,
                "hallucination_score": self._simulate_judge_score(response_text),
                "factual_consistency": self._simulate_factual_score(response_text),
                "completed_at": time.time(),
            }
            with self._lock:
                self._pending[job_id] = result
            if on_complete:
                on_complete(job_id, result)

        thread = threading.Thread(target=_background_evaluate, daemon=True)
        thread.start()
        return job_id

    @staticmethod
    def _simulate_judge_score(text: str) -> float:
        """Simulated hallucination score (0=grounded, 1=hallucinated)."""
        claims = _FACTUAL_CLAIM_RE.findall(text)
        return min(len(claims) * 0.15, 1.0)

    @staticmethod
    def _simulate_factual_score(text: str) -> float:
        """Simulated factual consistency (0=inconsistent, 1=consistent)."""
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
        entropy = _compute_shannon_entropy(window_text)
        needs_hedging = entropy >= self._entropy_threshold

        # 3. Hard block for extreme contradiction (>0.90) with factual claims
        claims = _extract_quantitative_claims(window_text)
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

        # 6. Dispatch async judge for deep evaluation (non-blocking)
        judge_dispatched = False
        if self._async_judge and (is_contradiction or needs_hedging or claims):
            self._dispatcher.dispatch(window_text, session_id)
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
        )

    def update_thresholds(self, contradiction: float, entropy: float) -> None:
        self._contradiction_threshold = contradiction
        self._entropy_threshold = entropy


def get_judge_dispatcher() -> AsyncJudgeDispatcher:
    return _judge_dispatcher
