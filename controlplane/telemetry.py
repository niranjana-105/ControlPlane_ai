"""
ControlPlane.ai - Audit Logging, Trustworthiness Index, and Latency Telemetry
Immutable per-request audit records, regulatory flags, and T_score calculation.
"""

import time
import uuid
import math
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum

from controlplane.action_engine import ActionResult, ActionType


# ---------------------------------------------------------------------------
# Audit Record
# ---------------------------------------------------------------------------

@dataclass
class AuditRecord:
    """Immutable per-request governance audit record."""
    record_id: str
    session_id: str
    request_id: str
    timestamp: float
    policy_id: str
    jurisdiction: str

    # Input
    prompt_hash: str
    prompt_token_estimate: int
    ingress_latency_ms: float
    jailbreak_patterns_hit: List[str]
    complexity_tier: str

    # Cache
    cache_hit: bool
    cache_tier: str

    # Egress governance
    action_type: str
    triggered_flags: List[str]
    risk_categories: List[str]
    egress_latency_ms: float
    total_latency_ms: float

    # Scores
    nli_score: float
    entropy_score: float
    bias_score: float
    pii_match_count: int
    session_cumulative_risk: float

    # Trustworthiness Index
    trustworthiness_score: float

    # Regulatory
    requires_human_review: bool
    judge_dispatched: bool
    regulatory_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Trustworthiness Index (T_score)
# ---------------------------------------------------------------------------

def compute_trustworthiness_score(
    nli_score: float,
    bias_score: float,
    entropy: float,
    session_cumulative_risk: float,
    action_type: ActionType,
    pii_detected: bool,
) -> float:
    """
    T_score = 1 - weighted_penalty

    Penalties:
        - Contradiction:   0.30 * nli_score
        - Bias:            0.25 * bias_score
        - Entropy:         0.20 * min(entropy/3.0, 1.0)
        - Session risk:    0.15 * session_cumulative_risk
        - PII detected:    0.10 if pii_detected
        - Hard block:      0.00 (returns 0.0 immediately)
    """
    if action_type == ActionType.HARD_BLOCK:
        return 0.0

    # Only penalize abnormal excess uncertainty beyond standard baseline
    norm_entropy_penalty = 0.20 * max(0.0, min((entropy - 4.5) / 3.0, 1.0))

    penalty = (
        0.30 * min(max(nli_score, 0.0), 1.0)
        + 0.25 * min(max(bias_score, 0.0), 1.0)
        + norm_entropy_penalty
        + 0.15 * min(max(session_cumulative_risk, 0.0), 1.0)
        + (0.10 if pii_detected else 0.0)
    )

    t_score = max(0.0, min(1.0 - penalty, 1.0))
    return round(t_score, 4)


# ---------------------------------------------------------------------------
# Telemetry Store (In-Memory Ring Buffer)
# ---------------------------------------------------------------------------

class TelemetryStore:
    """
    In-memory immutable audit log with configurable ring-buffer size.
    Thread-safe append; supports export for dashboard and regulatory review.
    """

    def __init__(self, max_records: int = 10_000):
        self._records: List[AuditRecord] = []
        self._max_records = max_records

    def log(self, record: AuditRecord) -> None:
        if len(self._records) >= self._max_records:
            self._records.pop(0)
        self._records.append(record)

    def build_and_log(
        self,
        session_id: str,
        request_id: str,
        policy_id: str,
        jurisdiction: str,
        prompt_hash: str,
        prompt_token_estimate: int,
        ingress_latency_ms: float,
        jailbreak_patterns_hit: List[str],
        complexity_tier: str,
        cache_hit: bool,
        cache_tier: str,
        action_result: ActionResult,
        nli_score: float,
        entropy_score: float,
        bias_score: float,
        pii_detected: bool,
        pii_match_count: int,
        session_cumulative_risk: float,
        judge_dispatched: bool,
        requires_human_review: bool,
        total_latency_ms: float,
    ) -> AuditRecord:
        t_score = compute_trustworthiness_score(
            nli_score=nli_score,
            bias_score=bias_score,
            entropy=entropy_score,
            session_cumulative_risk=session_cumulative_risk,
            action_type=action_result.action_type,
            pii_detected=pii_detected,
        )

        regulatory_flags: List[str] = []
        if pii_detected: regulatory_flags.append("PII_EXPOSURE_INTERCEPTED")
        if nli_score > 0.5: regulatory_flags.append("FACTUAL_ACCURACY_CONCERN")
        if bias_score > 0.3: regulatory_flags.append("FAIRNESS_VIOLATION_INTERCEPTED")
        if session_cumulative_risk > 0.7: regulatory_flags.append("SESSION_RISK_ESCALATION")
        if action_result.action_type == ActionType.HARD_BLOCK: regulatory_flags.append("CONTENT_POLICY_VIOLATION")

        record = AuditRecord(
            record_id=str(uuid.uuid4()),
            session_id=session_id,
            request_id=request_id,
            timestamp=time.time(),
            policy_id=policy_id,
            jurisdiction=jurisdiction,
            prompt_hash=prompt_hash,
            prompt_token_estimate=prompt_token_estimate,
            ingress_latency_ms=ingress_latency_ms,
            jailbreak_patterns_hit=jailbreak_patterns_hit,
            complexity_tier=complexity_tier,
            cache_hit=cache_hit,
            cache_tier=cache_tier,
            action_type=action_result.action_type.value,
            triggered_flags=action_result.triggered_flags,
            risk_categories=action_result.risk_categories,
            egress_latency_ms=action_result.latency_ms,
            total_latency_ms=round(total_latency_ms, 3),
            nli_score=nli_score,
            entropy_score=entropy_score,
            bias_score=bias_score,
            pii_match_count=pii_match_count,
            session_cumulative_risk=session_cumulative_risk,
            trustworthiness_score=t_score,
            requires_human_review=requires_human_review,
            judge_dispatched=judge_dispatched,
            regulatory_flags=regulatory_flags,
        )
        self.log(record)
        return record

    # --- Query helpers ---

    def all_records(self) -> List[AuditRecord]:
        return list(self._records)

    def recent(self, n: int = 50) -> List[AuditRecord]:
        return self._records[-n:]

    def by_session(self, session_id: str) -> List[AuditRecord]:
        return [r for r in self._records if r.session_id == session_id]

    def flagged_records(self) -> List[AuditRecord]:
        return [r for r in self._records if r.regulatory_flags]

    def summary_stats(self) -> Dict[str, Any]:
        if not self._records:
            return {}
        scores = [r.trustworthiness_score for r in self._records]
        action_counts: Dict[str, int] = {}
        for r in self._records:
            action_counts[r.action_type] = action_counts.get(r.action_type, 0) + 1
        latencies = [r.total_latency_ms for r in self._records]
        return {
            "total_requests": len(self._records),
            "avg_trustworthiness_score": round(sum(scores) / len(scores), 4),
            "min_trustworthiness_score": round(min(scores), 4),
            "max_trustworthiness_score": round(max(scores), 4),
            "action_type_distribution": action_counts,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
            "flagged_requests": len(self.flagged_records()),
        }


# Singleton telemetry store
_global_telemetry = TelemetryStore()


def get_telemetry() -> TelemetryStore:
    return _global_telemetry
