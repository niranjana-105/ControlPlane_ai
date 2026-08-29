"""
ControlPlane.ai - Session State and Multi-Turn Risk Aggregator
Tracks cumulative risk across conversation turns.
"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class EscalationLevel(str, Enum):
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    CRITICAL = "CRITICAL"
    BLOCKED = "BLOCKED"


@dataclass
class TurnRiskRecord:
    turn_id: int
    timestamp: float
    prompt_hash: str
    pii_detected: bool = False
    contradiction_detected: bool = False
    bias_detected: bool = False
    unverified_claim: bool = False
    hard_block_triggered: bool = False
    turn_risk_score: float = 0.0
    flags: List[str] = field(default_factory=list)


@dataclass
class SessionRiskResult:
    session_id: str
    turn_count: int
    cumulative_risk: float
    escalation_level: EscalationLevel
    escalated: bool
    escalated_block: bool
    repeated_pii_attempts: int
    repeated_jailbreak_attempts: int
    risk_trend: str
    last_turn_flags: List[str]


class SessionStateManager:
    """
    Stateful per-session multi-turn risk accumulator.

    Risk Decay Model:
        cumulative_risk(t) = 0.6 * risk(t) + 0.3 * risk(t-1) + 0.1 * risk(t-2)

    Escalation Thresholds:
        ELEVATED  : cumulative_risk >= session_escalation_threshold
        CRITICAL  : cumulative_risk >= session_escalation_threshold + 0.20
        BLOCKED   : cumulative_risk >= 0.95 OR repeated_jailbreak_attempts >= 3
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id: str = session_id or hashlib.sha256(
            str(time.time_ns()).encode()
        ).hexdigest()[:16]
        self.turns: List[TurnRiskRecord] = []
        self._cumulative_risk: float = 0.0
        self._created_at: float = time.time()

    def record_turn(
        self,
        prompt: str,
        pii_detected: bool = False,
        contradiction_detected: bool = False,
        bias_detected: bool = False,
        unverified_claim: bool = False,
        hard_block_triggered: bool = False,
        additional_flags: Optional[List[str]] = None,
    ) -> "SessionStateManager":
        turn_risk = self._compute_turn_risk(
            pii_detected, contradiction_detected, bias_detected,
            unverified_claim, hard_block_triggered
        )
        flags: List[str] = list(additional_flags or [])
        if pii_detected: flags.append("PII")
        if contradiction_detected: flags.append("CONTRADICTION")
        if bias_detected: flags.append("BIAS")
        if unverified_claim: flags.append("UNVERIFIED_CLAIM")
        if hard_block_triggered: flags.append("HARD_BLOCK")

        record = TurnRiskRecord(
            turn_id=len(self.turns),
            timestamp=time.time(),
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:12],
            pii_detected=pii_detected,
            contradiction_detected=contradiction_detected,
            bias_detected=bias_detected,
            unverified_claim=unverified_claim,
            hard_block_triggered=hard_block_triggered,
            turn_risk_score=turn_risk,
            flags=flags,
        )
        self.turns.append(record)
        self._update_cumulative_risk()
        return self

    def evaluate(self, escalation_threshold: float = 0.70) -> SessionRiskResult:
        cumulative = self._cumulative_risk
        repeated_pii = sum(1 for t in self.turns if t.pii_detected)
        repeated_jailbreak = sum(1 for t in self.turns if t.hard_block_triggered)
        last_flags = self.turns[-1].flags if self.turns else []

        if cumulative >= 0.95 or repeated_jailbreak >= 3:
            level = EscalationLevel.BLOCKED
        elif cumulative >= escalation_threshold + 0.20:
            level = EscalationLevel.CRITICAL
        elif cumulative >= escalation_threshold:
            level = EscalationLevel.ELEVATED
        else:
            level = EscalationLevel.NORMAL

        escalated = level in (EscalationLevel.ELEVATED, EscalationLevel.CRITICAL, EscalationLevel.BLOCKED)
        escalated_block = level in (EscalationLevel.CRITICAL, EscalationLevel.BLOCKED)

        risk_trend = "STABLE"
        if len(self.turns) >= 2:
            delta = self.turns[-1].turn_risk_score - self.turns[-2].turn_risk_score
            if delta > 0.10: risk_trend = "RISING"
            elif delta < -0.10: risk_trend = "FALLING"

        return SessionRiskResult(
            session_id=self.session_id,
            turn_count=len(self.turns),
            cumulative_risk=round(cumulative, 4),
            escalation_level=level,
            escalated=escalated,
            escalated_block=escalated_block,
            repeated_pii_attempts=repeated_pii,
            repeated_jailbreak_attempts=repeated_jailbreak,
            risk_trend=risk_trend,
            last_turn_flags=last_flags,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self._created_at,
            "turn_count": len(self.turns),
            "cumulative_risk": self._cumulative_risk,
            "turns": [
                {
                    "turn_id": t.turn_id, "timestamp": t.timestamp,
                    "prompt_hash": t.prompt_hash, "turn_risk_score": t.turn_risk_score,
                    "flags": t.flags,
                }
                for t in self.turns
            ],
        }

    def reset(self):
        self.turns = []
        self._cumulative_risk = 0.0

    @staticmethod
    def _compute_turn_risk(pii, contradiction, bias, unverified, hard_block) -> float:
        score = 0.0
        if hard_block:    score += 0.80
        if pii:           score += 0.35
        if contradiction: score += 0.30
        if bias:          score += 0.20
        if unverified:    score += 0.10
        return min(score, 1.0)

    def _update_cumulative_risk(self):
        n = len(self.turns)
        weights = [0.60, 0.30, 0.10]
        scores = [self.turns[-(i + 1)].turn_risk_score for i in range(min(n, 3))]
        self._cumulative_risk = sum(w * s for w, s in zip(weights, scores))


_session_registry: Dict[str, SessionStateManager] = {}


def get_or_create_session(session_id: Optional[str] = None) -> SessionStateManager:
    if session_id and session_id in _session_registry:
        return _session_registry[session_id]
    mgr = SessionStateManager(session_id)
    _session_registry[mgr.session_id] = mgr
    return mgr


def list_sessions() -> List[Dict[str, Any]]:
    return [
        {"session_id": sid, "turn_count": len(mgr.turns), "cumulative_risk": mgr._cumulative_risk}
        for sid, mgr in _session_registry.items()
    ]
