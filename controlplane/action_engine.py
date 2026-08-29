"""
ControlPlane.ai - Tier 2 Composable Action Engine
Distinct risk category resolver, composable text transformation pipeline,
and category-accurate ActionType classification.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from enum import Enum

from controlplane.nli_engine import NLIResult
from controlplane.pii_redactor import PIIResult
from controlplane.bias_detector import BiasResult
from controlplane.session_state import SessionRiskResult
from controlplane.config import PolicyProfile


# ---------------------------------------------------------------------------
# Action Types & Result
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    PASSTHROUGH        = "PASSTHROUGH"
    REDACT_PII         = "REDACT_PII"
    CASCADE_FALLBACK   = "CASCADE_FALLBACK"
    BIAS_NEUTRALIZE    = "BIAS_NEUTRALIZE"
    HEDGE_UNVERIFIED   = "HEDGE_UNVERIFIED"
    COMPOSITE_GOVERNED = "COMPOSITE_GOVERNED"
    HARD_BLOCK         = "HARD_BLOCK"


@dataclass
class ActionResult:
    transformed_text: str          # Fully composed and governed text
    action_type: ActionType        # Category-accurate primary action
    triggered_flags: List[str]     # All raw flags: ["PII_SSN", "PII_EMAIL", "CONTRADICTION"]
    risk_categories: List[str]     # Distinct categories: ["PII", "CONTRADICTION"]
    latency_ms: float              # Interception latency
    audit_payload: Dict            # Complete multi-dimensional audit record


# ---------------------------------------------------------------------------
# Action Engine
# ---------------------------------------------------------------------------

def resolve_actions(
    window_text: str,
    nli_res: NLIResult,
    pii_res: PIIResult,
    bias_res: BiasResult,
    session_res: SessionRiskResult,
    policy: PolicyProfile,
) -> ActionResult:
    """
    Tier 2 Composable Action Resolver.

    1. Collect ALL concurrent evaluation flags + distinct risk categories
    2. Hard block check (only early exit)
    3. Composable stream transformations (NO short-circuiting)
    4. Resolve category-accurate primary ActionType
    5. Return ActionResult with full audit payload
    """
    t0 = time.perf_counter()

    triggered_flags: List[str] = []
    risk_categories: Set[str] = set()

    # -------------------------------------------------------------------
    # Step 1: Collect all flags and distinct categories
    # -------------------------------------------------------------------
    if policy.enable_pii_redaction and pii_res.detected:
        triggered_flags.extend([f"PII_{t}" for t in pii_res.pii_types])
        risk_categories.add("PII")

    if policy.enable_nli_grounding and nli_res.is_contradiction:
        triggered_flags.append("CONTRADICTION")
        risk_categories.add("CONTRADICTION")

    if policy.enable_bias_detection and bias_res.is_biased:
        triggered_flags.extend([f"BIAS_{b}" for b in bias_res.categories])
        risk_categories.add("BIAS")

    if policy.enable_epistemic_hedging and nli_res.needs_hedging:
        triggered_flags.append("UNVERIFIED_CLAIM")
        risk_categories.add("UNVERIFIED")

    if policy.enable_session_risk_tracking and session_res.escalated:
        triggered_flags.append("SESSION_RISK_ESCALATION")

    # Guaranteed full audit payload capturing all concurrent signals
    audit_payload = {
        "flags":                    triggered_flags[:],
        "categories":               list(risk_categories),
        "nli_contradiction_score":  nli_res.score,
        "nli_entropy":              nli_res.entropy,
        "bias_score":               bias_res.score,
        "bias_categories":          bias_res.categories,
        "pii_types":                pii_res.pii_types,
        "pii_match_count":          pii_res.match_count,
        "session_id":               session_res.session_id,
        "session_turn_count":       session_res.turn_count,
        "session_cumulative_risk":  session_res.cumulative_risk,
        "session_escalation_level": session_res.escalation_level.value,
        "risk_trend":               session_res.risk_trend,
        "policy_id":                policy.name.value,
        "jurisdiction":             policy.jurisdiction.value,
        "judge_dispatched":         nli_res.judge_dispatched,
        "claims_extracted":         nli_res.claims_extracted,
    }

    # -------------------------------------------------------------------
    # Step 2: Hard Block — The ONLY early termination
    # -------------------------------------------------------------------
    if nli_res.hard_block or bias_res.severe_breach or session_res.escalated_block:
        triggered_flags.append("HARD_BLOCK")
        risk_categories.add("SAFETY_BLOCK")
        audit_payload["flags"] = triggered_flags
        audit_payload["categories"] = list(risk_categories)
        latency_ms = (time.perf_counter() - t0) * 1000 + max(
            pii_res.latency_ms, nli_res.latency_ms, bias_res.latency_ms
        )
        return ActionResult(
            transformed_text="[REQUEST TERMINATED: Content violated enterprise safety policy. This incident has been logged.]",
            action_type=ActionType.HARD_BLOCK,
            triggered_flags=triggered_flags,
            risk_categories=list(risk_categories),
            latency_ms=round(latency_ms, 3),
            audit_payload=audit_payload,
        )

    # -------------------------------------------------------------------
    # Step 3: Composable Stream Transformations (no short-circuiting)
    # -------------------------------------------------------------------
    current_text = window_text

    # Step A: In-flight PII Masking
    if policy.enable_pii_redaction and pii_res.detected:
        current_text = pii_res.redacted_text

    # Step B: Contradiction Replacement (substitute hallucinated clause)
    if policy.enable_nli_grounding and nli_res.is_contradiction:
        current_text = nli_res.fallback_text

    # Step C: Bias Neutralization (applied on top of current text)
    if policy.enable_bias_detection and bias_res.is_biased:
        current_text = bias_res.apply_neutralizer(current_text)

    # Step D: Epistemic Hedging for unverified/no-ground-truth assertions
    if policy.enable_epistemic_hedging and nli_res.needs_hedging:
        current_text += nli_res.hedge_suffix

    # -------------------------------------------------------------------
    # Step 4: Resolve Category-Accurate Primary Action Type
    # -------------------------------------------------------------------
    num_categories = len(risk_categories)
    if num_categories == 0:
        action_type = ActionType.PASSTHROUGH
    elif num_categories == 1:
        if "PII" in risk_categories:
            action_type = ActionType.REDACT_PII
        elif "CONTRADICTION" in risk_categories:
            action_type = ActionType.CASCADE_FALLBACK
        elif "BIAS" in risk_categories:
            action_type = ActionType.BIAS_NEUTRALIZE
        elif "UNVERIFIED" in risk_categories:
            action_type = ActionType.HEDGE_UNVERIFIED
        else:
            action_type = ActionType.PASSTHROUGH
    else:
        action_type = ActionType.COMPOSITE_GOVERNED

    latency_ms = (time.perf_counter() - t0) * 1000 + max(
        pii_res.latency_ms, nli_res.latency_ms, bias_res.latency_ms
    )

    return ActionResult(
        transformed_text=current_text,
        action_type=action_type,
        triggered_flags=triggered_flags,
        risk_categories=list(risk_categories),
        latency_ms=round(latency_ms, 3),
        audit_payload=audit_payload,
    )
