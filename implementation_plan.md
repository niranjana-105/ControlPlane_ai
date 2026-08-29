# ControlPlane.ai — Enterprise In-Flight AI Governance Layer
**Accenture Innovation Challenge 2026 | Problem Track 1: ControlPlane.ai**

---

## Executive Summary & Architecture Paradigm

Modern enterprises deploy generative AI across customer support chatbots, developer copilots, and regulated decision-support workflows. Existing governance approaches either fail on latency (inline LLM-as-a-judge adds 1,200–2,500ms) or arrive too late (post-hoc batch evaluations 24 hours later).

**ControlPlane.ai** introduces a high-throughput, sub-millisecond streaming reverse proxy (`/v1/chat/completions`) with:
1. **Tier 0 Ingress Gate (<5ms)**: Session risk aggregator, DFA jailbreak sanitizer, L1/L2 hierarchical semantic cache, and complexity router.
2. **Tier 1 Concurrent Egress Interceptor (15–20ms)**: 15-token sliding-window buffer running NLI contradiction checks, zero-copy PII redactor, bias/fairness detector, and entropy monitor **concurrently in parallel**.
3. **Tier 2 Composable Action Engine & Distinct Risk Category Resolver**: Concurrently accumulates all triggered flags, tracks **distinct risk categories** for accurate metrics/action classification, and composes text transformations (PII $\to$ Fallback $\to$ Bias $\to$ Hedging) without short-circuiting.
4. **Non-Blocking No-Ground-Truth Flow**: Instant local entropy/claim extraction with inline epistemic hedging, dispatching deep AI-as-Judge evaluations **asynchronously in the background without blocking streaming tokens**.

---

## Tier 2 Action Engine: Category-Accurate Action Resolution

```python
class ActionType(str, Enum):
    PASSTHROUGH = "PASSTHROUGH"
    REDACT_PII = "REDACT_PII"
    CASCADE_FALLBACK = "CASCADE_FALLBACK"
    BIAS_NEUTRALIZE = "BIAS_NEUTRALIZE"
    HEDGE_UNVERIFIED = "HEDGE_UNVERIFIED"
    COMPOSITE_GOVERNED = "COMPOSITE_GOVERNED"
    HARD_BLOCK = "HARD_BLOCK"

@dataclass
class ActionResult:
    transformed_text: str          # Fully composed & governed text
    action_type: ActionType        # Category-accurate primary action
    triggered_flags: list[str]     # All raw flags: ["PII_SSN", "PII_EMAIL", "CONTRADICTION"]
    risk_categories: list[str]     # Distinct categories: ["PII", "CONTRADICTION"]
    latency_ms: float              # Interception latency
    audit_payload: dict            # Complete multi-dimensional audit record

def resolve_actions(
    window_text: str,
    nli_res: NLIResult,
    pii_res: PIIResult,
    bias_res: BiasResult,
    session_res: SessionRiskResult,
    policy: PolicyProfile
) -> ActionResult:
    triggered_flags: list[str] = []
    risk_categories: set[str] = set()
    
    # 1. Collect all concurrent evaluation flags & distinct categories
    if pii_res.detected:
        triggered_flags.extend([f"PII_{t}" for t in pii_res.pii_types])
        risk_categories.add("PII")
    if nli_res.is_contradiction:
        triggered_flags.append("CONTRADICTION")
        risk_categories.add("CONTRADICTION")
    if bias_res.is_biased:
        triggered_flags.extend([f"BIAS_{b}" for b in bias_res.categories])
        risk_categories.add("BIAS")
    if nli_res.needs_hedging:
        triggered_flags.append("UNVERIFIED_CLAIM")
        risk_categories.add("UNVERIFIED")
    if session_res.escalated:
        triggered_flags.append("SESSION_RISK_ESCALATION")

    # Complete audit payload is guaranteed to capture all concurrent signals
    audit_payload = {
        "flags": triggered_flags,
        "categories": list(risk_categories),
        "nli_score": nli_res.score,
        "bias_score": bias_res.score,
        "entropy_score": nli_res.entropy,
        "pii_types": pii_res.pii_types,
        "session_cumulative_risk": session_res.cumulative_risk,
        "policy_id": policy.name,
        "jurisdiction": policy.jurisdiction
    }

    # 2. Hard Block: The sole early termination (for jailbreaks, extreme toxicity, or session security breach)
    if nli_res.hard_block or bias_res.severe_breach or session_res.escalated_block:
        triggered_flags.append("HARD_BLOCK")
        risk_categories.add("SAFETY_BLOCK")
        return ActionResult(
            transformed_text="[REQUEST TERMINATED: Content violated enterprise safety policy.]",
            action_type=ActionType.HARD_BLOCK,
            triggered_flags=triggered_flags,
            risk_categories=list(risk_categories),
            latency_ms=pii_res.latency_ms,
            audit_payload=audit_payload
        )

    # 3. Composable Stream Transformation (No short-circuiting)
    current_text = window_text

    # Step A: In-flight PII Masking
    if pii_res.detected:
        current_text = pii_res.redacted_text

    # Step B: Contradiction Replacement (Substitute hallucinated clause with grounded fallback)
    if nli_res.is_contradiction:
        current_text = nli_res.fallback_text

    # Step C: Bias Neutralization (Applied on top of current text)
    if bias_res.is_biased:
        current_text = bias_res.apply_neutralizer(current_text)

    # Step D: Epistemic Hedging for unverified/no-ground-truth assertions
    if nli_res.needs_hedging:
        current_text += " [Note: Unverified enterprise claim]"

    # 4. Resolve Category-Accurate Primary Action Type
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

    return ActionResult(
        transformed_text=current_text,
        action_type=action_type,
        triggered_flags=triggered_flags,
        risk_categories=list(risk_categories),
        latency_ms=max(pii_res.latency_ms, nli_res.latency_ms, bias_res.latency_ms),
        audit_payload=audit_payload
    )
```

---

## Complete Problem Statement Requirement Traceability Matrix

| Problem Statement Dimension | Handled In Codebase | Verified In Proposal / UI |
| :--- | :--- | :--- |
| **Different risk tolerance/latency per use case** | `config.py` (Tiered latency budgets & profiles) | Tab 3 Configurator & Proposal Sec. 3 |
| **Overlapping risks (hallucination + PII + bias)** | `action_engine.py` (Category-accurate composable resolver) | Tab 1 Stream Inspector & Proposal Sec. 2 |
| **Absence of reliable ground truth** | `nli_engine.py` (Entropy & async AI-as-Judge) | Tab 1 & Tab 2 Benchmark Suite |
| **Over-flagging vs under-flagging tradeoff** | `benchmark_scenarios.py` & sensitivity sweep | Tab 2 Tradeoff visualizer matrix |
| **Multi-turn / agentic compounding risk** | `session_state.py` (Cumulative risk & escalation) | Tab 1 & Tab 4 Audit Trail |
| **Regulatory & jurisdictional variation** | `config.py` (`EU_AI_ACT`, `HIPAA_US`, `GDPR_EU`) | Tab 3 & Proposal Sec. 3 |
| **API-layer reverse proxy (zero model internal dependency)** | `proxy_gateway.py` (`/v1/chat/completions`) | Architecture Diagram & Proposal Sec. 2 |
| **Diverse detection techniques** | DFA regex, L1/L2 vector cache, NLI, bias ontology, entropy | Entire `controlplane/` package |
| **Tiered decision logic** | `ActionType` enum (passthrough/redact/fallback/hedge/block) | `action_engine.py` |
| **Architecture placement reasoning (Inline vs Gate vs Batch)** | High-throughput streaming proxy design | **`BUSINESS_PROPOSAL.md` Section 2.1** |
| **Governance & immutable audit trails** | `telemetry.py` (Per-token records & regulatory flags) | Tab 4 Observability Dashboard |
| **Feedback loops & continuous learning** | Tab 5 UI (Human override queue & weight tuner) | Tab 5 Streamlit Hub |
| **Metrics & Trustworthiness Index** | `telemetry.py` ($T_{\text{score}}$ calculation) | Tab 4 Gauge & Proposal Sec. 4 |

---

## File Structure & Module Map

```
d:/accenture/
├── controlplane/
│   ├── __init__.py
│   ├── config.py                  # Policy profiles (Customer Support, Copilot, Clinical) + Jurisdictions (EU AI Act, HIPAA, GDPR)
│   ├── session_state.py           # Multi-turn cumulative risk aggregator & agentic tool interceptor
│   ├── cache.py                   # L1 SHA-256 exact cache & L2 Semantic Vector Cache
│   ├── ingress.py                 # Ingress Sanitizer & Predictive Complexity Router
│   ├── nli_engine.py              # Premise NLI contradiction engine + async non-blocking AI-as-Judge
│   ├── bias_detector.py           # Real-time stereotype, demographic & toxicity bias detector
│   ├── pii_redactor.py            # Jurisdiction-aware streaming PII/PHI masker
│   ├── action_engine.py           # Unified composable action resolver & distinct category tracker
│   ├── proxy_gateway.py           # FastAPI reverse proxy with SSE token interceptor (/v1/chat/completions)
│   ├── telemetry.py               # Audit logging, Trustworthiness Index ($T_{score}$) & latency telemetry
│   ├── simulator.py               # High-fidelity mock LLM streaming generator
│   └── benchmark_scenarios.py     # 44 Benchmark test scenarios + tradeoff evaluation engine
├── app.py                         # Streamlit Enterprise Command Center (5 Interactive Tabs)
├── run.py                         # One-click startup script (Gateway + Streamlit)
├── requirements.txt               # Dependencies
├── README.md                      # Comprehensive documentation, architecture & run guide
└── BUSINESS_PROPOSAL.md           # Round 2 Business Proposal & Technical Whitepaper
```
