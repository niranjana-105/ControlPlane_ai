# ControlPlane.ai
## Enterprise In-Flight AI Governance Layer
**Accenture Innovation Challenge 2026 | Problem Track 1**

> Sub-20ms real-time governance for LLM streaming responses — without blocking a single token.

---

## Quick Start

`ash
pip install -r requirements.txt
python run.py
`

- **Dashboard**: http://localhost:8501
- **API Gateway**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## Architecture Overview

ControlPlane.ai operates as a **streaming reverse proxy** between your application and any LLM provider, intercepting every token window through a 3-tier governance pipeline.

`
User Request
     |
     v
[TIER 0: Ingress Gate (<5ms)]
  - DFA jailbreak sanitizer (8 patterns)
  - L1/L2 hierarchical semantic cache
  - Complexity router (SIMPLE/MODERATE/COMPLEX)
     |
     v
[TIER 1: Concurrent Egress Interceptor (15-20ms)]
  Runs SIMULTANEOUSLY in parallel:
  |---- NLI Contradiction Engine (15-token sliding window)
  |---- PII/PHI Redactor (DFA regex, jurisdiction-aware)
  |---- Bias & Fairness Detector (5 ontology categories)
  |---- Async AI-as-Judge (non-blocking background dispatch)
     |
     v
[TIER 2: Composable Action Engine]
  - Collects ALL flags (no short-circuit)
  - Tracks distinct risk categories
  - Composes transformations: PII -> Fallback -> Bias -> Hedge
  - Returns category-accurate ActionType
     |
     v
Governed Streaming Response + Audit Record
`

---

## Governance Pipeline Details

### Tier 0 — Ingress Gate (<5ms)
| Component | Mechanism | Latency |
|---|---|---|
| Jailbreak Sanitizer | DFA regex (8 patterns: DAN, sudo, prompt injection, etc.) | <1ms |
| L1 Exact Cache | SHA-256 keyed LRU dictionary | <0.1ms |
| L2 Semantic Cache | TF-IDF cosine similarity (pure Python) | 1-3ms |
| Complexity Router | Heuristic token/structure scoring | <0.5ms |

### Tier 1 — Egress Interceptor (15-20ms concurrent)
| Component | Mechanism | Latency |
|---|---|---|
| NLI Engine | Lexical contradiction signals + Shannon entropy | 3-8ms |
| PII Redactor | 15 DFA regex patterns (SSN, CC, email, MRN, API keys...) | 2-5ms |
| Bias Detector | 5-category ontology (gender, race, age, disability, toxicity) | 2-5ms |
| Async Judge | Background thread dispatch (does NOT block streaming) | 0ms overhead |

### Tier 2 — Action Engine
| ActionType | Trigger Condition |
|---|---|
| PASSTHROUGH | No risk categories detected |
| REDACT_PII | Only PII detected |
| CASCADE_FALLBACK | Only contradiction detected |
| BIAS_NEUTRALIZE | Only bias detected |
| HEDGE_UNVERIFIED | Only unverified claim detected |
| COMPOSITE_GOVERNED | 2+ distinct risk categories |
| HARD_BLOCK | Toxicity, jailbreak, or critical session escalation |

---

## Action Types & Tradeoffs

| Dimension | Handled By |
|---|---|
| Different risk tolerance per use case | config.py — Tiered profiles |
| Overlapping risks | action_engine.py — Composable, no short-circuit |
| No reliable ground truth | nli_engine.py — Entropy + async AI-as-Judge |
| Over/under-flagging tradeoff | benchmark_scenarios.py — 44 scenarios + sweep |
| Multi-turn compounding risk | session_state.py — Weighted decay model |
| Regulatory/jurisdictional variation | config.py — EU AI Act, HIPAA, GDPR |

---

## Policy Profiles

### Customer Support Bot (GDPR_EU)
- Latency SLA: Ingress <5ms, Egress <18ms
- NLI threshold: 0.60 | Bias threshold: 0.25
- Entities: SSN, CC, Email, Phone, Passport, API Key, Location

### Internal Knowledge & Code Copilot (BASE_SOC2)
- Latency SLA: Ingress <3ms, Egress <12ms
- NLI threshold: 0.75 | Bias threshold: 0.55
- Entities: API Key, Password, SSN, Credit Card

### Clinical & Financial Decision Support (HIPAA_US)
- Latency SLA: Ingress <5ms, Egress <22ms
- NLI threshold: 0.45 | Bias threshold: 0.20
- Human oversight: Required | Audit retention: 6 years
- Entities: SSN, MRN, Health Plan ID, Diagnosis Code, Patient Name, CC, Phone, Email

---

## File Structure

`
d:/accenture/
controlplane/
    __init__.py           # Package init
    config.py             # Policy profiles + jurisdictions
    session_state.py      # Multi-turn cumulative risk aggregator
    cache.py              # L1/L2 hierarchical semantic cache
    ingress.py            # DFA jailbreak sanitizer + complexity router
    nli_engine.py         # NLI contradiction engine + async AI-as-Judge
    bias_detector.py      # Stereotype, demographic, toxicity detector
    pii_redactor.py       # Jurisdiction-aware streaming PII/PHI masker
    action_engine.py      # Composable action resolver
    simulator.py          # Mock LLM streaming generator (8 scenarios)
    benchmark_scenarios.py # 44 benchmark tests + tradeoff sweep
    proxy_gateway.py      # FastAPI /v1/chat/completions proxy
    telemetry.py          # Audit logging + T_score + latency telemetry
app.py                    # Streamlit 5-tab dashboard
run.py                    # One-click startup
requirements.txt          # Dependencies
`

---

## Trustworthiness Index

T_{score} = 1 - (0.30 \cdot S_{NLI} + 0.25 \cdot S_{bias} + 0.20 \cdot \min(H/3, 1) + 0.15 \cdot R_{session} + 0.10 \cdot \mathbf{1}_{PII})

Where H = Shannon entropy, R = session cumulative risk.

---

## Dashboard Tabs

| Tab | Description |
|---|---|
| Stream Inspector | Live governance demo with 8 test scenarios |
| Benchmark Suite | 44-scenario precision/recall evaluation + threshold sweep |
| Policy Configurator | Side-by-side profile comparison + placement rationale |
| Observability | Audit trail + T-score + cache/latency telemetry |
| Human Feedback Hub | Override queue + weight tuner + learning curves |

---

## Requirements

`
fastapi>=0.110.0
uvicorn>=0.28.0
streamlit>=1.32.0
pydantic>=2.6.0
requests>=2.31.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.19.0
httpx>=0.27.0
aiohttp>=3.9.0
`

All governance modules use **zero external ML dependencies** — pure Python stdlib for sub-millisecond DFA pattern matching.
