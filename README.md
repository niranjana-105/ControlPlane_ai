# ControlPlane.ai

ControlPlane.ai is a real-time, in-flight AI governance layer and streaming reverse proxy. It intercepts Large Language Model (LLM) token streams in-memory to detect and mitigate privacy leaks (PII/PHI), factual hallucinations, demographic bias, and adversarial prompt injections in **under 5 milliseconds**—well within strict enterprise latency budgets (<20ms).

For a complete breakdown of the business case, financial ROI, and regulatory analysis, see the [Business Proposal](BUSINESS_PROPOSAL.md).

Track bug reports, feature suggestions, or view live commits in the [GitHub Repository](https://github.com/niranjana-105/ControlPlane_ai).

---

## Table of contents

- [Requirements](#requirements)
- [Recommended integrations](#recommended-integrations)
- [Installation](#installation)
- [Configuration](#configuration)
- [Architecture & Governance Pipeline](#architecture--governance-pipeline)
- [Policy Profiles & Jurisdictions](#policy-profiles--jurisdictions)
- [Benchmark Suite (44 Scenarios)](#benchmark-suite-44-scenarios)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Maintainers & Competition Submission](#maintainers--competition-submission)

---

## Requirements

ControlPlane.ai requires:

- **Python 3.10+** (tested on Python 3.11 and 3.12)
- **Groq Cloud API Key** (for real-time streaming LLM inference)
- The following core Python packages (specified in `requirements.txt`):
  - `streamlit >= 1.30.0`
  - `fastapi >= 0.104.0`
  - `uvicorn >= 0.24.0`
  - `httpx >= 0.25.0`
  - `pandas >= 2.0.0`
  - `plotly >= 5.18.0`
  - `scikit-learn >= 1.3.0`
  - `pydantic >= 2.0.0`

---

## Recommended integrations

- **Groq API Cloud**: Provides ultra-fast inference (<200ms TTFT) for live streaming interception demonstrations.
- **Enterprise SIEM / Splunk / Datadog**: ControlPlane.ai exposes an immutable audit telemetry log (`controlplane/telemetry.py`) formatted for automated ingestion into enterprise SIEM platforms.
- **Redis Cluster**: Recommended for multi-node deployments of the L1/L2 Hierarchical Semantic Cache in production.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/niranjana-105/ControlPlane_ai.git
   cd ControlPlane_ai
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   # On Windows:
   .\env\Scripts\activate
   # On Linux/macOS:
   source env/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure your environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and insert your `GROQ_API_KEY`:
   ```ini
   GROQ_API_KEY=gsk_your_actual_groq_api_key_here
   ```

---

## Configuration

### Starting the Application

Launch the unified governance dashboard and API proxy with a single command:
```bash
python run.py
```

* **Interactive Governance Console**: `http://localhost:8501`
* **FastAPI Reverse Proxy Gateway**: `http://localhost:8000`
* **OpenAPI Documentation**: `http://localhost:8000/docs`
* **Gateway Health Check**: `http://localhost:8000/health`

---

## Architecture & Governance Pipeline

ControlPlane.ai operates as an inline reverse proxy and stream interceptor operating across three non-blocking tiers:

```
User Prompt ──▶ [Tier 0: DFA Ingress Gate (<0.4ms)]
                      │ (Blocks Jailbreaks / Injections)
                      ▼
               [Live Upstream LLM: Groq API]
                      │ (Streaming SSE Tokens)
                      ▼
        [Tier 1: Parallel Egress Interceptor (<5ms)]
         ├─ DFA PII/PHI Redactor (15 Regex Patterns)
         ├─ Sliding-Window NLI Engine (Contradiction Scoring)
         ├─ Multi-Category Bias Detector (Gender/Age/Race)
         └─ Shannon Entropy Uncertainty Estimator
                      │
                      ▼
        [Tier 2: Composable Action Engine]
         ├─ PII Masking ([REDACTED_...])
         ├─ Contradiction Replacement (Cascade Fallback)
         ├─ Bias Neutralization (String Substitution)
         └─ Epistemic Hedging ([Caution: Unverified...])
                      │
                      ▼
        Governed Clean Output Delivered to End User + Immutable Audit Log
```

### Key Technical Modules

* `controlplane/ingress.py`: Tier 0 DFA sanitizer evaluating prompt complexity and 8 compiled jailbreak patterns.
* `controlplane/pii_redactor.py`: 15 compiled DFA patterns for SSN, credit cards, emails, phone numbers, API keys, passwords, HIPAA MRNs, and ICD-10 diagnosis codes.
* `controlplane/nli_engine.py`: 15-token sliding-window lexical contradiction detector, negative assertion booster, and Shannon entropy uncertainty calculator.
* `controlplane/bias_detector.py`: Multi-category ontology detector with in-flight string neutralizers.
* `controlplane/action_engine.py`: Composable non-short-circuiting transform engine that resolves category-accurate primary actions (`REDACT_PII`, `BIAS_NEUTRALIZE`, `CASCADE_FALLBACK`, `HEDGE_UNVERIFIED`, `HARD_BLOCK`, `COMPOSITE_GOVERNED`).
* `controlplane/cache.py`: L1 SHA-256 exact match + L2 TF-IDF cosine-similarity semantic vector cache.
* `controlplane/session_state.py`: Multi-turn risk aggregator computing $R(t) = 0.6 r(t) + 0.3 r(t-1) + 0.1 r(t-2)$.
* `controlplane/telemetry.py`: In-memory immutable audit ring buffer with Trustworthiness Index ($T_{\text{score}}$) calculations.

---

## Policy Profiles & Jurisdictions

ControlPlane.ai provides declarative, code-free policy profiles configured in `controlplane/config.py`:

| Profile | Regulatory Jurisdiction | Latency Budget | Active Protections | Primary Failure Mitigated |
| :--- | :--- | :--- | :--- | :--- |
| **Customer Support Bot** | **GDPR (EU)** | $\le 18\text{ms}$ | SSN, Credit Cards, Emails, Phone, Contextual ZIP, Anti-Toxicity | Leaking customer PII or brand-damaging toxic responses |
| **Internal Dev Copilot** | **SOC 2 (IT Security)** | $\le 12\text{ms}$ | API Keys, Admin Passwords, Database Credentials, Sub-5ms Ingress | Committing hardcoded credentials or cloud secrets |
| **Clinical & Financial Support** | **HIPAA (US Healthcare)** | $\le 22\text{ms}$ | Patient Names, Medical Record Numbers (MRN), ICD-10 Codes, Zero Contradictions | Medical malpractice liability and hallucinated clinical guidance |

---

## Benchmark Suite (44 Scenarios)

The system includes an automated 44-scenario test harness (`controlplane/benchmark_scenarios.py`) validating precision, recall, and F1 across five critical failure categories:

1. **PII & Secrets Leaks (10 Scenarios):** Direct SSNs, email leaks, credit cards, passwords, API keys, HIPAA MRNs, and ICD-10 diagnosis codes.
2. **Hallucinations & Contradictions (10 Scenarios):** Direct numerical contradictions, conflicting metrics, policy violations, and high-entropy assertions.
3. **Bias & Stereotypes (10 Scenarios):** Gender leadership stereotypes, racial generalizations, age discrimination, ableist slurs, and toxicity.
4. **Composite Violations (8 Scenarios):** Simultaneous multi-risk occurrences (e.g. PII + contradiction + bias in a single turn).
5. **Edge Cases (6 Scenarios):** Legal section numbers, technical jargon, zero-width unicode injection, and long clean inputs.

---

## Troubleshooting & FAQ

### Troubleshooting

- **The dashboard displays a connection error:**
  - Verify that `GROQ_API_KEY` is present in your `.env` file.
  - Test connectivity with `python test_client.py`.
- **Latency exceeds expected budget:**
  - Ensure you are using high-throughput Groq models (e.g. `qwen/qwen3.8-27b`, `openai/gpt-oss-120b`).
  - Verify that the L1/L2 cache is enabled in the sidebar under *Advanced Guard Settings*.

### Frequently Asked Questions (FAQ)

**Q: Does the in-flight inspection add perceptible delay to streaming responses?**  
**A:** No. ControlPlane.ai’s parallel heuristics execute in **1.1ms – 4.5ms**, which is over $4\times$ faster than human visual reading speed (~15ms/token).

**Q: Does the Human Oversight Hub (Tab 4) block the user while waiting for human review?**  
**A:** No. The end-user receives their safe, sanitized answer immediately with zero delay. Tab 4 functions as an asynchronous background audit queue for compliance officers to verify flagged high-risk responses.

---

## Maintainers & Competition Submission

**Accenture Innovation Challenge 2026 | Problem Track 1: AI Governance**  
**Team Name:** BAZOOKA  
**Institution:** Indian Institute of Technology (IIT) Guwahati (Class of 2027)  

* **Niranjana Nitin** — [GitHub](https://github.com/niranjana-105)
* **Manan Sadana** — [GitHub](https://github.com/manansadana)

---
*Copyright © 2026 Team BAZOOKA. Built for the Accenture Innovation Challenge 2026.*
