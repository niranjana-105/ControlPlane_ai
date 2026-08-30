# ControlPlane.ai

ControlPlane.ai is a real-time, in-flight AI governance layer and streaming reverse proxy
for enterprise Generative AI deployments. It intercepts Large Language Model (LLM) token
streams in-memory — detecting and sanitizing PII/PHI leaks, factual hallucinations,
demographic bias, and adversarial prompt injections — in **under 5 milliseconds**, well
within strict enterprise latency budgets (<20ms).

Built for the Accenture Innovation Challenge 2026, Problem Track 1: In-Flight AI
Governance.

For the full business case, financial ROI derivations, and regulatory analysis, see the
[Business Proposal](BUSINESS_PROPOSAL.md).

Submit bug reports and feature suggestions, or track changes in the
[issue queue](https://github.com/niranjana-105/ControlPlane_ai/issues).


## Table of contents

- [Requirements](#requirements)
- [Recommended integrations](#recommended-integrations)
- [Installation](#installation)
- [Configuration](#configuration)
- [Architecture & governance pipeline](#architecture--governance-pipeline)
- [Policy profiles & jurisdictions](#policy-profiles--jurisdictions)
- [Governance action types](#governance-action-types)
- [Benchmark suite (44 scenarios)](#benchmark-suite-44-scenarios)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Maintainers](#maintainers)


## Requirements

This project requires the following:

- **Python 3.10 or higher** (tested on Python 3.11 and 3.12)
- **Groq Cloud API Key** — free tier available at [console.groq.com](https://console.groq.com)
- The following Python packages (all specified in `requirements.txt`):
    - `streamlit >= 1.30.0`
    - `fastapi >= 0.104.0`
    - `uvicorn >= 0.24.0`
    - `httpx >= 0.25.0`
    - `pandas >= 2.0.0`
    - `plotly >= 5.18.0`
    - `scikit-learn >= 1.3.0`
    - `pydantic >= 2.0.0`

All governance modules use **zero external ML model dependencies** — pure Python stdlib
deterministic DFA pattern matching for sub-millisecond latency.


## Recommended integrations

- **Groq API Cloud**: Provides ultra-fast LLM inference (<200ms TTFT) for live streaming
  interception demonstrations. Recommended models: `qwen/qwen3.8-27b`,
  `meta-llama/llama-3.3-70b-versatile`.
- **Redis Cluster**: Recommended for production multi-node deployments of the L1/L2
  Hierarchical Semantic Cache to persist cache state across restarts.
- **Enterprise SIEM (Splunk, Datadog, Elastic)**: The immutable audit telemetry log in
  `controlplane/telemetry.py` is formatted for automated ingestion into enterprise SIEM
  platforms for SOC 2 and HIPAA compliance dashboards.


## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/niranjana-105/ControlPlane_ai.git
    cd ControlPlane_ai
    ```

1. Create and activate a virtual environment:

    ```bash
    python -m venv env

    # On Windows:
    .\env\Scripts\activate

    # On Linux/macOS:
    source env/bin/activate
    ```

1. Install all required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

1. Set up your environment variables by copying the provided template:

    ```bash
    cp .env.example .env
    ```

1. Open `.env` and insert your Groq API key:

    ```ini
    GROQ_API_KEY=gsk_your_actual_groq_api_key_here
    ```


## Configuration

### Starting the application

Launch the unified governance dashboard and FastAPI reverse proxy gateway with a single
command:

```bash
python run.py
```

The following endpoints become available immediately:

- **Interactive Governance Console**: http://localhost:8501
- **FastAPI Reverse Proxy Gateway**: http://localhost:8000
- **OpenAPI Documentation**: http://localhost:8000/docs
- **Gateway Health Check**: http://localhost:8000/health

### Governance policy profile selection

In the left sidebar of the Streamlit console, select a policy profile from the
**Governance Policy** dropdown to apply the appropriate regulatory rules for your
deployment context:

- **Customer Support Bot** — enforces GDPR privacy rules and anti-toxicity filters.
- **Internal Developer Copilot** — protects API keys, database passwords, and cloud
  secrets under SOC 2.
- **Clinical & Financial Support** — enforces HIPAA MRN/ICD-10 protection and
  zero-contradiction medical safety.

Policy profiles are declared in `controlplane/config.py` and require no code changes
to update thresholds or protected entity types.


## Architecture & governance pipeline

ControlPlane.ai operates as an inline reverse proxy and stream interceptor across three
non-blocking tiers:

```
User Prompt
    |
    v
[Tier 0: DFA Ingress Gate (<0.4ms)]
  - 8 compiled DFA regex jailbreak matchers (DAN, sudo, prompt injection)
  - Complexity router & heuristic token scorer
  - L1 SHA-256 exact cache check + L2 TF-IDF semantic cache check
    |
    | (clean prompts only — injections hard-blocked here)
    v
[Live Upstream LLM: Groq API]
    |
    | (streaming SSE token window)
    v
[Tier 1: Concurrent Parallel Egress Interceptor (<5ms)]
  |- DFA PII/PHI Redactor (15 compiled regex patterns)
  |- Sliding-Window NLI Contradiction Engine (15-token window + Shannon entropy)
  |- Multi-Category Bias Detector (gender, age, race, disability, toxicity)
  \- Async AI-as-Judge dispatcher (non-blocking background thread, 0ms overhead)
    |
    | (collects all risk categories — no short-circuit)
    v
[Tier 2: Composable Action Engine]
  |- PII masking     -> [REDACTED_SSN], [REDACTED_EMAIL], [REDACTED_CREDIT_CARD]
  |- Bias transform  -> in-flight inclusive language substitution
  |- Contradiction   -> cascade fallback replacement
  \- Epistemic hedge -> [Caution: Unverified assertion...]
    |
    v
Governed clean streaming output delivered to end user
+ Immutable audit record written to telemetry ring buffer
```

### Key technical modules

- `controlplane/ingress.py`: Tier 0 DFA sanitizer evaluating prompt complexity and 8
  compiled jailbreak patterns.
- `controlplane/pii_redactor.py`: 15 compiled DFA patterns for SSN, credit cards,
  emails, phone numbers, API keys, passwords, HIPAA MRNs, and ICD-10 diagnosis codes.
- `controlplane/nli_engine.py`: 15-token sliding-window lexical contradiction detector,
  negative assertion booster, and Shannon entropy uncertainty calculator.
- `controlplane/bias_detector.py`: Multi-category ontology detector with in-flight
  string neutralizers for ageist, sexist, and toxic generalizations.
- `controlplane/action_engine.py`: Composable non-short-circuiting transform engine
  resolving category-accurate actions: `REDACT_PII`, `BIAS_NEUTRALIZE`,
  `CASCADE_FALLBACK`, `HEDGE_UNVERIFIED`, `HARD_BLOCK`, `COMPOSITE_GOVERNED`.
- `controlplane/cache.py`: L1 SHA-256 exact match + L2 TF-IDF cosine-similarity
  semantic vector cache delivering a 35% average hit rate on enterprise traffic.
- `controlplane/session_state.py`: Multi-turn risk aggregator using weighted decay:
  R(t) = 0.6 * r(t) + 0.3 * r(t-1) + 0.1 * r(t-2).
- `controlplane/telemetry.py`: In-memory immutable audit ring buffer computing the
  Trustworthiness Index (T_score) for every processed request.
- `controlplane/benchmark_scenarios.py`: 44 automated test scenarios with precision,
  recall, and F1 evaluation across all risk categories.
- `app.py`: Streamlit 4-tab governance console (Live Stream Inspector, Audit &
  Observability, Benchmark Suite, Human Oversight Hub).
- `run.py`: Single-command startup launcher.


## Policy profiles & jurisdictions

ControlPlane.ai provides three declarative, code-free policy profiles in
`controlplane/config.py`. Switching profiles requires no code changes.

| Profile | Jurisdiction | Latency SLA | Protected Entities | Primary Risk Mitigated |
| :--- | :--- | :--- | :--- | :--- |
| Customer Support Bot | GDPR (EU) | <=18ms | SSN, Credit Cards, Emails, Phone, ZIP, Toxicity | Customer PII leaks and brand-damaging outputs |
| Internal Dev Copilot | SOC 2 | <=12ms | API Keys, DB Passwords, Cloud Credentials | Hardcoded secrets in generated code |
| Clinical & Financial Support | HIPAA (US) | <=22ms | Patient Names, MRN, ICD-10 Codes, Diagnosis | Medical hallucinations and PHI exposure |


## Governance action types

The Tier 2 Composable Action Engine resolves one of the following actions per request:

| Action Type | Trigger Condition |
| :--- | :--- |
| `PASSTHROUGH` | No risk categories detected — clean response delivered as-is |
| `REDACT_PII` | Only PII/PHI entities detected — sensitive fields masked in-place |
| `BIAS_NEUTRALIZE` | Only bias/stereotype detected — ageist/sexist phrases replaced |
| `CASCADE_FALLBACK` | Only factual contradiction detected — conflicting claim replaced |
| `HEDGE_UNVERIFIED` | Only unverified quantitative assertion detected — advisory appended |
| `COMPOSITE_GOVERNED` | Two or more distinct risk categories detected simultaneously |
| `HARD_BLOCK` | Jailbreak, sudo injection, or critical session escalation detected |


## Benchmark suite (44 scenarios)

An automated 44-scenario test harness in `controlplane/benchmark_scenarios.py` validates
precision, recall, and F1 across five failure categories. Run it from Tab 3 of the
Streamlit console.

- **PII & secrets leaks (10 scenarios)**: SSNs, email addresses, credit cards,
  passwords, API keys, HIPAA MRNs, and ICD-10 diagnosis codes.
- **Hallucinations & contradictions (10 scenarios)**: Numerical contradictions,
  conflicting financial metrics, policy violations, and high-entropy ungrounded claims.
- **Bias & stereotypes (10 scenarios)**: Gender leadership stereotypes, racial
  generalizations, age discrimination, ableist slurs, and workplace toxicity.
- **Composite violations (8 scenarios)**: Simultaneous multi-risk occurrences such as
  PII plus contradiction plus bias in a single response window.
- **Edge cases (6 scenarios)**: Legal section numbers, RSA cryptographic constants,
  zero-width unicode injection, and long clean business communications.


## Troubleshooting & FAQ

**Q: The dashboard shows a connection error or API failure.**

A: Verify that `GROQ_API_KEY` is correctly set in your `.env` file. Test your key with:

```bash
python -c "from controlplane.live_upstream import LiveLLMClient; print('Key OK')"
```

**Q: Interception latency is higher than the expected 5ms budget.**

A: Ensure you are routing through a high-throughput Groq model such as
`qwen/qwen3.8-27b` or `meta-llama/llama-3.3-70b-versatile`. Verify that the L1/L2
cache is active by checking **Semantic Cache Performance** in Tab 2 of the console.

**Q: Does the in-flight inspection add perceptible delay to streaming responses?**

A: No. ControlPlane.ai's parallel heuristics execute in 1.1ms to 4.5ms per token
window, which is over 4x faster than average human visual reading speed of approximately
15ms per token.

**Q: Does the Human Oversight Hub (Tab 4) block the user while waiting for review?**

A: No. The end-user receives their safe, sanitized response immediately with zero delay.
Tab 4 operates as a fully asynchronous background compliance audit queue. Compliance
officers review flagged events after the fact, and their decisions are recorded into the
immutable audit trail for regulatory sign-off under EU AI Act Article 14.

**Q: How do I add a new PII entity type?**

A: Add a new compiled regex pattern to the `PII_PATTERNS` dictionary in
`controlplane/pii_redactor.py` and assign a replacement label. No other files need
to be modified.


## Maintainers

- Niranjana Nitin - [niranjana-105](https://github.com/niranjana-105)
- Manan Sadana - [manansadana](https://github.com/manansadana)

---

**Accenture Innovation Challenge 2026 | Problem Track 1: ControlPlane.ai**  
**Team BAZOOKA | IIT Guwahati | Mechanical Engineering | Class of 2027**  
*Copyright © 2026 Team BAZOOKA. All rights reserved.*
