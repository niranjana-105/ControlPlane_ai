# ControlPlane.ai — Enterprise AI Governance Layer
## Detailed Business Proposal & Implementation Strategy
**Accenture Innovation Challenge 2026 | Round 2 Submission**  
**Team Name:** BAZOOKA  
**Team Members:** Niranjana Nitin, Manan Sadana  
**Institution:** Indian Institute of Technology (IIT) Guwahati (Class of 2027)  
**Track:** Problem Track 1 — AI Governance & Responsible AI  
**GitHub Repository:** [https://github.com/niranjana-105/ControlPlane_ai](https://github.com/niranjana-105/ControlPlane_ai)

---

## Executive Summary

Enterprise adoption of Generative AI is accelerating across customer-facing assistants, employee copilots, and regulated decision-support tools. However, enterprise deployment has hit an **"Implementation & Compliance Barrier"**: foundation models consume untrusted prompts, output non-deterministic responses, leak sensitive credentials, hallucinate numerical facts, and exhibit demographic biases.

**ControlPlane.ai** is a real-time, in-flight AI governance middleware that intercepts streaming LLM tokens in memory before they reach end users. By combining DFA lexical scanners, 15-token sliding-window Natural Language Inference (NLI), Shannon-entropy uncertainty estimation, and composable action resolution, ControlPlane.ai delivers **end-to-end responsible AI enforcement in under 5 milliseconds**—well below the strict 20ms enterprise SLA threshold.

---

## 1. Problem Framing & Real-World Complexity

### 1.1 The Enterprise Trilemma: Speed vs. Safety vs. Compliance
Modern enterprises consume third-party foundation models (OpenAI, Groq, Anthropic, Meta Llama) via API endpoints. Because organizations do not own the internal weights of these models, traditional model-retraining and fine-tuning cannot solve real-time governance:

1. **Catastrophic Privacy & Credential Leaks:** Models inadvertently reproduce Personally Identifiable Information (PII), Protected Health Information (PHI), database connection strings, and production API keys (`sk-prod-...`).
2. **Hallucination & Numerical Contradiction:** Generative models make factual self-contradictions (e.g., reporting 847% profit growth alongside a 14% revenue loss in the same financial memo).
3. **Workplace Bias & Legal Liability:** Subconscious training biases introduce gender, age, racial, or socioeconomic stereotypes into hiring, performance reviews, and executive decision-making.
4. **Latency Constraints:** Post-generation batch safety filters add 500ms–2000ms delays, breaking streaming user experiences and violating real-time application SLAs.
5. **Regulatory Exposure:** Non-compliance with emerging frameworks (EU AI Act, GDPR, HIPAA, US Executive Order 14110) introduces multi-million dollar fines (up to 4% of global turnover under GDPR / €35M under EU AI Act).

```
Traditional Approach (Post-Hoc Batch Inspection)
User ──▶ LLM (Generates 500 tokens) ──▶ [Slow Safety LLM (1200ms)] ──▶ Output (Delayed)

ControlPlane.ai Approach (In-Flight Stream Interception)
User ──▶ [Tier 0: DFA Ingress] ──▶ LLM (Streaming) ──▶ [Tier 1: Parallel Interceptor (<5ms)] ──▶ Governed Stream
```

---

## 2. Solution Design & Technical Architecture

ControlPlane.ai is engineered as an inline reverse proxy and stream interceptor operating across three non-blocking tiers:

```mermaid
graph TD
    A[User Input / Prompt] --> B[Tier 0: DFA Ingress Gate]
    B -->|Jailbreak / Injection Detected| C[Instant Rejection <0.1ms]
    B -->|Clean / Approved| D[Live Upstream LLM: Groq API]
    
    D -->|Streaming SSE Tokens| E[Tier 1: Parallel Egress Interceptor]
    
    subgraph "Tier 1: Parallel Interceptor (<5ms)"
        E1[DFA PII & PHI Redactor]
        E2[Sliding Window NLI Engine]
        E3[Multi-Category Bias Detector]
        E4[Shannon Entropy Uncertainty Estimator]
    end
    
    E --> E1
    E --> E2
    E --> E3
    E --> E4
    
    E1 & E2 & E3 & E4 --> F[Tier 2: Composable Action Engine]
    
    subgraph "Tier 2: Action Resolution"
        F1[PII Masking]
        F2[Contradiction Replacement]
        F3[Bias Neutralization]
        F4[Epistemic Hedging]
    end
    
    F --> F1 & F2 & F3 & F4
    F --> G[Governed Clean Output to User]
    F --> H[(Immutable Telemetry & Audit Trail)]
    F --> I[Async AI-as-Judge Dispatcher]
```

### 2.1 Technical Core Modules
1. **Tier 0 Ingress Gate (`controlplane/ingress.py`):**
   * Pre-execution DFA sanitizer analyzing prompt complexity, token estimates, and 8 compiled jailbreak patterns (`DAN_JAILBREAK`, `SUDO_ESCALATION`, `SYSTEM_OVERRIDE`).
   * Execution latency: **`0.08ms – 0.40ms`**.
2. **Tier 1 Parallel Egress Interceptors:**
   * **DFA PII / PHI Redactor (`controlplane/pii_redactor.py`):** 15 regex DFAs scanning for SSN, credit cards, emails, phone numbers, API keys, passwords, HIPAA MRNs, and ICD-10 diagnosis codes.
   * **Sliding-Window NLI Engine (`controlplane/nli_engine.py`):** Evaluates 15-token lexical contradiction density, negative assertion boosts, and Shannon entropy uncertainty ($H$).
   * **Real-Time Bias Detector (`controlplane/bias_detector.py`):** Identifies demographic stereotypes across gender, race, age, and toxicity with in-flight string neutralizers.
3. **Tier 2 Composable Action Engine (`controlplane/action_engine.py`):**
   * Non-short-circuiting transform engine that applies multiple concurrent actions (`REDACT_PII`, `BIAS_NEUTRALIZE`, `CASCADE_FALLBACK`, `HEDGE_UNVERIFIED`, `HARD_BLOCK`) and categorizes multi-risk responses into `COMPOSITE_GOVERNED`.
4. **Hierarchical Semantic Cache (`controlplane/cache.py`):**
   * **L1 Cache:** SHA-256 exact-match cache (<0.05ms).
   * **L2 Cache:** TF-IDF cosine-similarity semantic vector cache (<1.0ms) saving up to 40% upstream API token costs.
5. **Multi-Turn Session Escalation Aggregator (`controlplane/session_state.py`):**
   * Cumulative multi-turn risk equation:
     $$\text{Risk}(t) = 0.6 \cdot r(t) + 0.3 \cdot r(t-1) + 0.1 \cdot r(t-2)$$
   * Automatically escalates risk level and restricts capabilities when an adversary tests repeated prompt variations.
6. **Regulatory Audit & Trustworthiness Index ($T_{\text{score}}$) (`controlplane/telemetry.py`):**
   * Computes mathematical trustworthiness index:
     $$T_{\text{score}} = \max\left(0, 1 - \left[ 0.30 S_{\text{NLI}} + 0.25 S_{\text{Bias}} + 0.20 S_{\text{Entropy}} + 0.15 R_{\text{Session}} + 0.10 P_{\text{PII}} \right]\right)$$

---

## 3. Target Users & Enterprise Personas

ControlPlane.ai delivers tailored policy profiles for distinct enterprise deployment environments:

| Enterprise Profile | Primary Persona | Latency Budget | Active Protections & Jurisdiction Rules | Primary Failure Mitigated |
| :--- | :--- | :--- | :--- | :--- |
| **Customer Support Bot** | Customer Experience & Operations | $\le 18\text{ms}$ | GDPR_EU Profile: SSN, Credit Cards, Emails, Phone, Contextual ZIP, Anti-Toxicity | Leaking customer PII or brand-damaging toxic responses |
| **Internal Dev Copilot** | Software Engineering & IT Security | $\le 12\text{ms}$ | Base SOC2 Profile: API Keys, Admin Passwords, Database Credentials, Sub-5ms Ingress | Committing hardcoded credentials or cloud secrets |
| **Clinical & Financial Support** | Healthcare Clinicians & Risk Officers | $\le 22\text{ms}$ | HIPAA_US Profile: Patient Names, Medical Record Numbers (MRN), ICD-10 Codes, Zero Contradiction Tolerance | Medical malpractice liability and hallucinated clinical guidance |

---

## 4. Business Case, Financial Impact & ROI

### 4.1 Cost Savings & Risk Reduction Model
For a mid-to-large enterprise processing **10,000,000 GenAI requests per month**:

| Metric | Traditional Unmanaged Deployment | With ControlPlane.ai Layer | Quantified Business Value |
| :--- | :--- | :--- | :--- |
| **Regulatory Risk (GDPR/EU AI Act)** | High liability of unmasked PII leaks | 100% Deterministic Regex Masking | **Avoidance of fines up to €20M or 4% global turnover** |
| **Upstream API Token Spend** | \$30,000 / month (\$0.003 / req) | \$19,500 / month (35% Cache Hit Rate) | **\$126,000 Annual Direct Token Savings** |
| **Governance Latency Overhead** | +1,200ms (Post-call safety model) | **1.1ms – 4.5ms (In-flight Interceptor)** | **Preserves real-time streaming UX (<20ms SLA)** |
| **Security Audit Compliance Cost** | \$250,000 / year in manual audits | Automated Immutable Telemetry Logs | **70% reduction in audit prep & SOC2 compliance hours** |
| **Brand Reputation & Safety** | Vulnerable to jailbreaks & biased outputs | Tier 0 Ingress Block + Real-Time Neutralizer | **Eliminates public PR risk and customer attrition** |

### 4.2 Return on Investment (ROI)
* **Estimated Implementation & Cloud Hosting Cost:** \$75,000 (Year 1)
* **Direct Token & Productivity Savings:** \$210,000 (Year 1)
* **Net Projected ROI:** **`280% in Year 1`** (excluding catastrophic legal fine avoidance).

---

## 5. Phased Implementation Roadmap

```mermaid
gantt
    title ControlPlane.ai 9-Month Enterprise Rollout Roadmap
    dateFormat  YYYY-MM-DD
    section Phase 1: Foundation (M1-M3)
    Ingress DFA Gate & In-Flight PII Redaction        :active, p1_1, 2026-01-01, 2026-02-15
    L1/L2 Semantic Cache & Background Telemetry      :active, p1_2, 2026-02-15, 2026-03-31
    section Phase 2: Core Intelligence (M4-M6)
    Sliding-Window NLI Contradiction Engine          :p2_1, 2026-04-01, 2026-05-15
    Multi-Category Bias Detection & Neutralization   :p2_2, 2026-05-01, 2026-06-30
    section Phase 3: Enterprise Scale (M7-M9)
    Human Oversight Hub & Compliance Review Queue   :p3_1, 2026-07-01, 2026-08-15
    SOC2 / HIPAA / GDPR Compliance Sign-off          :p3_2, 2026-08-15, 2026-09-30
```

* **Phase 1 (Months 1–3) — Perimeter Ingress & Privacy Shield:**
  * Deploy lightweight FastAPI proxy gateway in shadow mode.
  * Enforce Tier 0 Ingress DFA gate for prompt injections and Tier 1 PII masking.
* **Phase 2 (Months 4–6) — In-Flight Semantic Interception:**
  * Activate sliding-window NLI contradiction detection and multi-category bias neutralizers.
  * Integrate multi-turn session escalation tracking and hierarchical semantic caching.
* **Phase 3 (Months 7–9) — Enterprise Scale & Governance Workflows:**
  * Roll out Tab 4 Human Oversight review queue for compliance officers.
  * Achieve SOC2 Type II and HIPAA audit certification across all corporate AI pipelines.

---

## 6. Key Enterprise Risks & Mitigation Strategies

| Risk Factor | Probability / Impact | Real-World Failure Scenario | ControlPlane.ai Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **False Positives & Alert Fatigue** | Moderate / High | Overly sensitive filters block legitimate engineering or mathematical queries (e.g., Fermat prime `65537` or technical terms). | **Contextual Pattern Scanners:** Requires address/postal prefix for ZIP codes; calibrated Shannon entropy thresholds ($H \ge 7.5$) preventing false flags on educational technical text. |
| **Latency Budget Degradation** | Low / Critical | Complex deep evaluation slows streaming tokens to users. | **Decoupled Architecture:** Sub-5ms lightweight heuristic interceptor runs in-flight; deep AI-as-Judge runs **asynchronously in the background**. |
| **Adversarial Multi-Turn Jailbreaks** | Moderate / High | Attackers use multi-step conversational probing to bypass single-turn checks. | **Session State Risk Aggregator:** Computes rolling historical risk ($0.6 r_t + 0.3 r_{t-1} + 0.1 r_{t-2}$), triggering automated session restrictions. |
| **Regulatory Drift across Jurisdictions** | High / Moderate | Changing EU AI Act or local privacy rules make static code obsolete. | **Modular Policy Profiles (`config.py`):** Decoupled declarative JSON/dataclass policies allow compliance teams to update retention rules and sensitive entity sets without code deployments. |

---

## 7. Conclusion & Competitive Advantage

ControlPlane.ai delivers the industry's first **true in-flight, multi-jurisdiction AI governance gateway**. By intercepting tokens before client rendering with **$<5\text{ms}$ latency overhead**, it transforms Generative AI from an unpredictable enterprise liability into a secure, compliant, and auditable corporate capability.

**Accenture Innovation Challenge 2026 | Team BAZOOKA (IIT Guwahati)**
