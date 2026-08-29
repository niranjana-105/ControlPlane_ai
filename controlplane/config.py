"""
ControlPlane.ai Policy & Configuration Engine
Defines enterprise profiles, latency budgets, jurisdiction-specific compliance rules, and threshold configurations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set


class Jurisdiction(str, Enum):
    BASE_SOC2 = "BASE_SOC2"
    EU_AI_ACT = "EU_AI_ACT"
    HIPAA_US = "HIPAA_US"
    GDPR_EU = "GDPR_EU"


class PolicyProfileType(str, Enum):
    CUSTOMER_SUPPORT = "Customer Support Bot"
    INTERNAL_COPILOT = "Internal Knowledge & Code Copilot"
    CLINICAL_FINANCIAL = "Clinical & Financial Decision Support"


@dataclass
class PolicyProfile:
    name: PolicyProfileType
    description: str
    jurisdiction: Jurisdiction = Jurisdiction.BASE_SOC2
    
    # Latency Budget SLA (ms)
    max_ingress_latency_ms: float = 5.0
    max_egress_window_latency_ms: float = 20.0
    
    # Thresholds (0.0 to 1.0)
    nli_contradiction_threshold: float = 0.65  # > threshold => contradiction
    bias_threshold: float = 0.35              # > threshold => bias detected
    entropy_uncertainty_threshold: float = 1.8 # > threshold => ungrounded claim hedging
    session_escalation_threshold: float = 0.70 # > cumulative risk => tighten policy
    
    # Caching parameters
    cache_similarity_threshold: float = 0.94
    cache_ttl_seconds: int = 3600
    
    # Enabled Guards
    enable_semantic_cache: bool = True
    enable_complexity_routing: bool = True
    enable_pii_redaction: bool = True
    enable_nli_grounding: bool = True
    enable_bias_detection: bool = True
    enable_session_risk_tracking: bool = True
    enable_epistemic_hedging: bool = True
    
    # Jurisdiction-Specific Sensitive Entity Categories
    sensitive_entities: Set[str] = field(default_factory=lambda: {
        "SSN", "CREDIT_CARD", "EMAIL", "PHONE", "API_KEY"
    })
    
    # Regulatory Metadata
    audit_retention_days: int = 365
    requires_human_oversight_log: bool = False
    
    def apply_jurisdiction(self, jurisdiction: Jurisdiction):
        self.jurisdiction = jurisdiction
        if jurisdiction == Jurisdiction.HIPAA_US:
            self.sensitive_entities.update({"MRN", "HEALTH_PLAN_ID", "DIAGNOSIS_CODE", "PATIENT_NAME"})
            self.audit_retention_days = 2190  # 6 years for HIPAA
            self.requires_human_oversight_log = True
        elif jurisdiction == Jurisdiction.EU_AI_ACT:
            self.sensitive_entities.update({"BIOMETRIC_DATA", "NATIONAL_ID", "POLITICAL_OPINION"})
            self.audit_retention_days = 3650  # 10 years for High-Risk AI
            self.requires_human_oversight_log = True
            self.bias_threshold = min(self.bias_threshold, 0.25)
        elif jurisdiction == Jurisdiction.GDPR_EU:
            self.sensitive_entities.update({"IP_ADDRESS", "PASSPORT", "LOCATION_DATA"})
            self.audit_retention_days = 730
            self.requires_human_oversight_log = True


# Pre-configured Enterprise Use-Case Profiles
DEFAULT_PROFILES: Dict[PolicyProfileType, PolicyProfile] = {
    PolicyProfileType.CUSTOMER_SUPPORT: PolicyProfile(
        name=PolicyProfileType.CUSTOMER_SUPPORT,
        description="Public-facing support agent. Zero tolerance for PII leaks, low bias tolerance, balanced grounding.",
        jurisdiction=Jurisdiction.GDPR_EU,
        max_ingress_latency_ms=5.0,
        max_egress_window_latency_ms=18.0,
        nli_contradiction_threshold=0.60,
        bias_threshold=0.25,
        entropy_uncertainty_threshold=1.75,
        session_escalation_threshold=0.65,
        enable_semantic_cache=True,
        enable_complexity_routing=True,
        enable_pii_redaction=True,
        enable_nli_grounding=True,
        enable_bias_detection=True,
        enable_session_risk_tracking=True,
        sensitive_entities={"SSN", "CREDIT_CARD", "EMAIL", "PHONE", "PASSPORT", "API_KEY", "LOCATION_DATA"},
        audit_retention_days=730,
        requires_human_oversight_log=False
    ),
    
    PolicyProfileType.INTERNAL_COPILOT: PolicyProfile(
        name=PolicyProfileType.INTERNAL_COPILOT,
        description="Internal engineering & documentation assistant. Optimized for sub-10ms speed, code extraction, and developer ergonomics.",
        jurisdiction=Jurisdiction.BASE_SOC2,
        max_ingress_latency_ms=3.0,
        max_egress_window_latency_ms=12.0,
        nli_contradiction_threshold=0.75,
        bias_threshold=0.55,
        entropy_uncertainty_threshold=2.2,
        session_escalation_threshold=0.85,
        enable_semantic_cache=True,
        enable_complexity_routing=True,
        enable_pii_redaction=True,
        enable_nli_grounding=True,
        enable_bias_detection=True,
        enable_session_risk_tracking=True,
        sensitive_entities={"API_KEY", "PASSWORD", "SSN", "CREDIT_CARD"},
        audit_retention_days=365,
        requires_human_oversight_log=False
    ),
    
    PolicyProfileType.CLINICAL_FINANCIAL: PolicyProfile(
        name=PolicyProfileType.CLINICAL_FINANCIAL,
        description="Regulated decision-support tool. Strict factual grounding, zero-hallucination tolerance, dynamic cascade fallback, and full audit trail.",
        jurisdiction=Jurisdiction.HIPAA_US,
        max_ingress_latency_ms=5.0,
        max_egress_window_latency_ms=22.0,
        nli_contradiction_threshold=0.45,
        bias_threshold=0.20,
        entropy_uncertainty_threshold=1.40,
        session_escalation_threshold=0.50,
        enable_semantic_cache=False,  # Bypasses cache to enforce deterministic real-time premise validation
        enable_complexity_routing=True,
        enable_pii_redaction=True,
        enable_nli_grounding=True,
        enable_bias_detection=True,
        enable_session_risk_tracking=True,
        sensitive_entities={"SSN", "MRN", "HEALTH_PLAN_ID", "DIAGNOSIS_CODE", "PATIENT_NAME", "CREDIT_CARD", "PHONE", "EMAIL"},
        audit_retention_days=2190,
        requires_human_oversight_log=True
    )
}
