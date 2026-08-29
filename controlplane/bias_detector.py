"""
ControlPlane.ai - Real-Time Bias & Fairness Detector
Concurrent stereotype, demographic bias, and toxicity detection.
Runs in the Tier 1 egress window alongside NLI and PII scanners.
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple


@dataclass
class BiasResult:
    is_biased: bool
    severe_breach: bool          # Hard-block trigger (extreme toxicity)
    categories: List[str]        # e.g. ["GENDER_STEREOTYPE", "RACIAL_BIAS"]
    score: float                 # Aggregate bias score [0.0-1.0]
    latency_ms: float
    violations: List[str]        # Human-readable violation descriptions
    neutralized_text: Optional[str] = None

    def apply_neutralizer(self, text: str) -> str:
        """Apply the neutralizer to a given text (used in action engine composition)."""
        if self.neutralized_text is not None:
            return self.neutralized_text
        return _apply_neutralization_rules(text)


# ---------------------------------------------------------------------------
# Bias Ontology — Pattern Registry (Robust & Comprehensive)
# ---------------------------------------------------------------------------

_BIAS_PATTERNS: Dict[str, List[Tuple[re.Pattern, str]]] = {
    "GENDER_STEREOTYPE": [
        (re.compile(r"\b(women|girls?|females?)\s+(are|tend\s+to\s+be|should\s+be|become|seem)\s+(too\s+)?(emotional|weak|irrational|hysterical|better\s+at\s+cleaning|less\s+(logical|rational|technical))", re.I),
         "Gender-stereotyping language detected"),
        (re.compile(r"\b(men|boys?|males?)\s+(are|tend\s+to\s+be|should\s+be)\s+(naturally\s+)?(better\s+leaders?|more\s+(logical|rational|dominant|aggressive|suited))", re.I),
         "Masculinity-superiority stereotype detected"),
        (re.compile(r"\b(women|females?)\s+(cannot|can\'t|shouldn\'t|aren\'t\s+suited\s+to)\s+(lead|code|program|engineer|manage)", re.I),
         "Female workplace capability exclusion pattern detected"),
        (re.compile(r"\b(manly|girly|like\s+a\s+girl|man\s+up)\b", re.I),
         "Gender-diminutive phrasing detected"),
    ],
    "RACIAL_BIAS": [
        (re.compile(r"\b(all|most|many)\s+(black|white|asian|hispanic|latino|jewish|arab|muslim)\s+(people|men|women|individuals|engineers|coders|workers)\s+(are|tend|like|prefer|naturally\s+better)\b", re.I),
         "Racial generalization pattern detected"),
        (re.compile(r"\b(asian|black|white|hispanic)\s+(engineers?|people|workers?)\s+(are\s+naturally\s+(better|worse|superior|inferior))\b", re.I),
         "Racial innate ability stereotype detected"),
        (re.compile(r"\bthug|ghetto|exotic\s+look|articulate\s+for\s+a\b", re.I),
         "Racially coded language detected"),
    ],
    "AGE_DISCRIMINATION": [
        (re.compile(r"\b(old|elderly|senior)\s+(people|workers|employees)\s+(are|tend|can\'t|cannot|won\'t|too\s+old|out\s+of\s+touch)\b", re.I),
         "Age-based generalization detected"),
        (re.compile(r"\b(too\s+old\s+to\s+learn|past\s+their\s+prime|out\s+of\s+touch\s+with\s+technology)\b", re.I),
         "Ageist language detected"),
    ],
    "DISABILITY_BIAS": [
        (re.compile(r"\b(crazy|insane|retarded|handicapped|suffers?\s+from)\b", re.I),
         "Ableist or stigmatizing disability language detected"),
        (re.compile(r"\b(wheelchair-bound|mentally\s+ill\s+people\s+(are|can\'t|always))\b", re.I),
         "Disability stereotyping detected"),
    ],
    "SOCIOECONOMIC_BIAS": [
        (re.compile(r"\b(poor\s+people|low[-\s]income\s+(individuals|workers))\s+(are|tend|always|never|should)\b", re.I),
         "Socioeconomic stereotyping detected"),
        (re.compile(r"\b(welfare\s+queen|lazy\s+poor|underserving\s+poor)\b", re.I),
         "Harmful socioeconomic generalization detected"),
    ],
    "TOXICITY": [
        (re.compile(r"\b(kill|destroy|eliminate|exterminate)\s+(all|those|the)\s+\w+\b", re.I),
         "Extreme violent language detected"),
        (re.compile(r"\b(hate|despise|loathe)\s+(all|those|the)\s+(people|humans|group)\b", re.I),
         "Hate speech pattern detected"),
        (re.compile(r"(?:f+u+c+k+|s+h+i+t+|a+s+s+h+o+l+e+|b+i+t+c+h+)\s+(?:you|them|those|all)", re.I),
         "Severe profanity targeting individuals/groups"),
    ],
}

_SEVERE_CATEGORIES = {"TOXICITY"}

_NEUTRALIZATION_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(women|girls?|females?)\s+(are|tend\s+to\s+be)\s+(too\s+)?(emotional|weak)\b", re.I),
        "individuals may have varying emotional responses"),
    (re.compile(r"\b(men|boys?|males?)\s+(are|tend\s+to\s+be)\s+(naturally\s+)?(more\s+)?(logical|dominant|aggressive|suited)\b", re.I),
        "individuals may display varying technical and behavioral traits"),
    (re.compile(r"\b(all|most)\s+asian\s+engineers\s+are\s+(naturally\s+)?better\b", re.I),
        "engineers from all backgrounds demonstrate diverse proficiencies"),
    (re.compile(r"\bmanly\b", re.I), "capable"),
    (re.compile(r"\bgirly\b", re.I), "thoughtful"),
    (re.compile(r"\bcrazy\b|\binsane\b", re.I), "unconventional"),
    (re.compile(r"\bretarded\b", re.I), "challenging"),
    (re.compile(r"\bthug\b", re.I), "individual"),
    (re.compile(r"\bghetto\b", re.I), "underserved community"),
]


def _apply_neutralization_rules(text: str) -> str:
    """Apply all neutralization substitutions to text."""
    result = text
    for pattern, replacement in _NEUTRALIZATION_RULES:
        result = pattern.sub(replacement, result)
    return result


# ---------------------------------------------------------------------------
# Bias Detector
# ---------------------------------------------------------------------------

class BiasDetector:
    """
    Real-time multi-category bias detector.
    Runs concurrently with NLI and PII in the Tier 1 egress interceptor.
    """

    def __init__(self, bias_threshold: float = 0.25, enabled_categories: Optional[Set[str]] = None):
        self._threshold = bias_threshold
        self._enabled = enabled_categories or set(_BIAS_PATTERNS.keys())

    def evaluate(self, text: str) -> BiasResult:
        t0 = time.perf_counter()

        hit_categories: List[str] = []
        violations: List[str] = []
        severe = False
        total_hits = 0

        for category, patterns in _BIAS_PATTERNS.items():
            if category not in self._enabled:
                continue
            for pattern, description in patterns:
                if pattern.search(text):
                    if category not in hit_categories:
                        hit_categories.append(category)
                    violations.append(description)
                    total_hits += 1
                    if category in _SEVERE_CATEGORIES:
                        severe = True

        # Absolute score assignment when explicit ontology patterns hit
        if total_hits > 0:
            raw_score = max(0.60, min(0.60 + total_hits * 0.20, 1.0))
        else:
            raw_score = 0.0

        if severe:
            raw_score = 1.0

        is_biased = raw_score >= self._threshold
        neutralized = _apply_neutralization_rules(text) if is_biased else None

        latency_ms = (time.perf_counter() - t0) * 1000

        return BiasResult(
            is_biased=is_biased,
            severe_breach=severe,
            categories=hit_categories,
            score=round(raw_score, 4),
            latency_ms=round(latency_ms, 3),
            violations=violations,
            neutralized_text=neutralized,
        )

    def update_threshold(self, threshold: float) -> None:
        self._threshold = threshold

    def enable_categories(self, categories: Set[str]) -> None:
        self._enabled = categories
