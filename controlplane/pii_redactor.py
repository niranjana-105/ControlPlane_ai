"""
ControlPlane.ai - Jurisdiction-Aware Streaming PII/PHI Masker
Zero-copy streaming redactor using compiled DFA regex patterns.
Supports GDPR, HIPAA, SOC2, and EU AI Act sensitive entity categories.
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Pattern
from enum import Enum


# ---------------------------------------------------------------------------
# Lazy spaCy NER Model Loader
# Falls back silently if spaCy or the model is not installed.
# ---------------------------------------------------------------------------

_nlp = None          # spaCy NLP pipeline (loaded on first use)
_ner_available = None  # None = not checked yet; True/False after first load


def _get_ner_model():
    """
    Lazy-loads the spaCy en_core_web_sm model on first call.
    Returns the model if available, None if spaCy is not installed.
    """
    global _nlp, _ner_available
    if _ner_available is None:
        try:
            import spacy  # noqa: PLC0415
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Model not downloaded yet — auto-download
                import subprocess, sys
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    capture_output=True, check=False
                )
                try:
                    _nlp = spacy.load("en_core_web_sm")
                except OSError:
                    _nlp = None
            _ner_available = _nlp is not None
        except ImportError:
            _ner_available = False
            _nlp = None
    return _nlp if _ner_available else None


# NER entity labels → PII category names
_NER_ENTITY_MAP = {
    "PERSON": "PERSON_NAME",    # Full person names — critical for HIPAA patient names, GDPR
    "GPE": "LOCATION",          # Countries, cities, states
    "LOC": "LOCATION",          # Natural locations, landmarks
}

# Which sensitive_entity keys enable each NER category
_NER_TRIGGER_ENTITIES = {
    "PERSON_NAME": {"PERSON_NAME", "PATIENT_NAME"},   # Either key activates PERSON NER
    "LOCATION": {"LOCATION_DATA"},                     # GDPR location data
}


def _apply_ner_redaction(
    text: str,
    active_entities: Set[str],
) -> Tuple[str, List[str], int]:
    """
    Runs spaCy NER on text and redacts detected entities.
    Processes spans right-to-left to preserve character indices during replacement.
    Returns (redacted_text, detected_pii_types, match_count).
    """
    nlp = _get_ner_model()
    if nlp is None:
        return text, [], 0

    doc = nlp(text)
    replacements: List[Tuple[int, int, str]] = []  # (start, end, pii_type)

    for ent in doc.ents:
        pii_category = _NER_ENTITY_MAP.get(ent.label_)
        if not pii_category:
            continue
        # Check if any triggering sensitive entity is active in this policy
        trigger_set = _NER_TRIGGER_ENTITIES.get(pii_category, set())
        if not trigger_set.intersection(active_entities):
            continue
        replacements.append((ent.start_char, ent.end_char, pii_category))

    if not replacements:
        return text, [], 0

    # Sort right-to-left so index positions remain valid as we replace
    replacements.sort(key=lambda x: x[0], reverse=True)
    detected_types: List[str] = []
    match_count = 0

    for start, end, pii_type in replacements:
        replacement_label = f"[REDACTED_{pii_type}]"
        text = text[:start] + replacement_label + text[end:]
        if pii_type not in detected_types:
            detected_types.append(pii_type)
        match_count += 1

    return text, detected_types, match_count


class RedactionMode(str, Enum):
    MASK = "MASK"           # Replace with [REDACTED_<TYPE>]
    TOKENIZE = "TOKENIZE"   # Replace with reversible token (for auditing)
    HASH = "HASH"           # Replace with SHA-prefix hash


@dataclass
class PIIResult:
    detected: bool
    redacted_text: str
    pii_types: List[str]          # e.g. ["SSN", "EMAIL"]
    match_count: int
    latency_ms: float
    redaction_mode: RedactionMode = RedactionMode.MASK


# ---------------------------------------------------------------------------
# PII Pattern Registry — DFA compiled regexes
# ---------------------------------------------------------------------------

_PII_PATTERNS: Dict[str, re.Pattern] = {
    "SSN":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[\s\-]){3}\d{4}\b"),
    "EMAIL":       re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "PHONE":       re.compile(r"\b(?:\+?1[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b"),
    "IP_ADDRESS":  re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "PASSPORT":    re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
    "API_KEY":     re.compile(r"\b(?:sk-|pk-|api-key-|token-)[A-Za-z0-9_\-]{16,}\b", re.I),
    "PASSWORD":    re.compile(r"(?i)\b(password|passwd|pwd|secret)\s*[:=]\s*\S+"),
    "LOCATION_DATA": re.compile(r"(?i)\b(?:zip|postal(?:\s+code)?|address|location|city|state)\s*[:=]?\s*(\d{5}(?:-\d{4})?)\b|\b[A-Z]{2}\s+(\d{5}(?:-\d{4})?)\b"),  # Contextual ZIP codes
    # HIPAA-specific
    "MRN":              re.compile(r"\bMRN[-#:\s]?\d{6,10}\b", re.I),
    "HEALTH_PLAN_ID":   re.compile(r"\bHPID[-#:\s]?\d{8,12}\b", re.I),
    "DIAGNOSIS_CODE":   re.compile(r"\b[A-Z]\d{2}(?:\.\d{1,4})?\b"),  # ICD-10 pattern
    "PATIENT_NAME":     re.compile(r"(?i)patient\s+name\s*[:=]\s*[A-Z][a-z]+ [A-Z][a-z]+"),
    # EU AI Act
    "BIOMETRIC_DATA":   re.compile(r"(?i)(fingerprint|facial\s+recognition|retina\s+scan)\s+(?:id|hash|code)?\s*[:=]?\s*[A-Za-z0-9+/=]{16,}"),
    "NATIONAL_ID":      re.compile(r"\b[A-Z]{2}\d{8,12}[A-Z]?\b"),
    "POLITICAL_OPINION": re.compile(r"(?i)(voted|supports|endorses)\s+(party|candidate|politician)"),
}

# Default entity sets per jurisdiction (references config.py names)
_JURISDICTION_ENTITIES: Dict[str, Set[str]] = {
    "BASE_SOC2": {"SSN", "CREDIT_CARD", "EMAIL", "PHONE", "API_KEY", "PASSWORD"},
    "GDPR_EU":   {"SSN", "CREDIT_CARD", "EMAIL", "PHONE", "IP_ADDRESS", "PASSPORT", "LOCATION_DATA", "API_KEY"},
    "HIPAA_US":  {"SSN", "CREDIT_CARD", "EMAIL", "PHONE", "MRN", "HEALTH_PLAN_ID", "DIAGNOSIS_CODE", "PATIENT_NAME"},
    "EU_AI_ACT": {"SSN", "CREDIT_CARD", "EMAIL", "PHONE", "IP_ADDRESS", "BIOMETRIC_DATA", "NATIONAL_ID", "POLITICAL_OPINION"},
}


# ---------------------------------------------------------------------------
# PII Redactor
# ---------------------------------------------------------------------------

class PIIRedactor:
    """
    Streaming-compatible DFA-based PII/PHI redactor, augmented with spaCy NER.
    Stage 1: 15 compiled regex DFA patterns (structured PII — SSN, credit cards, API keys).
    Stage 2: spaCy NER (unstructured PII — person names, locations that regex cannot detect).
    NER stage only activates when enable_ner=True and the policy has relevant entities.
    """

    def __init__(
        self,
        sensitive_entities: Optional[Set[str]] = None,
        mode: RedactionMode = RedactionMode.MASK,
        enable_ner: bool = False,
    ):
        # Default to base SOC2 set
        self._entities = sensitive_entities or _JURISDICTION_ENTITIES["BASE_SOC2"]
        self._mode = mode
        self._enable_ner = enable_ner
        # Pre-select active regex patterns
        self._active_patterns: Dict[str, re.Pattern] = {
            name: pat for name, pat in _PII_PATTERNS.items() if name in self._entities
        }

    def redact(self, text: str) -> PIIResult:
        t0 = time.perf_counter()
        redacted = text
        detected_types: List[str] = []
        total_matches = 0

        # --- Stage 1: Regex DFA patterns (structured PII) ---
        for entity_type, pattern in self._active_patterns.items():
            matches = pattern.findall(redacted)
            if matches:
                detected_types.append(entity_type)
                total_matches += len(matches)
                replacement = f"[REDACTED_{entity_type}]"
                redacted = pattern.sub(replacement, redacted)

        # --- Stage 2: spaCy NER (unstructured PII — names, locations) ---
        if self._enable_ner and redacted.strip():
            ner_redacted, ner_types, ner_count = _apply_ner_redaction(redacted, self._entities)
            if ner_types:
                redacted = ner_redacted
                for t in ner_types:
                    if t not in detected_types:
                        detected_types.append(t)
                total_matches += ner_count

        latency_ms = (time.perf_counter() - t0) * 1000

        return PIIResult(
            detected=bool(detected_types),
            redacted_text=redacted,
            pii_types=detected_types,
            match_count=total_matches,
            latency_ms=round(latency_ms, 3),
            redaction_mode=self._mode,
        )

    def redact_stream_chunk(self, chunk: str) -> Tuple[str, List[str]]:
        """
        Lightweight variant for streaming token windows.
        Returns (redacted_chunk, detected_types).
        """
        result = self.redact(chunk)
        return result.redacted_text, result.pii_types

    def update_entities(self, entities: Set[str]) -> None:
        self._entities = entities
        self._active_patterns = {
            name: pat for name, pat in _PII_PATTERNS.items() if name in entities
        }

    @staticmethod
    def from_jurisdiction(jurisdiction_name: str, mode: RedactionMode = RedactionMode.MASK, enable_ner: bool = False) -> "PIIRedactor":
        entities = _JURISDICTION_ENTITIES.get(jurisdiction_name, _JURISDICTION_ENTITIES["BASE_SOC2"])
        return PIIRedactor(sensitive_entities=entities, mode=mode, enable_ner=enable_ner)
