"""
ControlPlane.ai - Tier 0 Ingress Gate
DFA-based jailbreak sanitizer, complexity router, and session risk pre-check.
Target: <5ms total ingress latency.
"""

import re
import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum


class ComplexityTier(str, Enum):
    SIMPLE = "SIMPLE"        # Cache candidate, trivial Q&A
    MODERATE = "MODERATE"    # Standard inference path
    COMPLEX = "COMPLEX"      # Full governance pipeline, may need cascade


class IngressVerdict(str, Enum):
    ALLOW = "ALLOW"
    SANITIZE = "SANITIZE"
    BLOCK = "BLOCK"


@dataclass
class IngressResult:
    verdict: IngressVerdict
    complexity: ComplexityTier
    sanitized_prompt: str
    original_prompt: str
    jailbreak_patterns_hit: List[str]
    complexity_score: float          # 0.0 - 1.0
    latency_ms: float
    blocked_reason: Optional[str] = None
    estimated_tokens: int = 0


# ---------------------------------------------------------------------------
# DFA Jailbreak Pattern Registry
# ---------------------------------------------------------------------------

# Each tuple: (pattern_name, compiled_regex)
_JAILBREAK_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("PROMPT_INJECTION_IGNORE",  re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", re.I)),
    ("ROLE_PLAY_JAILBREAK",      re.compile(r"(pretend|act|roleplay|imagine)\s+(you\s+are|as\s+if|that\s+you)", re.I)),
    ("DAN_JAILBREAK",            re.compile(r"\b(DAN|do\s+anything\s+now|jailbreak|uncensored\s+mode)\b", re.I)),
    ("SYSTEM_OVERRIDE",          re.compile(r"(system\s+prompt|override\s+(safety|filter|guardrail|restriction))", re.I)),
    ("CONFIDENTIAL_EXTRACTION",  re.compile(r"(repeat|reveal|print|show|output)\s+(your\s+)?(system\s+prompt|instructions?|training\s+data)", re.I)),
    ("SUDO_ESCALATION",          re.compile(r"\b(sudo|root\s+access|admin\s+mode|developer\s+mode)\b", re.I)),
    ("HYPOTHETICAL_BYPASS",      re.compile(r"(hypothetically|theoretically|in\s+a\s+fictional|for\s+educational\s+purposes)\s*.{0,30}(hack|exploit|bypass|attack|malware)", re.I)),
    ("TOKEN_SMUGGLING",          re.compile(r"[\u200b\u200c\u200d\ufeff]")),  # Zero-width chars
    ("EXCESSIVE_SPECIAL_CHARS",  re.compile(r"[<>{}\[\]]{5,}")),
]

# Patterns that trigger hard BLOCK (not just flag)
_HARD_BLOCK_PATTERNS = {"DAN_JAILBREAK", "SYSTEM_OVERRIDE", "CONFIDENTIAL_EXTRACTION", "SUDO_ESCALATION"}

# PII quick-scan patterns for ingress pre-check (lightweight)
_INGRESS_PII_HINTS = re.compile(
    r"\b(\d{3}-\d{2}-\d{4}|"          # SSN
    r"\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}|"  # Credit card
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}|"  # Email
    r"\b\d{3}[\s.-]\d{3}[\s.-]\d{4})\b",  # Phone
    re.I
)


# ---------------------------------------------------------------------------
# Complexity Scoring
# ---------------------------------------------------------------------------

def _estimate_complexity(prompt: str) -> Tuple[ComplexityTier, float, int]:
    """
    Fast heuristic complexity scorer. Returns (tier, score 0-1, est_tokens).
    No ML inference - pure text analysis for sub-ms performance.
    """
    # Token estimation (rough: 1 token ~ 4 chars)
    est_tokens = max(1, len(prompt) // 4)

    score = 0.0

    # Length factor
    if est_tokens > 500:   score += 0.40
    elif est_tokens > 150: score += 0.20
    else:                  score += 0.05

    # Question structure
    question_words = len(re.findall(r"\b(why|how|explain|compare|analyze|evaluate|what\s+if|pros\s+and\s+cons)\b", prompt, re.I))
    score += min(question_words * 0.10, 0.30)

    # Multi-step indicators
    steps = len(re.findall(r"\b(step\s+\d|first|then|finally|additionally|furthermore|also)\b", prompt, re.I))
    score += min(steps * 0.05, 0.20)

    # Code or technical content
    if re.search(r"(```|def |class |function |import |SELECT |FROM |WHERE )", prompt):
        score += 0.15

    score = min(score, 1.0)

    if score < 0.25:
        tier = ComplexityTier.SIMPLE
    elif score < 0.60:
        tier = ComplexityTier.MODERATE
    else:
        tier = ComplexityTier.COMPLEX

    return tier, round(score, 3), est_tokens


# ---------------------------------------------------------------------------
# Ingress Gate
# ---------------------------------------------------------------------------

class IngressGate:
    """
    Tier 0 DFA Ingress Gate.
    Runs in <5ms: pattern matching, complexity routing, pii pre-scan.
    """

    def __init__(self, enable_jailbreak_detection: bool = True, enable_complexity_routing: bool = True):
        self._jailbreak_enabled = enable_jailbreak_detection
        self._complexity_enabled = enable_complexity_routing

    def evaluate(self, prompt: str) -> IngressResult:
        t0 = time.perf_counter()

        jailbreak_hits: List[str] = []
        hard_block = False
        blocked_reason: Optional[str] = None

        # 1. DFA Jailbreak Scan
        if self._jailbreak_enabled:
            for pattern_name, pattern in _JAILBREAK_PATTERNS:
                if pattern.search(prompt):
                    jailbreak_hits.append(pattern_name)
                    if pattern_name in _HARD_BLOCK_PATTERNS:
                        hard_block = True
                        blocked_reason = f"Jailbreak pattern detected: {pattern_name}"

        verdict = IngressVerdict.BLOCK if hard_block else (
            IngressVerdict.SANITIZE if jailbreak_hits else IngressVerdict.ALLOW
        )

        # 2. Sanitize (strip zero-width chars, trim suspicious structures)
        sanitized = prompt
        if jailbreak_hits and not hard_block:
            sanitized = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", prompt)
            sanitized = re.sub(r"[<>{}\[\]]{5,}", "[SANITIZED]", sanitized)

        # 3. Complexity routing
        complexity, comp_score, est_tokens = (
            _estimate_complexity(prompt) if self._complexity_enabled
            else (ComplexityTier.MODERATE, 0.5, len(prompt) // 4)
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        return IngressResult(
            verdict=verdict,
            complexity=complexity,
            sanitized_prompt=sanitized,
            original_prompt=prompt,
            jailbreak_patterns_hit=jailbreak_hits,
            complexity_score=comp_score,
            latency_ms=round(latency_ms, 3),
            blocked_reason=blocked_reason,
            estimated_tokens=est_tokens,
        )

    def has_pii_hints(self, prompt: str) -> bool:
        """Quick PII pre-scan (does not redact, just signals for egress)."""
        return bool(_INGRESS_PII_HINTS.search(prompt))
