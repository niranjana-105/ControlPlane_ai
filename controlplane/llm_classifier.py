"""
ControlPlane.ai - Async LLM Zero-Shot Classifier
Uses Groq API for nuanced, context-aware classification of bias and content risks.
Catches implicit bias that regex patterns can never detect.
Always runs async — never blocks the user stream.
"""

import os
import json
import time
import hashlib
import threading
import httpx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable


def _load_env_file():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and not os.getenv(k):
                            os.environ[k] = v
        except Exception:
            pass


_load_env_file()


# ---------------------------------------------------------------------------
# Classification Verdict
# ---------------------------------------------------------------------------

@dataclass
class ClassificationVerdict:
    job_id: str
    task: str                        # e.g. "BIAS_DETECTION"
    score: float                     # 0.0 = clean, 1.0 = severe violation
    confidence: float                # Model's confidence in its score
    reasoning: str                   # One-sentence LLM explanation
    detected_phrases: List[str]      # Specific phrases that triggered the flag
    implicit_bias: bool              # True if bias is subtle/implicit (not caught by regex)
    model_used: str
    completed_at: float
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "task": self.task,
            "score": self.score,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "detected_phrases": self.detected_phrases,
            "implicit_bias": self.implicit_bias,
            "model_used": self.model_used,
            "completed_at": self.completed_at,
            "success": self.success,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Classifier Prompts
# ---------------------------------------------------------------------------

_BIAS_CLASSIFIER_PROMPT = """\
You are an enterprise workplace fairness auditor with expertise in detecting subtle and implicit bias.

Analyze the text for ANY form of bias including:
- Explicit stereotypes (gender, race, age, disability, religion, socioeconomic)
- Implicit bias (unconscious assumptions, e.g. "surprisingly articulate", "fits our culture")
- Microaggressions
- Coded exclusionary language
- Assumptions linking demographics to ability or role

Respond ONLY with valid JSON — no markdown, no extra text:
{
  "score": 0.0,
  "confidence": 0.0,
  "reasoning": "one concise sentence",
  "detected_phrases": ["phrase1", "phrase2"],
  "implicit_bias": false
}

score: 0.0 = completely unbiased, 1.0 = severe bias
confidence: how certain you are (0.0-1.0)
detected_phrases: exact phrases from the text that signal bias (empty list if none)
implicit_bias: true if the bias is subtle/implicit rather than explicit"""

_CLASSIFIER_MODEL = "qwen/qwen3.8-27b"


# ---------------------------------------------------------------------------
# LLM Classifier
# ---------------------------------------------------------------------------

class LLMClassifier:
    """
    Reusable async zero-shot LLM classifier using Groq.
    Dispatches in a background daemon thread — never blocks the stream.
    """

    def __init__(self):
        _load_env_file()
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.base_url = "https://api.groq.com/openai/v1"
        self._results: Dict[str, ClassificationVerdict] = {}
        self._lock = threading.Lock()

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def dispatch_bias_check(
        self,
        text: str,
        session_id: str = "default",
        on_complete: Optional[Callable[[str, ClassificationVerdict], None]] = None,
    ) -> str:
        """
        Async bias classification — catches implicit bias regex cannot detect.
        Returns job_id immediately. Result stored and retrievable via get_result().
        """
        return self._dispatch(
            text=text,
            task="BIAS_DETECTION",
            system_prompt=_BIAS_CLASSIFIER_PROMPT,
            session_id=session_id,
            on_complete=on_complete,
        )

    def _dispatch(
        self,
        text: str,
        task: str,
        system_prompt: str,
        session_id: str,
        on_complete: Optional[Callable],
    ) -> str:
        job_id = hashlib.sha256(f"{task}{session_id}{time.time_ns()}".encode()).hexdigest()[:12]

        def _run():
            verdict = self._call_groq(text, task, system_prompt, job_id, session_id)
            with self._lock:
                self._results[job_id] = verdict
            if on_complete:
                on_complete(job_id, verdict)

        thread = threading.Thread(target=_run, daemon=True, name=f"llm-classifier-{job_id[:6]}")
        thread.start()
        return job_id

    def _call_groq(
        self,
        text: str,
        task: str,
        system_prompt: str,
        job_id: str,
        session_id: str,
    ) -> ClassificationVerdict:
        """Synchronous Groq call — runs inside a background thread only."""
        if not self.is_configured():
            return ClassificationVerdict(
                job_id=job_id, task=task, score=0.0, confidence=0.0,
                reasoning="GROQ_API_KEY not configured — LLM classifier disabled.",
                detected_phrases=[], implicit_bias=False,
                model_used="none", completed_at=time.time(), success=False,
                error="API key missing",
            )

        eval_text = text[:1500] if len(text) > 1500 else text

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": _CLASSIFIER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classify this text:\n\n{eval_text}"},
            ],
            "temperature": 0.1,
            "max_tokens": 200,
            "stream": False,
        }

        try:
            with httpx.Client(timeout=20.0) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
            if response.status_code != 200:
                raise ValueError(f"Groq API {response.status_code}: {response.text[:120]}")

            raw_content = response.json()["choices"][0]["message"]["content"].strip()

            # Strip markdown fences
            if "```" in raw_content:
                parts = raw_content.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        raw_content = part
                        break

            start = raw_content.find("{")
            end = raw_content.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON found in classifier response")
            raw_content = raw_content[start:end]

            data = json.loads(raw_content)

            return ClassificationVerdict(
                job_id=job_id,
                task=task,
                score=min(1.0, max(0.0, float(data.get("score", 0.0)))),
                confidence=min(1.0, max(0.0, float(data.get("confidence", 0.0)))),
                reasoning=str(data.get("reasoning", "")),
                detected_phrases=list(data.get("detected_phrases", [])),
                implicit_bias=bool(data.get("implicit_bias", False)),
                model_used=_CLASSIFIER_MODEL,
                completed_at=time.time(),
                success=True,
            )

        except json.JSONDecodeError as e:
            return ClassificationVerdict(
                job_id=job_id, task=task, score=0.0, confidence=0.0,
                reasoning=f"JSON parse error: {str(e)}",
                detected_phrases=[], implicit_bias=False,
                model_used=_CLASSIFIER_MODEL, completed_at=time.time(),
                success=False, error=str(e),
            )
        except Exception as e:
            return ClassificationVerdict(
                job_id=job_id, task=task, score=0.0, confidence=0.0,
                reasoning=f"Classifier failed: {str(e)[:100]}",
                detected_phrases=[], implicit_bias=False,
                model_used=_CLASSIFIER_MODEL, completed_at=time.time(),
                success=False, error=str(e),
            )

    def get_result(self, job_id: str) -> Optional[ClassificationVerdict]:
        with self._lock:
            return self._results.get(job_id)

    def get_all_results(self) -> Dict[str, ClassificationVerdict]:
        with self._lock:
            return dict(self._results)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_classifier = LLMClassifier()


def get_llm_classifier() -> LLMClassifier:
    return _global_classifier
