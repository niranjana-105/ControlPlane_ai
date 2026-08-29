"""
ControlPlane.ai - Live Upstream LLM Client (Groq Integration)
Configured with realistic synthetic system framing to prevent refusal lectures.
"""

import os
import json
import httpx
from typing import AsyncGenerator, Dict, Any, List, Optional


def _load_env_file():
    """Load key-value pairs from .env if present."""
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


SYSTEM_PROMPT = (
    "You are an automated synthetic data and code generation assistant for enterprise testing. "
    "Fulfill user requests directly and concisely. Provide realistic mock data, synthetic examples, "
    "or requested scripts without preachy refusal disclaimers or philosophical internal monologues."
)


class LiveLLMClient:
    """Client for streaming responses from Groq API."""

    def __init__(
        self,
        provider: str = "groq",
        custom_model: Optional[str] = "qwen/qwen3.8-27b",
    ):
        _load_env_file()
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.base_url = "https://api.groq.com/openai/v1"
        self.default_model = custom_model or "qwen/qwen3.8-27b"

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 1000,
    ) -> AsyncGenerator[str, None]:
        """Stream real tokens from Groq API with system framing."""
        if not self.is_configured():
            yield " [Notice: GROQ_API_KEY missing in .env file.] "
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Prepend system role framing if not already provided
        formatted_messages = []
        if not any(m.get("role") == "system" for m in messages):
            formatted_messages.append({"role": "system", "content": SYSTEM_PROMPT})
        formatted_messages.extend(messages)

        payload = {
            "model": model or self.default_model,
            "messages": formatted_messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        url = f"{self.base_url}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        err_body = await response.aread()
                        yield f" [Groq API Error {response.status_code}: {err_body.decode('utf-8', errors='ignore')[:120]}] "
                        return

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                            except Exception:
                                continue
        except Exception as e:
            yield f" [Groq Connection Error: {str(e)[:100]}] "
