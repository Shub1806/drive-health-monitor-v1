"""Language model access.

The provider and model come from the environment:

    LLM_PROVIDER   openai_compatible (any service that offers an HTTPS chat-completions endpoint)
                   or cortex (a model run inside Snowflake, where the account allows it)
    LLM_MODEL      the model name to send
    LLM_API_KEY    the key for the HTTPS endpoint, sent only as a bearer token. Optional: a local server
                   such as Ollama ignores it, and when it is empty no Authorization header is sent
    LLM_BASE_URL   the endpoint's base URL (required for openai_compatible)
    LLM_REASONING_EFFORT  optional; when set (for example "none") it is sent as `reasoning_effort`, which
                   lets a thinking model be told not to spend its token budget on hidden reasoning.
                   When blank, the field is not sent at all

No vendor is built in: the HTTPS adapter works with whichever service the base URL points at.
"""
import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from dotenv import load_dotenv

from src.rag.prompt import messages_json

# A four-section answer for an Isolation Forest report needed 1,638 tokens with a local 8B model, and a retry
# answer can be longer. A lower limit cuts the answer off (finish_reason "length"), so its last section goes
# missing and it fails validation for that reason alone. 2,000 leaves about 20% headroom and, with the largest
# retry prompt seen so far (about 3,600 tokens), stays well inside an 8,192-token context.
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TIMEOUT_SECONDS = 60
PROVIDERS = ("openai_compatible", "cortex")


class LLMError(Exception):
    """The model call failed or returned something that could not be read."""


@dataclass
class LLMResult:
    text: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    latency_ms: int
    model: str


def parse_complete_response(raw) -> tuple:
    """Pull the answer text and token counts out of an in-Snowflake model response."""
    try:
        data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        choice = data["choices"][0]
        text = choice.get("messages")
        if text is None:
            text = choice["message"]["content"]
        usage = data.get("usage", {})
        return str(text), usage.get("prompt_tokens"), usage.get("completion_tokens")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMError(f"unreadable model response: {exc}") from exc


def parse_chat_response(data) -> tuple:
    """Answer text and token counts from a chat-completions style response."""
    try:
        text = data["choices"][0]["message"]["content"]
        if not isinstance(text, str):
            raise TypeError("message content is not text")
        usage = data.get("usage") or {}
        return text, usage.get("prompt_tokens"), usage.get("completion_tokens")
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"unreadable model response: {exc}") from exc


class CortexCompleteClient:
    SQL = "SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, PARSE_JSON(%s), PARSE_JSON(%s))"

    def __init__(self, cursor, model: str, max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = 0.0):
        self.cursor = cursor
        self.model = model
        self.options = {"temperature": temperature, "max_tokens": max_tokens}

    def complete(self, messages: Sequence[dict]) -> LLMResult:
        started = time.time()
        try:
            self.cursor.execute(self.SQL, (self.model, messages_json(messages), json.dumps(self.options)))
            raw = self.cursor.fetchone()[0]
        except Exception as exc:
            raise LLMError(f"model call failed: {exc}") from exc
        text, prompt_tokens, completion_tokens = parse_complete_response(raw)
        return LLMResult(text, prompt_tokens, completion_tokens, int((time.time() - started) * 1000), self.model)


class OpenAICompatibleClient:
    """Chat-completions over HTTPS: POST <base_url>/chat/completions, with a bearer key when one is set."""

    def __init__(self, base_url: str, model: str, api_key: str, max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = 0.0, timeout: int = DEFAULT_TIMEOUT_SECONDS,
                 post: Optional[Callable] = None, reasoning_effort: Optional[str] = None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.options = {"temperature": temperature, "max_tokens": max_tokens}
        if reasoning_effort:
            self.options["reasoning_effort"] = reasoning_effort
        self.timeout = timeout
        self._api_key = api_key
        self._post = post

    def __repr__(self) -> str:
        key = "<hidden>" if self._api_key else "none"
        return f"OpenAICompatibleClient(url={self.url!r}, model={self.model!r}, api_key={key})"

    def _http_post(self):
        if self._post is not None:
            return self._post
        import requests

        return requests.post

    def _scrub(self, text: str) -> str:
        return text.replace(self._api_key, "<hidden>") if self._api_key else text

    def complete(self, messages: Sequence[dict]) -> LLMResult:
        started = time.time()
        payload = {"model": self.model, "messages": list(messages), **self.options}
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            response = self._http_post()(self.url, headers=headers, json=payload, timeout=self.timeout)
        except Exception as exc:
            raise LLMError(f"model call failed: {self._scrub(str(exc))[:300]}") from exc
        if response.status_code != 200:
            raise LLMError(f"model call failed with HTTP {response.status_code}: {self._scrub(str(response.text))[:300]}")
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError("unreadable model response: the body is not JSON") from exc
        text, prompt_tokens, completion_tokens = parse_chat_response(data)
        return LLMResult(text, prompt_tokens, completion_tokens, int((time.time() - started) * 1000), self.model)


@dataclass
class LLMSettings:
    provider: str
    model: str
    api_key: str
    base_url: str
    reasoning_effort: str = ""


def llm_settings() -> LLMSettings:
    load_dotenv()
    env = os.environ.get
    return LLMSettings(
        provider=env("LLM_PROVIDER", "").strip(),
        model=env("LLM_MODEL", "").strip(),
        api_key=env("LLM_API_KEY", "").strip(),
        base_url=env("LLM_BASE_URL", "").strip(),
        reasoning_effort=env("LLM_REASONING_EFFORT", "").strip(),
    )


def make_client(cursor=None, settings: Optional[LLMSettings] = None, post: Optional[Callable] = None):
    """Build the client named by LLM_PROVIDER, raising a clear error for anything missing."""
    s = settings or llm_settings()
    if s.provider not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER '{s.provider}' is not supported; use one of {list(PROVIDERS)}")
    if not s.model:
        raise ValueError("LLM_MODEL is not set")
    if s.provider == "cortex":
        if cursor is None:
            raise ValueError("the cortex provider needs a Snowflake cursor")
        return CortexCompleteClient(cursor, s.model)
    if not s.base_url:
        raise ValueError("LLM_BASE_URL must be set for LLM_PROVIDER=openai_compatible")
    return OpenAICompatibleClient(s.base_url, s.model, s.api_key, post=post, reasoning_effort=s.reasoning_effort.strip() or None)
