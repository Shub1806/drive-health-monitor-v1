import sys
from types import SimpleNamespace

import pytest

from src.rag import ask, llm, retrieve
from src.rag.ask import generate_explanation
from src.rag.llm import (
    CortexCompleteClient, LLMError, LLMSettings, OpenAICompatibleClient, llm_settings, make_client, parse_chat_response,
)
from src.rag.prompt import assemble_context, build_messages
from src.rag.report import build_report, evidence_only_answer
from tests.rag_helpers import KNOWLEDGE, RULE_STATS, FakeClient, FakeCursor, drive_cursor, firing_features, rule_anomaly

KEY = "sk-secret-123456"
BASE = "https://llm.example.org/v1"
MESSAGES = [{"role": "system", "content": "rules"}, {"role": "user", "content": "question"}]


class FakeResponse:
    def __init__(self, status=200, data=None, text="", bad_json=False):
        self.status_code, self._data, self.text, self._bad = status, data, text, bad_json

    def json(self):
        if self._bad:
            raise ValueError("no json")
        return self._data


class Recorder:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, url, headers, json, timeout):
        self.calls.append(dict(url=url, headers=headers, json=json, timeout=timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def chat(text, prompt=11, completion=4):
    return FakeResponse(200, {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": prompt, "completion_tokens": completion}})


def client(post, base=BASE):
    return OpenAICompatibleClient(base, "some-model", KEY, post=post)


# --- request shape ---

def test_the_request_goes_to_chat_completions_with_a_bearer_key_and_deterministic_options():
    post = Recorder(chat("hi"))

    client(post).complete(MESSAGES)

    call = post.calls[0]
    assert call["url"] == "https://llm.example.org/v1/chat/completions"
    assert call["headers"]["Authorization"] == f"Bearer {KEY}"
    assert call["json"] == {"model": "some-model", "messages": MESSAGES, "temperature": 0.0, "max_tokens": llm.DEFAULT_MAX_TOKENS}
    assert call["timeout"] == llm.DEFAULT_TIMEOUT_SECONDS


def test_the_default_answer_limit_is_2000_tokens():
    assert llm.DEFAULT_MAX_TOKENS == 2000
    assert OpenAICompatibleClient(BASE, "m", "").options["max_tokens"] == 2000
    assert CortexCompleteClient(None, "m").options["max_tokens"] == 2000
    assert make_client(settings=LLMSettings("openai_compatible", "m", "", BASE)).options["max_tokens"] == 2000


def test_the_request_uses_the_default_limit_unless_it_is_overridden():
    default = Recorder(chat("hi"))
    client(default).complete(MESSAGES)

    assert default.calls[0]["json"]["max_tokens"] == 2000


def test_an_explicit_max_tokens_override_still_wins():
    via_constructor = Recorder(chat("hi"))
    OpenAICompatibleClient(BASE, "m", KEY, max_tokens=321, post=via_constructor).complete(MESSAGES)

    via_options = Recorder(chat("hi"))
    configured = client(via_options)
    configured.options["max_tokens"] = 450
    configured.complete(MESSAGES)

    assert via_constructor.calls[0]["json"]["max_tokens"] == 321
    assert via_options.calls[0]["json"]["max_tokens"] == 450
    assert CortexCompleteClient(None, "m", max_tokens=77).options["max_tokens"] == 77


# --- optional reasoning effort ---

def local_settings(effort=""):
    return LLMSettings("openai_compatible", "gemma4:e4b", "", "http://localhost:11434/v1", effort)


def sent_body(settings):
    post = Recorder(chat("hi"))
    make_client(settings=settings, post=post).complete(MESSAGES)
    return post.calls[0]["json"]


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_reasoning_effort_is_not_sent_at_all(blank):
    body = sent_body(local_settings(blank))

    assert "reasoning_effort" not in body
    assert set(body) == {"model", "messages", "temperature", "max_tokens"}


def test_a_reasoning_effort_is_passed_through_in_the_request_body(monkeypatch):
    for name, value in (("LLM_PROVIDER", "openai_compatible"), ("LLM_MODEL", "gemma4:e4b"), ("LLM_API_KEY", ""),
                        ("LLM_BASE_URL", "http://localhost:11434/v1"), ("LLM_REASONING_EFFORT", " none ")):
        monkeypatch.setenv(name, value)
    settings = llm_settings()
    post = Recorder(chat("hi"))
    client_ = make_client(settings=settings, post=post)

    client_.complete(MESSAGES)

    assert settings.reasoning_effort == "none"
    assert post.calls[0]["json"]["reasoning_effort"] == "none"
    assert post.calls[0]["json"]["max_tokens"] == llm.DEFAULT_MAX_TOKENS        # the token limit is untouched
    assert client_.options["reasoning_effort"] == "none"                         # recorded with the other request options


def test_requests_are_unchanged_when_the_setting_is_unset(monkeypatch):
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)       # do not read a developer's own .env here
    for name, value in (("LLM_PROVIDER", "openai_compatible"), ("LLM_MODEL", "m"), ("LLM_API_KEY", KEY), ("LLM_BASE_URL", BASE)):
        monkeypatch.setenv(name, value)
    settings = llm_settings()
    post = Recorder(chat("hi"))

    make_client(settings=settings, post=post).complete(MESSAGES)

    assert settings.reasoning_effort == ""
    assert post.calls[0]["json"] == {"model": "m", "messages": MESSAGES, "temperature": 0.0, "max_tokens": llm.DEFAULT_MAX_TOKENS}
    assert make_client(settings=settings).options == {"temperature": 0.0, "max_tokens": llm.DEFAULT_MAX_TOKENS}


def test_the_direct_client_and_the_cortex_client_are_unchanged_by_the_new_option():
    direct = OpenAICompatibleClient(BASE, "m", KEY)
    cortex = make_client(FakeCursor(), LLMSettings("cortex", "m", "", "", "none"))

    assert "reasoning_effort" not in direct.options
    assert isinstance(cortex, CortexCompleteClient) and "reasoning_effort" not in cortex.options


def test_the_example_environment_file_lists_the_setting_blank():
    import pathlib

    env = {l.split("=", 1)[0]: l.split("=", 1)[1] for l in pathlib.Path(".env.example").read_text().splitlines() if "=" in l}

    assert env["LLM_REASONING_EFFORT"] == ""


def test_without_a_key_no_authorization_header_is_sent():
    post = Recorder(chat("hi"))

    OpenAICompatibleClient(BASE, "some-model", "", post=post).complete(MESSAGES)

    assert "Authorization" not in post.calls[0]["headers"]
    assert post.calls[0]["headers"]["Content-Type"] == "application/json"


def test_the_api_key_is_optional_for_a_local_server():
    local = make_client(settings=LLMSettings("openai_compatible", "gemma3:4b", "", "http://localhost:11434/v1"))

    assert isinstance(local, OpenAICompatibleClient) and local.url == "http://localhost:11434/v1/chat/completions"
    assert "api_key=none" in repr(local)


def test_an_error_body_is_not_mangled_when_there_is_no_key_to_hide():
    post = Recorder(FakeResponse(404, text='{"error":{"message":"model \'gemma3:4b\' not found"}}'))

    with pytest.raises(LLMError, match="HTTP 404.*gemma3:4b"):
        OpenAICompatibleClient(BASE, "m", "", post=post).complete(MESSAGES)


def test_a_trailing_slash_on_the_base_url_is_handled():
    post = Recorder(chat("hi"), chat("hi"))

    client(post, BASE + "/").complete(MESSAGES)

    assert post.calls[0]["url"] == "https://llm.example.org/v1/chat/completions"


def test_the_answer_text_tokens_and_latency_are_returned():
    result = client(Recorder(chat("the answer", 120, 33))).complete(MESSAGES)

    assert (result.text, result.prompt_tokens, result.completion_tokens, result.model) == ("the answer", 120, 33, "some-model")
    assert isinstance(result.latency_ms, int)


def test_missing_usage_is_allowed():
    result = client(Recorder(FakeResponse(200, {"choices": [{"message": {"content": "x"}}]}))).complete(MESSAGES)

    assert (result.prompt_tokens, result.completion_tokens) == (None, None)


# --- failures never leak the key ---

def test_an_http_error_raises_llm_error_with_the_status_and_without_the_key():
    post = Recorder(FakeResponse(401, text=f"invalid key {KEY}"))

    with pytest.raises(LLMError) as err:
        client(post).complete(MESSAGES)

    assert "HTTP 401" in str(err.value) and KEY not in str(err.value) and "<hidden>" in str(err.value)


def test_a_network_error_raises_llm_error_without_the_key():
    post = Recorder(ConnectionError(f"could not reach host using {KEY}"))

    with pytest.raises(LLMError) as err:
        client(post).complete(MESSAGES)

    assert KEY not in str(err.value) and "model call failed" in str(err.value)


@pytest.mark.parametrize("response", [
    FakeResponse(200, bad_json=True),
    FakeResponse(200, {"choices": []}),
    FakeResponse(200, {"nothing": 1}),
    FakeResponse(200, {"choices": [{"message": {"content": None}}]}),
])
def test_an_unreadable_response_raises_llm_error(response):
    with pytest.raises(LLMError, match="unreadable model response"):
        client(Recorder(response)).complete(MESSAGES)


def test_the_client_representation_hides_the_key():
    assert KEY not in repr(client(Recorder())) and "<hidden>" in repr(client(Recorder()))
    assert parse_chat_response({"choices": [{"message": {"content": "t"}}]}) == ("t", None, None)


# --- settings and provider selection ---

def test_settings_are_read_from_the_four_environment_variables(monkeypatch):
    for name, value in (("LLM_PROVIDER", " openai_compatible "), ("LLM_MODEL", "m"), ("LLM_API_KEY", KEY), ("LLM_BASE_URL", BASE)):
        monkeypatch.setenv(name, value)

    assert llm_settings() == LLMSettings("openai_compatible", "m", KEY, BASE)


def test_the_provider_is_chosen_by_name_and_no_vendor_is_built_in():
    http = make_client(settings=LLMSettings("openai_compatible", "m", KEY, BASE))
    cortex = make_client(FakeCursor(), LLMSettings("cortex", "m", "", ""))

    assert isinstance(http, OpenAICompatibleClient) and http.url == BASE + "/chat/completions"
    assert isinstance(cortex, CortexCompleteClient)
    assert llm.PROVIDERS == ("openai_compatible", "cortex")


@pytest.mark.parametrize("settings, message", [
    (LLMSettings("", "m", KEY, BASE), "is not supported"),
    (LLMSettings("somevendor", "m", KEY, BASE), "is not supported"),
    (LLMSettings("openai_compatible", "", KEY, BASE), "LLM_MODEL is not set"),
    (LLMSettings("openai_compatible", "m", KEY, ""), "LLM_BASE_URL must be set"),
    (LLMSettings("openai_compatible", "m", "", ""), "LLM_BASE_URL must be set"),
])
def test_missing_or_unsupported_settings_give_clear_errors(settings, message):
    with pytest.raises(ValueError, match=message):
        make_client(settings=settings)


def test_the_cortex_provider_still_needs_a_cursor():
    with pytest.raises(ValueError, match="needs a Snowflake cursor"):
        make_client(settings=LLMSettings("cortex", "m", "", ""))


# --- the existing validate / retry / fallback flow works unchanged with this client ---

REPORT = build_report(rule_anomaly(), firing_features(), RULE_STATS)
CONTEXT = assemble_context("why was this drive flagged?", REPORT, KNOWLEDGE)
GOOD = evidence_only_answer(REPORT, KNOWLEDGE)
BAD = GOOD + "\nThis drive will fail soon."


def test_a_valid_answer_from_the_http_client_is_accepted_after_one_call():
    post = Recorder(chat(GOOD))

    result = generate_explanation(CONTEXT, client(post), GOOD)

    assert (result.validation_status, result.attempts, len(post.calls)) == ("passed", 1, 1)
    assert post.calls[0]["json"]["messages"] == build_messages(CONTEXT)


def test_an_invalid_answer_is_retried_once_through_the_http_client():
    post = Recorder(chat(BAD), chat(GOOD))

    result = generate_explanation(CONTEXT, client(post), GOOD)

    assert (result.validation_status, len(post.calls)) == ("passed_after_retry", 2)
    assert "failure_prediction" in post.calls[1]["json"]["messages"][-1]["content"]


def test_two_invalid_answers_or_an_http_failure_fall_back_without_exposing_the_key():
    twice = generate_explanation(CONTEXT, client(Recorder(chat(BAD), chat(BAD))), GOOD)
    failing = generate_explanation(CONTEXT, client(Recorder(FakeResponse(500, text=f"boom {KEY}"))), GOOD)

    assert twice.validation_status == "fallback" and twice.answer == GOOD
    assert failing.validation_status == "fallback" and failing.attempts == 1
    assert KEY not in str(failing.errors)


# --- command line wiring ---

def test_the_cli_uses_the_local_index_and_the_configured_client(monkeypatch, capsys):
    cursor = drive_cursor([rule_anomaly()])
    connection = SimpleNamespace(cursor=lambda: cursor, close=lambda: None)

    class Index:
        fingerprint = "local-model|rev=1|dim=4"

        def __init__(self):
            self.questions = []

        def search(self, question, top_k=3):
            self.questions.append(question)
            return [retrieve.Hit(k["doc_id"], k["title"], k["text"], k["source_name"], k["source_reference"], k["source_version"], 0.9) for k in KNOWLEDGE]

    index, fake = Index(), FakeClient(GOOD)
    monkeypatch.setattr(ask.retrieve, "build_local_index", lambda: index)
    monkeypatch.setattr(ask, "make_client", lambda cursor=None: fake)

    code = ask.main(["why was drive ZHZ43WYS flagged?", "--no-store"], connect=lambda: connection)

    out = capsys.readouterr().out
    assert code == 0 and len(fake.calls) == 1 and index.questions == ["why was drive ZHZ43WYS flagged?"]
    assert "## Observed telemetry" in out and "eval_" not in out
    assert "CORTEX" not in cursor.sql_text().upper()      # no Snowflake model or embedding function is used


def test_a_dry_run_loads_no_embedding_model_and_makes_no_http_call(monkeypatch):
    cursor = drive_cursor([rule_anomaly()])
    connection = SimpleNamespace(cursor=lambda: cursor, close=lambda: None)
    monkeypatch.setattr(ask.retrieve, "build_local_index", lambda: pytest.fail("the index must not be built in a dry run"))
    monkeypatch.setattr(ask, "make_client", lambda cursor=None: pytest.fail("no client in a dry run"))

    assert ask.main(["why was drive ZHZ43WYS flagged?", "--dry-run"], connect=lambda: connection) == 0
