import pytest

from assistant.config import get_settings
from assistant.observability.tracing import get_langfuse_client, tracing_config


@pytest.fixture
def tracing_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:9")
    get_settings.cache_clear()
    get_langfuse_client.cache_clear()
    yield
    get_settings.cache_clear()
    get_langfuse_client.cache_clear()


def test_trace_handler_records_events_inline_so_spans_keep_their_order(
    tracing_enabled: None,
) -> None:
    [handler] = tracing_config("omar", "engineer", "thread-1")["callbacks"]

    assert handler.run_inline is True


def test_no_callbacks_are_added_when_tracing_is_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    get_settings.cache_clear()
    try:
        assert tracing_config("omar", "engineer", "thread-1") == {}
    finally:
        get_settings.cache_clear()
