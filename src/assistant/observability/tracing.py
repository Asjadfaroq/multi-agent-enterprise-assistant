from contextlib import AbstractContextManager, nullcontext
from functools import lru_cache

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler
from langfuse.span_filter import is_langfuse_span

from assistant.config import get_settings
from assistant.guardrails.pii import mask_pii_in_data


def is_tracing_enabled() -> bool:
    settings = get_settings()
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


@lru_cache
def get_langfuse_client() -> Langfuse:
    settings = get_settings()
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        # A last line of defence: anything a trace records is PII-masked before it is exported.
        mask=mask_pii_in_data,
        # The Anthropic SDK emits its own span per request; exporting it next to the LangChain
        # generation would count every call's tokens and cost twice.
        should_export_span=is_langfuse_span,
    )


def tracing_config(user_id: str, role: str, thread_id: str) -> RunnableConfig:
    """Callbacks that record every node, LLM call and tool call, tagged with who asked."""
    if not is_tracing_enabled():
        return {}
    # The handler finds its client by public key, so the client must exist before the handler.
    get_langfuse_client()
    return {
        "callbacks": [CallbackHandler(public_key=get_settings().langfuse_public_key)],
        "metadata": {"user": user_id, "role": role, "thread_id": thread_id},
    }


def trace_attributes(user_id: str, role: str, thread_id: str) -> AbstractContextManager[object]:
    """Sets the trace's user, session and tags for everything run inside the block."""
    # Set around the whole graph run: attributes set from inside a LangChain callback do not
    # reach the async tasks LangGraph runs nodes in.
    if not is_tracing_enabled():
        return nullcontext()
    return propagate_attributes(
        user_id=user_id, session_id=thread_id, tags=[f"role:{role}"], metadata={"role": role}
    )


async def record_guardrail_event(tag: str, detail: str, config: RunnableConfig) -> None:
    """Adds a span named after the guardrail to the current trace and tags the trace with it."""
    marker = RunnableLambda(lambda event_detail: event_detail, name=tag)
    # The tag is propagated to the marker span, which belongs to the node's trace.
    tagging = propagate_attributes(tags=[tag]) if is_tracing_enabled() else nullcontext()
    with tagging:
        await marker.ainvoke(detail, config)


def flush_traces() -> None:
    if is_tracing_enabled():
        get_langfuse_client().flush()
