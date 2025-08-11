# backend/graphs/monitor.py
from __future__ import annotations
import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

try:
    from langfuse.openai import openai as lf_openai  # type: ignore
except Exception:
    lf_openai = None

_langfuse_client: Optional[Langfuse] = None
_client_lock = threading.Lock()

def get_langfuse_client() -> Optional[Langfuse]:
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client
    pub = os.getenv("LANGFUSE_PUBLIC_KEY")
    sec = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    if not (pub and sec):
        return None
    with _client_lock:
        if _langfuse_client is None:
            _langfuse_client = Langfuse(public_key=pub, secret_key=sec, host=host)
    return _langfuse_client

def get_callback() -> Optional[CallbackHandler]:
    if get_langfuse_client() is None:
        return None
    return CallbackHandler()

# Thread-local trace context
_thread_ctx = threading.local()

def begin_trace(trace_name: str, **attrs: Dict[str, Any]) -> None:
    """
    Create (or reuse) a single top-level trace for the current session/thread.
    Repeated calls with the same name will reuse the same trace object.
    """
    client = get_langfuse_client()
    if client is None:
        return
    current = getattr(_thread_ctx, "trace", None)
    if current is not None:
        # Already have a trace; do not replace.
        return
    tr = client.trace(name=str(trace_name), input=attrs if attrs else None)
    _thread_ctx.trace = tr

def get_current_trace():
    return getattr(_thread_ctx, "trace", None)

def get_traced_openai_together_client():
    if lf_openai is None:
        return None
    if get_langfuse_client() is None:
        return None
    if not os.getenv("TOGETHER_API_KEY"):
        return None
    lf_openai.langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    lf_openai.langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    lf_openai.langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    lf_openai.langfuse_enabled = True
    client = lf_openai.OpenAI(
        api_key=os.getenv("TOGETHER_API_KEY"),
        base_url="https://api.together.xyz/v1",
    )
    return client

@contextmanager
def traced_span(name: str, **attrs: Dict[str, Any]):
    """
    Create a span on the active session trace. If no trace exists yet, create one ad hoc using
    provided conversation_id or a default name.
    """
    client = get_langfuse_client()
    if client is None:
        yield
        return

    parent_trace = getattr(_thread_ctx, "trace", None)
    if parent_trace is None:
        # No trace bound yet - create a best-effort session trace
        trace_name = attrs.get("conversation_id") or attrs.get("trace_id") or "chat_session"
        parent_trace = client.trace(name=str(trace_name), input=attrs)
        _thread_ctx.trace = parent_trace

    span = parent_trace.span(name=name, metadata=attrs if attrs else None)
    try:
        yield
        span.end()
    except Exception as exc:
        span.end(error=str(exc))
        raise

def end_current_trace():
    client = get_langfuse_client()
    if client is None:
        return
    tr = getattr(_thread_ctx, "trace", None)
    if tr is not None:
        try:
            tr.end()
        finally:
            _thread_ctx.trace = None
