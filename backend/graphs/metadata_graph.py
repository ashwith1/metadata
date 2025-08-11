# backend/graphs/metadata_graph.py
import logging
import time
from typing import Any, Dict, Optional
from backend.graphs.monitor import get_callback, traced_span
from backend.llm.metadata_chain import metadata_chain

logger = logging.getLogger(__name__)

class MetadataGraphError(Exception):
    pass

_EMPTY_META: Dict[str, Any] = {
    "Module Name": None,
    "ECTS Credits": None,
    "Instructor Name(s)": None,
    "Course Duration": None,
    "Prerequisites": None,
    "Language of Instruction": None,
    "Course Objective": None,
    "Assessment Methods": None,
}

def run_metadata_graph(
    *,
    text: str,
    llm_model: str,
    callbacks: Optional[list] = None,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    span_attrs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not text.strip():
        raise MetadataGraphError("`text` must be a non-empty string")
    if not llm_model.strip():
        raise MetadataGraphError("`llm_model` must be provided")

    try:
        _ = get_callback()
    except Exception:
        logger.warning("Could not initialize Langfuse callback; proceeding without tracing")

    attempt = 0
    metadata: Dict[str, Any]
    attrs = {"llm_model": llm_model, "action": "metadata_extraction"}
    if span_attrs:
        attrs.update(span_attrs)

    with traced_span("metadata_extraction", **attrs):
        while True:
            attempt += 1
            try:
                metadata = metadata_chain.run({
                    "text": text,
                    "llm_model": llm_model,
                })
                break
            except Exception as e:
                msg = str(e)
                if "Error 503" in msg and attempt < max_retries:
                    wait = backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "Together 503 (attempt %d/%d). Retrying after %.1f s…",
                        attempt, max_retries, wait
                    )
                    time.sleep(wait)
                    continue
                logger.exception("Metadata extraction failed on attempt %d", attempt)
                metadata = {**_EMPTY_META, "Module Name": "Extraction unavailable"}
                break

        if not isinstance(metadata, dict):
            logger.error("Unexpected metadata type %s; using empty defaults", type(metadata))
            metadata = {**_EMPTY_META, "Module Name": "Extraction error"}

        return metadata
