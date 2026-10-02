import json
import os
from typing import Any

from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph
from pydantic import BaseModel

from ..utils.deepseek_client import chat_text

EXTRACTOR_MAX_TOKENS = int(os.getenv("EXTRACTOR_MAX_TOKENS", "5000"))


class ExtractorState(BaseModel):
    process_text: str
    extraction: dict[str, Any] | None = None
    error: str | None = None


def _clean_json_text(text: str) -> str:
    cleaned = (text or "").replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        return cleaned[start : end + 1]
    return cleaned or "{}"


def run_extractor(state: ExtractorState, model_name: str) -> dict[str, Any]:
    process_text = (state.process_text or "").strip()
    if not process_text:
        return {"extraction": None, "error": "No process_text provided."}

    system_prompt = (
        "You are a Process Engineering Assistant. Extract structured JSON from a chemical process description.\n\n"
        "Return only one JSON object with these keys:\n"
        "{\n"
        '  "units": [ {"id": "Str", "name": "Str", "tags": ["Str"]} ],\n'
        '  "feed_streams": [ {"id": "Str", "to_unit": "Str", "name": "Str", "connect_point": "Str"} ],\n'
        '  "intermediate_streams": [ {"id": "Str", "from_unit": "Str", "to_unit": "Str", "name": "Str", "connect_point": "Str"} ],\n'
        '  "product_streams": [ {"id": "Str", "from_unit": "Str", "name": "Str", "connect_point": "Str"} ]\n'
        "}\n\n"
        "Rules:\n"
        "- Feed streams enter the system from outside.\n"
        "- Product streams leave the system.\n"
        "- Intermediate streams connect two internal units.\n"
        '- Use exact IDs from the description. If a stream has a number, use "S" plus the number.\n'
        '- Use "overhead", "bottom", or "side" for connect_point when clear; otherwise use "unknown".\n'
        "- Exclude utilities and energy streams.\n"
        "- Return JSON only. No explanations.\n"
    )

    response_text = chat_text(
        model=model_name,
        system=system_prompt,
        user=process_text,
        temperature=0.0,
        max_tokens=EXTRACTOR_MAX_TOKENS,
    )

    try:
        extraction = json.loads(_clean_json_text(response_text))
        return {"extraction": extraction, "error": None}
    except Exception as error:
        return {"extraction": {"_error": str(error)}, "error": str(error)}


def build_extractor_graph(model_name: str):
    graph = StateGraph(ExtractorState)
    graph.add_node(
        "extractor",
        RunnableLambda(lambda state: run_extractor(state, model_name=model_name)),
    )
    graph.set_entry_point("extractor")
    graph.set_finish_point("extractor")
    return graph.compile()
