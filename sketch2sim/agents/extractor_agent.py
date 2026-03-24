import json
import os
from typing import Any, Dict, Optional

import requests
from pydantic import BaseModel
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda


EXTRACTOR_TIMEOUT = int(os.getenv("EXTRACTOR_TIMEOUT", "900"))
EXTRACTOR_NUM_CTX = int(os.getenv("EXTRACTOR_NUM_CTX", "21000"))
OLLAMA_CLOUD_HOST = os.getenv("OLLAMA_CLOUD_HOST", "http://localhost:11434")


class ExtractorState(BaseModel):
    process_text: str
    extraction: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def _clean_json_text(text: str) -> str:
    cleaned = (text or "").replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        return cleaned[start:end + 1]
    return cleaned or "{}"


def run_extractor(state: ExtractorState, model_name: str) -> Dict[str, Any]:
    process_text = (state.process_text or "").strip()
    if not process_text:
        return {"extraction": None, "error": "No process_text provided."}

    api_key = os.getenv("OLLAMA_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("OLLAMA_API_KEY is not set.")

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

    payload = {
        "model": model_name,
        "stream": True,
        "think": False,
        "keep_alive": "10m",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": process_text},
        ],
        "options": {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": 42,
            "num_ctx": EXTRACTOR_NUM_CTX,
        },
    }

    response = requests.post(
        f"{OLLAMA_CLOUD_HOST.rstrip('/')}/api/chat",
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        stream=True,
        timeout=(30, EXTRACTOR_TIMEOUT),
    )
    response.raise_for_status()

    chunks = []
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        data = json.loads(line)
        message = data.get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content:
            chunks.append(content)
        if data.get("done"):
            break

    try:
        extraction = json.loads(_clean_json_text("".join(chunks)))
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