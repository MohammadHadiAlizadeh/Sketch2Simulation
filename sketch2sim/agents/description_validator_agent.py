import json
import os
import re
from typing import Any

import requests
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

DESCRIPTION_VALIDATOR_TIMEOUT = int(os.getenv("DESCRIPTION_VALIDATOR_TIMEOUT", "900"))
OLLAMA_CLOUD_HOST = os.getenv("OLLAMA_CLOUD_HOST", "http://localhost:11434")


class DescriptionValidatorState(BaseModel):
    description: str = Field(
        ..., description="Raw description produced by the descriptor agent."
    )
    validation: dict[str, Any] | None = None


DESCRIPTION_VALIDATOR_SYSTEM_PROMPT = """
You are a Description Validator Agent for process-flow diagram descriptions.

You receive JSON with:
- "description": text written by another model.

Your job:
1) Evaluate only the given description. Do not invent new equipment, streams, or numbers.
2) Detect severe issues:
   - repetition loops
   - nonsense or self-contradictions
   - confusing units, such as using kW like a flowrate or kmol/hr like a heat duty
3) Produce a lightly cleaned version:
   - remove obvious repeated blocks
   - keep it short and coherent
   - do not add new facts

Return one JSON object with exact keys:

{
  "status": "ok" | "needs_revision" | "invalid",
  "issues": [
    {"type": "short_code", "message": "<= 40 words"}
  ],
  "suggested_fix_mode": "minor_edits" | "regenerate",
  "short_feedback_for_upstream_model": "1-2 sentences",
  "cleaned_description": "cleaned text or empty string"
}

Rules:
- Output must be valid JSON only.
- No markdown.
- No extra keys.
"""


def _safe_json_loads(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    candidate = match.group(0) if match else cleaned

    try:
        data = json.loads(candidate)
    except Exception:
        return {
            "status": "invalid",
            "issues": [
                {
                    "type": "malformed_json",
                    "message": "Validator did not return valid JSON.",
                }
            ],
            "suggested_fix_mode": "regenerate",
            "short_feedback_for_upstream_model": "Validator output was malformed JSON; regenerate the description.",
            "cleaned_description": "",
        }

    allowed_keys = {
        "status",
        "issues",
        "suggested_fix_mode",
        "short_feedback_for_upstream_model",
        "cleaned_description",
    }
    data = {key: value for key, value in data.items() if key in allowed_keys}

    data.setdefault("status", "needs_revision")
    data.setdefault("issues", [])
    data.setdefault("suggested_fix_mode", "minor_edits")
    data.setdefault("short_feedback_for_upstream_model", "")
    data.setdefault("cleaned_description", "")

    return data


def run_description_validator(
    state: DescriptionValidatorState, model_name: str
) -> dict[str, Any]:
    api_key = os.getenv("OLLAMA_API_KEY", "").strip()
    if not api_key:
        raise OSError("OLLAMA_API_KEY is not set.")

    payload = {
        "model": model_name,
        "stream": True,
        "keep_alive": "10m",
        "think": False,
        "messages": [
            {"role": "system", "content": DESCRIPTION_VALIDATOR_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"description": state.description})},
        ],
        "options": {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": 42,
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
        timeout=(30, DESCRIPTION_VALIDATOR_TIMEOUT),
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

    return {"validation": _safe_json_loads("".join(chunks).strip())}


def build_description_validator_graph(model_name: str):
    graph = StateGraph(DescriptionValidatorState)
    graph.add_node(
        "description_validator",
        RunnableLambda(
            lambda state: run_description_validator(state, model_name=model_name)
        ),
    )
    graph.set_entry_point("description_validator")
    graph.set_finish_point("description_validator")
    return graph.compile()
