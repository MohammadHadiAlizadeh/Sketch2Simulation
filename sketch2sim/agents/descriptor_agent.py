import base64
import io
import json
import os

import requests
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph
from PIL import Image
from pydantic import BaseModel

MAX_IMAGE_WIDTH = int(os.getenv("DESCRIPTOR_MAX_IMAGE_WIDTH", "2048"))
DESCRIPTOR_NUM_PREDICT = int(os.getenv("DESCRIPTOR_NUM_PREDICT", "5000"))
DESCRIPTOR_NUM_CTX = int(os.getenv("DESCRIPTOR_NUM_CTX", "21000"))
DESCRIPTOR_TIMEOUT = int(os.getenv("DESCRIPTOR_TIMEOUT", "900"))
OLLAMA_CLOUD_HOST = os.getenv("OLLAMA_CLOUD_HOST", "http://localhost:11434")


class DescriptorState(BaseModel):
    image_path: str
    description: str | None = None


def _encode_image_to_jpeg_b64(image_path: str) -> str:
    with Image.open(image_path) as img:
        img = img.convert("RGB")

        if img.width > MAX_IMAGE_WIDTH:
            scale = MAX_IMAGE_WIDTH / img.width
            img = img.resize((MAX_IMAGE_WIDTH, int(img.height * scale)))

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85, optimize=True)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _build_prompt() -> str:
    return (
        "You are a senior chemical process engineer. "
        "You are given a process flow diagram image.\n\n"
        "Return only the following sections:\n"
        "1. PROCESS DESCRIPTION\n"
        "2. EQUIPMENT AND CONNECTIONS\n"
        "3. FINAL CONSISTENCY CHECK\n\n"
        "Rules:\n"
        "- Describe the process from left to right.\n"
        "- List each visible equipment item once.\n"
        "- Do not invent hidden equipment.\n"
        "- Do not write code.\n"
    )


def call_ollama_cloud(model_name: str, prompt: str, image_b64: str) -> str:
    api_key = os.getenv("OLLAMA_API_KEY", "").strip()
    if not api_key:
        raise OSError("OLLAMA_API_KEY is not set.")

    payload = {
        "model": model_name,
        "stream": True,
        "keep_alive": "10m",
        "think": False,
        "messages": [
            {
                "role": "system",
                "content": "You are a senior chemical process engineer. Follow the user's instructions carefully.",
            },
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            },
        ],
        "options": {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": 42,
            "num_predict": DESCRIPTOR_NUM_PREDICT,
            "num_ctx": DESCRIPTOR_NUM_CTX,
        },
    }

    url = f"{OLLAMA_CLOUD_HOST.rstrip('/')}/api/chat"
    response = requests.post(
        url,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        stream=True,
        timeout=(30, DESCRIPTOR_TIMEOUT),
    )
    response.raise_for_status()

    content_chunks = []
    thinking_chunks = []

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue

        data = json.loads(line)
        message = data.get("message") or {}

        content = message.get("content")
        thinking = message.get("thinking")

        if isinstance(content, str) and content:
            content_chunks.append(content)
        if isinstance(thinking, str) and thinking:
            thinking_chunks.append(thinking)

        if data.get("done"):
            break

    return "".join(content_chunks).strip() or "".join(thinking_chunks).strip()


def run_descriptor(state: DescriptorState, model_name: str) -> dict:
    image_b64 = _encode_image_to_jpeg_b64(state.image_path)
    prompt = _build_prompt()
    description = call_ollama_cloud(model_name, prompt, image_b64)
    return {"description": description}


def build_descriptor_graph(model_name: str):
    graph = StateGraph(DescriptorState)
    graph.add_node(
        "descriptor",
        RunnableLambda(lambda state: run_descriptor(state, model_name=model_name)),
    )
    graph.set_entry_point("descriptor")
    graph.set_finish_point("descriptor")
    return graph.compile()
