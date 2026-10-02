import base64
import io
import os

from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph
from PIL import Image
from pydantic import BaseModel

from ..utils.deepseek_client import chat_vision

MAX_IMAGE_WIDTH = int(os.getenv("DESCRIPTOR_MAX_IMAGE_WIDTH", "2048"))
DESCRIPTOR_NUM_PREDICT = int(os.getenv("DESCRIPTOR_NUM_PREDICT", "5000"))


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


def call_descriptor_model(model_name: str, prompt: str, image_b64: str) -> str:
    return chat_vision(
        model=model_name,
        system="You are a senior chemical process engineer. Follow the user's instructions carefully.",
        user_text=prompt,
        image_b64_jpeg=image_b64,
        temperature=0.0,
        max_tokens=DESCRIPTOR_NUM_PREDICT,
    )


def run_descriptor(state: DescriptorState, model_name: str) -> dict:
    image_b64 = _encode_image_to_jpeg_b64(state.image_path)
    prompt = _build_prompt()
    description = call_descriptor_model(model_name, prompt, image_b64)
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
