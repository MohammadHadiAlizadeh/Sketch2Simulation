import os
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph
from pydantic import BaseModel

from ..utils.deepseek_client import build_llm

#: DeepSeek has no Ollama-style context/output settings, so the generation
#: budget is controlled directly.
CONFIGURATION_MAX_TOKENS = int(os.getenv("CONFIGURATION_MAX_TOKENS", "5000"))

_STEP6_HEADER_RE = re.compile(
    r"(?im)^\s*#\s*===\s*Step\s*6:\s*Connect\s*Streams\s*\(Agent\s*3\)\s*===\s*$"
)
_STEP7_HEADER_RE = re.compile(
    r"(?im)^\s*#\s*===\s*Step\s*7:\s*Save\s*Simulation\s*===\s*$"
)


class ConfigurationState(BaseModel):
    description: str
    instantiation_path: str
    instruction_paths: list[str]
    python_code: str | None = None
    summary: str | None = None
    output_path: str | None = None
    prompt_debug: str | None = None
    raw_output: str | None = None


def _load_text(path: str) -> str:
    with open(path, encoding="utf-8") as file:
        return file.read()


def _load_json_text(maybe_path_or_json: str) -> str:
    text = (maybe_path_or_json or "").strip()
    if text and os.path.exists(text):
        return Path(text).read_text(encoding="utf-8", errors="ignore")
    return text


def _load_instructions(paths: list[str]) -> str:
    if not paths:
        return ""

    parts: list[str] = []
    for path in paths:
        if path and os.path.exists(path):
            parts.append(_load_text(path).strip())
        elif path:
            raise FileNotFoundError(f"Instruction file not found: {path}")

    return "\n\n".join(parts)


def _extract_first_code_block(text: str) -> str:
    if not text:
        return ""

    match = re.search(
        r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
    )
    return match.group(1).strip() if match else text.strip()


def _extract_step_6_only(code: str) -> str:
    if not code:
        return ""

    code = code.replace("\r\n", "\n")
    step_6_match = _STEP6_HEADER_RE.search(code)
    if not step_6_match:
        return code.strip()

    start = step_6_match.start()
    step_7_match = _STEP7_HEADER_RE.search(code, pos=step_6_match.end())
    end = step_7_match.start() if step_7_match else len(code)

    return code[start:end].strip()


def _extract_code(text: str) -> str:
    return _extract_step_6_only(_extract_first_code_block(text))


def _build_system_prompt() -> str:
    return (
        "You are the Configuration Agent.\n\n"
        "Your only task is to connect streams exactly as defined in the provided units_unit_view JSON.\n\n"
        "Rules:\n"
        "1. For each unit in units_unit_view:\n"
        "   - For every item in streams_out, connect this stream from the current unit to to_unit.\n"
        "   - If to_unit is null, do not connect downstream.\n"
        "   - For every item in streams_in, connect this stream to to_unit.\n"
        "   - If from_unit is null, do not connect upstream.\n"
        "2. Do not create any connection that is not explicitly listed in the JSON.\n"
        "3. Do not skip any connection that is explicitly listed in the JSON.\n"
        "4. Do not create new streams or new units.\n\n"
        "Implementation:\n"
        "- Use the existing unit and stream objects already created in the Instantiation script.\n"
        "- Match objects using their IDs.\n"
        "- Only append connection code.\n"
        "- Use connect_point to decide if stream is liquid (bottom) or gas (overhead) for multiphase flow.\n\n"
        "Output:\n"
        "- Return only the full Python script.\n"
        "- No explanations. No markdown. No extra text.\n\n"
        "# === Step 6: Connect Streams (Agent 3) ===\n"
    ).strip()


def _build_user_prompt(
    description: str, instantiation_code: str, instructions: str
) -> str:
    return (
        "STRUCTURED_JSON:\n"
        "----------------\n"
        f"{description}\n\n"
        "IMPORTANT:\n"
        "- Only create connections that are explicitly listed under STRUCTURED_JSON.connections.\n"
        "- If STRUCTURED_JSON.connections is missing or empty, do not connect anything.\n\n"
        "CURRENT PYTHON SCRIPT (AFTER INSTANTIATION):\n"
        "----------------\n"
        "```python\n"
        f"{instantiation_code}\n"
        "```\n\n"
        "INSTRUCTION BLOCK:\n"
        "----------------\n"
        "[INSTRUCTIONS START]\n"
        f"{instructions}\n"
        "[INSTRUCTIONS END]\n"
    ).strip()


def _replace_step_6(instantiation_code: str, step_6_code: str) -> str:
    pattern = (
        r"(?s)#\s*===\s*Step\s*6:\s*Connect\s*Streams\s*\(Agent\s*3\)\s*===.*?"
        r"#\s*===\s*Step\s*7:\s*Save\s*Simulation\s*==="
    )
    if not re.search(pattern, instantiation_code, flags=re.IGNORECASE):
        raise ValueError("Could not find Step 6 block markers in Instantiation code.")

    replacement = step_6_code.rstrip() + "\n\n# === Step 7: Save Simulation ==="
    return re.sub(
        pattern, lambda _: replacement, instantiation_code, count=1, flags=re.IGNORECASE
    )


def generate_configuration_script(
    state: ConfigurationState, model_name: str
) -> dict[str, Any]:
    llm = build_llm(
        model_name,
        max_tokens=CONFIGURATION_MAX_TOKENS,
        temperature=0.0,
    )

    instantiation_code = _load_text(state.instantiation_path)
    instructions = _load_instructions(state.instruction_paths)
    structured_json_text = _load_json_text(state.description)

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(
        description=structured_json_text,
        instantiation_code=instantiation_code,
        instructions=instructions,
    )

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    raw_output = response.content or ""

    step_6_code = _extract_code(raw_output)
    full_script = _replace_step_6(instantiation_code, step_6_code)

    return {
        "python_code": full_script,
        "raw_output": raw_output,
        "prompt_debug": system_prompt + "\n\n" + user_prompt,
        "summary": "Configuration script generated and merged into full file.",
        "output_path": None,
    }


def build_configuration_graph(model_name: str):
    graph = StateGraph(ConfigurationState)
    graph.add_node(
        "configuration",
        RunnableLambda(lambda state: generate_configuration_script(state, model_name)),
    )
    graph.set_entry_point("configuration")
    graph.set_finish_point("configuration")
    return graph.compile()
