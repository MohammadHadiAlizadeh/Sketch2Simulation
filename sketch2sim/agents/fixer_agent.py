from collections.abc import Iterable
from pathlib import Path

from langchain_ollama import ChatOllama
from utils.agent_utils import read_text_file
from utils.logging_utils import log_text

FIXER_SYSTEM_PROMPT = """
You are a code-fixing agent.

You will receive:
1. The HYSYS scripting instructions for instantiation.
2. The HYSYS scripting instructions for configuration.
3. The unit view JSON.
4. A structured validator issues report.
5. The current Python code.

Your task:
- Fix only the issues explicitly listed in the validator issues report.
- Follow the HYSYS scripting instructions exactly.
- Prefer inserting new lines.
- If absolutely required to resolve a listed issue, you may minimally edit
  existing lines that are directly implicated by that issue.
- Do not refactor, rename unrelated objects, or reorder code unless required
  by a listed issue.
- Add missing streams or objects only if required by a listed issue.
- Return only the full corrected Python script.
- No explanations, no comments, no markdown fences, no JSON.
""".strip()


def _build_fixer_prompt(
    instantiation_instructions_text: str,
    configuration_instructions_text: str,
    unit_view_json_str: str,
    validator_issues: str,
    code_text: str,
) -> str:
    return f"""{FIXER_SYSTEM_PROMPT}

---------------------------
[INSTANTIATION_INSTRUCTIONS]
<<<
{instantiation_instructions_text}
>>>

[CONFIGURATION_INSTRUCTIONS]
<<<
{configuration_instructions_text}
>>>

[UNIT_VIEW_JSON]
<<<
{unit_view_json_str}
>>>

[VALIDATOR_ISSUES]
<<<
{validator_issues}
>>>

[CODE_PY]
<<<
{code_text}
>>>
"""


def _call_fixer_llm(
    prompt: str,
    model_name: str,
    temperature: float = 0.0,
) -> str:
    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
        num_ctx=21000,
        num_predict=5000,
        top_p=1.0,
    )

    response = llm.invoke(prompt)
    content = getattr(response, "content", response)

    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content)

    text = str(content).strip()

    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            inner = parts[1].strip()
            inner_lines = inner.splitlines()
            if inner_lines and inner_lines[0].lower().startswith("python"):
                inner = "\n".join(inner_lines[1:])
            text = inner.strip()

    return text


def fix_code_text(
    *,
    instantiation_instructions_text: str,
    configuration_instructions_text: str,
    unit_view_json_str: str,
    validator_issues: str,
    code_text: str,
    model_name: str = "qwen2.5-coder:latest",
    temperature: float = 0.0,
) -> str:
    prompt = _build_fixer_prompt(
        instantiation_instructions_text=instantiation_instructions_text,
        configuration_instructions_text=configuration_instructions_text,
        unit_view_json_str=unit_view_json_str,
        validator_issues=validator_issues,
        code_text=code_text,
    )

    return _call_fixer_llm(
        prompt=prompt,
        model_name=model_name,
        temperature=temperature,
    )


def fix_code_file(
    *,
    code_path: str | Path,
    unit_view_json_str: str,
    validator_issues: str,
    model_name: str = "qwen2.5-coder:latest",
    temperature: float = 0.0,
    log_dir: str | Path | None = None,
    instantiation_instruction_paths: Iterable[str | Path],
    configuration_instruction_paths: Iterable[str | Path],
) -> Path:
    instantiation_instructions_text = "\n\n".join(
        read_text_file(str(path)) for path in instantiation_instruction_paths
    )

    configuration_instructions_text = "\n\n".join(
        read_text_file(str(path)) for path in configuration_instruction_paths
    )

    code_path = Path(code_path)
    code_text = code_path.read_text(encoding="utf-8")

    fixed_code = fix_code_text(
        instantiation_instructions_text=instantiation_instructions_text,
        configuration_instructions_text=configuration_instructions_text,
        unit_view_json_str=unit_view_json_str,
        validator_issues=validator_issues,
        code_text=code_text,
        model_name=model_name,
        temperature=temperature,
    )

    output_path = code_path.with_name(code_path.stem + "_fixed" + code_path.suffix)
    output_path.write_text(fixed_code, encoding="utf-8")

    if log_dir is not None:
        log_text(log_dir, "runtime_fixed_code.py", fixed_code)

    print(f"[Fixer] Wrote fixed code to: {output_path}")
    return output_path
