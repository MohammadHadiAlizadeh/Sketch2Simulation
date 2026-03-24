import os
import subprocess
from datetime import datetime
from typing import Any

from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph
from pydantic import BaseModel


class ExecutorState(BaseModel):
    code_path: str
    log_dir: str = "logs"
    result_summary: str | None = None


def run_executor(state: ExecutorState) -> dict[str, Any]:
    os.makedirs(state.log_dir, exist_ok=True)

    if not os.path.exists(state.code_path):
        raise FileNotFoundError(f"Python script not found: {state.code_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(state.log_dir, f"executor_run_{timestamp}.log")

    print(f"[Executor Agent] Executing script -> {state.code_path}")

    try:
        result = subprocess.run(
            ["python", state.code_path],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        summary = f"Timeout while executing {state.code_path}"
        with open(log_file, "w", encoding="utf-8") as file:
            file.write(summary)
        return {
            "result_summary": summary,
            "stdout": "",
            "stderr": "TimeoutExpired",
        }

    with open(log_file, "w", encoding="utf-8") as file:
        file.write("=== STDOUT ===\n")
        file.write(result.stdout or "(none)")
        file.write("\n\n=== STDERR ===\n")
        file.write(result.stderr or "(none)")
        file.write(f"\n\nExit Code: {result.returncode}\n")

    summary = (
        f"HYSYS Executor ran: {os.path.abspath(state.code_path)}\n"
        f"Exit Code: {result.returncode}\n"
        f"Log File:  {os.path.abspath(log_file)}"
    )

    if result.returncode == 0:
        summary += "\nStatus: Success"
    else:
        summary += "\nStatus: Failed (see log)"

    return {
        "result_summary": summary,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def build_executor_graph():
    graph = StateGraph(ExecutorState)
    graph.add_node("executor", RunnableLambda(run_executor))
    graph.set_entry_point("executor")
    graph.set_finish_point("executor")
    return graph.compile()
