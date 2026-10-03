import os

import httpx
from openai import OpenAI

#: Connection/read budget for DeepSeek calls. Script generation is slow, so the
#: read timeout is deliberately generous.
_TIMEOUT = httpx.Timeout(connect=30.0, read=900.0, write=300.0, pool=30.0)

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}

#: One switch for the whole pipeline. When thinking is enabled DeepSeek emits
#: `reasoning_content` (chain-of-thought) BEFORE `content` (the answer), and
#: BOTH count against `max_tokens`. A cap sized for the answer alone therefore
#: starves the answer -> empty/truncated output. The API default is 8K tokens
#: non-thinking vs 64K thinking, so mirror that here.
_THINKING_MAX_TOKENS = int(os.getenv("DEEPSEEK_THINKING_MAX_TOKENS", "64000"))


def _thinking_enabled() -> bool:
    return os.getenv("DEEPSEEK_THINKING", "").strip().lower() in _TRUE_VALUES


def _thinking_extra() -> dict:
    """Body fragment selecting the thinking / non-thinking model."""
    return {"thinking": {"type": "enabled" if _thinking_enabled() else "disabled"}}


def _resolve_max_tokens(max_tokens: int) -> int:
    """Give thinking mode room for CoT + answer; never shrink the caller's cap."""
    if _thinking_enabled():
        return max(max_tokens, _THINKING_MAX_TOKENS)
    return max_tokens


def _final_answer(resp) -> str:
    """Return the assistant's final answer.

    Fails loudly instead of silently returning an empty string: when thinking
    is enabled the chain-of-thought can consume the whole budget, leaving
    `content` empty, which downstream agents read as "no content" and turn into
    an empty structure document.
    """
    choice = resp.choices[0]
    message = choice.message
    content = (message.content or "").strip()
    if content:
        return content

    reasoning = (getattr(message, "reasoning_content", None) or "").strip()
    if reasoning and choice.finish_reason != "length":
        return reasoning

    raise RuntimeError(
        "DeepSeek returned no final answer "
        f"(finish_reason={choice.finish_reason!r}). "
        "In thinking mode the chain-of-thought shares `max_tokens` with the "
        "answer; raise DEEPSEEK_THINKING_MAX_TOKENS or set DEEPSEEK_THINKING=0."
    )


def build_http_client() -> httpx.Client:
    """Build the HTTP client used for every DeepSeek request.

    Environment proxy settings (``HTTP_PROXY`` / ``HTTPS_PROXY`` / ``ALL_PROXY``)
    are ignored by default. A locally installed proxy that re-signs TLS traffic
    breaks certificate verification with "self-signed certificate in certificate
    chain", which surfaces as an ``APIConnectionError``. Set
    ``DEEPSEEK_USE_ENV_PROXY=1`` to honour the environment proxy instead.
    """
    trust_env = os.getenv("DEEPSEEK_USE_ENV_PROXY", "").strip().lower() in _TRUE_VALUES
    return httpx.Client(trust_env=trust_env, timeout=_TIMEOUT)


def get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise OSError("DEEPSEEK_API_KEY is not set.")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        http_client=build_http_client(),
    )


def chat_text(
    *,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 20000,
) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=_resolve_max_tokens(max_tokens),
        extra_body=_thinking_extra(),
    )
    return _final_answer(resp)


def chat_vision(
    *,
    model: str,
    system: str,
    user_text: str,
    image_b64_jpeg: str,
    temperature: float = 0.0,
    max_tokens: int = 20000,
) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64_jpeg}"
                        },
                    },
                ],
            },
        ],
        temperature=temperature,
        max_tokens=_resolve_max_tokens(max_tokens),
        extra_body=_thinking_extra(),
    )
    return _final_answer(resp)


def build_llm(
    model_name: str,
    *,
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
):
    """Build the LangChain chat model used by the script-generating agents.

    Drop-in replacement for ``langchain_ollama.ChatOllama`` that talks to
    DeepSeek instead of a local Ollama server.  Ollama-only options such as
    ``num_ctx`` have no equivalent and are replaced by ``max_tokens``.
    """
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise OSError("DEEPSEEK_API_KEY is not set.")

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=_resolve_max_tokens(max_tokens),
        top_p=top_p,
        api_key=api_key,
        base_url="https://api.deepseek.com",
        http_client=build_http_client(),
        # Keep langchain-openai from replacing the transport we configured.
        http_socket_options=(),
        extra_body=_thinking_extra(),
    )
