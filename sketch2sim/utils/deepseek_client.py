import os

import httpx
from openai import OpenAI

#: Connection/read budget for DeepSeek calls. Script generation is slow, so the
#: read timeout is deliberately generous.
_TIMEOUT = httpx.Timeout(connect=30.0, read=900.0, write=300.0, pool=30.0)

_TRUE_VALUES = {"1", "true", "yes", "on"}


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
        max_tokens=max_tokens,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return (resp.choices[0].message.content or "").strip()


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
        max_tokens=max_tokens,
        extra_body={"thinking": {"type": "enabled"},
        "reasoning_effort": "max"},
    )
    return (resp.choices[0].message.content or "").strip()


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
        max_tokens=max_tokens,
        top_p=top_p,
        api_key=api_key,
        base_url="https://api.deepseek.com",
        http_client=build_http_client(),
        # Keep langchain-openai from replacing the transport we configured.
        http_socket_options=(),
        extra_body={"thinking": {"type": "disabled"}},
    )
