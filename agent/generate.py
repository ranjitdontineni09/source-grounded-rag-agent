"""Answer only from retrieved chunks.

Optional OpenAI-compatible / LoRA endpoint via ``LLM_URL``. Empty retrieval is
handled by the harness, not this module.
"""

from __future__ import annotations

import os

import httpx

from agent.chunk import Chunk

SYSTEM = (
    "Answer only from the provided sources. Quote them. "
    "If they are insufficient, reply with exactly REFUSE."
)


def synthesize(question: str, hits: list[Chunk]) -> str:
    """Build a grounded answer by quoting the top chunks.

    Args:
        question: User query (unused; kept so the signature matches :func:`generate`).
        hits: Retrieved sources. Must be non-empty.

    Returns:
        Concatenated ``From {doc}: {text}`` sentences.
    """
    parts = [f"From {h.doc}: {h.text}" for h in hits[:3]]
    return " ".join(parts)


def generate(question: str, hits: list[Chunk], history: list[dict]) -> str:
    """Produce an answer from ``hits`` only.

    Args:
        question: Current user question.
        hits: Retrieved chunks. Callers must not pass an empty list.
        history: Prior turns with ``question`` and ``answer`` keys.

    Returns:
        Model or synthesizer text. ``REFUSE`` if the optional LLM declines.
    """
    if os.environ.get("LLM_URL"):
        text = _llm(question, hits, history)
        if text.strip().upper().startswith("REFUSE"):
            return "REFUSE"
        return text
    return synthesize(question, hits)


def _llm(question: str, hits: list[Chunk], history: list[dict]) -> str:
    """Call an OpenAI-compatible chat endpoint with sources in the user turn.

    Args:
        question: Current user question.
        hits: Retrieved chunks included as the source block.
        history: Prior turns; last four are sent.

    Returns:
        Assistant message content.

    Raises:
        httpx.HTTPStatusError: If the remote API returns a non-2xx status.
    """
    sources = "\n\n".join(f"[{h.doc}] {h.text}" for h in hits)
    messages = [{"role": "system", "content": SYSTEM}]
    for turn in history[-4:]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append(
        {
            "role": "user",
            "content": f"Sources:\n{sources}\n\nQuestion: {question}",
        }
    )
    payload = {
        "model": os.environ.get("LLM_MODEL", "llama"),
        "messages": messages,
        "temperature": 0.1,
    }
    headers = {}
    if os.environ.get("LLM_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['LLM_API_KEY']}"
    url = os.environ["LLM_URL"].rstrip("/") + "/v1/chat/completions"
    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
