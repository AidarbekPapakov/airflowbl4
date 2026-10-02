"""Map-reduce summarization over ASR turns via an OpenAI-compatible endpoint.

Uses `requests` directly against the chat/completions endpoint rather than
the `openai` SDK: LiteLLM's endpoint is a plain HTTP API, and pulling in the
SDK for one call type isn't worth the extra dependency (Simplicity First).
"""

import json
import logging
import re
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://litellm:4000"
DEFAULT_MODEL = "local-llm"
# Qwen3's context is 16k; ~10k tokens of transcript leaves room for the
# prompt and completion.
CHUNK_TOKEN_BUDGET = 10_000
# Rough chars-per-token for English transcript text; good enough for chunking,
# not for billing.
CHARS_PER_TOKEN = 4

MAP_PROMPT = """You are summarising part of a CS2 (Counter-Strike 2) esports commentary video.
The transcript below is machine-transcribed and may contain typos. Speakers
are labelled speaker_0, speaker_1, etc.

Transcript chunk {chunk_num}/{total_chunks}:
{transcript}

Return ONLY valid JSON with this shape:
{{
  "summary": "2-4 sentence summary of this chunk",
  "positions": {{"speaker_0": "what this speaker argued/said in this chunk", ...}},
  "entities": {{"teams": [...], "players": [...], "events": [...]}}
}}"""

REDUCE_PROMPT = """You are merging partial summaries of a CS2 esports commentary video
into one final digest entry. Here are the partial summaries, in order:

{partials}

Return ONLY valid JSON with this shape:
{{
  "tldr": "2-3 sentence overall summary",
  "positions": {{"speaker_0": "this speaker's overall position/argument across the video", ...}},
  "entities": {{"teams": [...], "players": [...], "events": [...]}}
}}
Merge duplicate entities and per-speaker positions rather than concatenating them."""


def _strip_think(content: str) -> str:
    """Strip Qwen3 thinking-mode <think> blocks, as asr/src/normalize._extract_json does."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    return content


def _chat_json(
    prompt: str, base_url: str, model: str, api_key: str, timeout_s: int = 180
) -> Dict:
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(_strip_think(content))


def chunk_turns(turns: List[Dict], token_budget: int = CHUNK_TOKEN_BUDGET) -> List[List[Dict]]:
    """Split turns into chunks of about `token_budget` tokens each, never
    splitting a single turn across chunks."""
    char_budget = token_budget * CHARS_PER_TOKEN
    chunks: List[List[Dict]] = []
    current: List[Dict] = []
    current_chars = 0
    for turn in turns:
        turn_chars = len(turn["text"]) + len(turn["speaker"]) + 4
        if current and current_chars + turn_chars > char_budget:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(turn)
        current_chars += turn_chars
    if current:
        chunks.append(current)
    return chunks or [[]]


def summarize_turns(
    turns: List[Dict],
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    api_key: str = "",
) -> Dict:
    """Returns {"tldr": str, "positions": {...}, "entities": {...}}."""
    chunks = chunk_turns(turns, token_budget=CHUNK_TOKEN_BUDGET)

    if len(chunks) == 1:
        transcript = "\n".join(f"[{t['speaker']}] {t['text']}" for t in chunks[0])
        prompt = MAP_PROMPT.format(chunk_num=1, total_chunks=1, transcript=transcript)
        partial = _chat_json(prompt, base_url, model, api_key)
        return {
            "tldr": partial["summary"],
            "positions": partial["positions"],
            "entities": partial["entities"],
        }

    partials = []
    for i, chunk in enumerate(chunks, start=1):
        transcript = "\n".join(f"[{t['speaker']}] {t['text']}" for t in chunk)
        prompt = MAP_PROMPT.format(chunk_num=i, total_chunks=len(chunks), transcript=transcript)
        partials.append(_chat_json(prompt, base_url, model, api_key))

    reduce_prompt = REDUCE_PROMPT.format(partials=json.dumps(partials, indent=2))
    return _chat_json(reduce_prompt, base_url, model, api_key)
