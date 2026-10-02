"""Calls asr_service /analyze with CS2 hotwords, no LLM pass (run_llm=0)."""

import json
from pathlib import Path
from typing import Dict, List

import requests

DEFAULT_ASR_URL = "http://asr_service:8400/analyze"
# Generous: a 60-90 min episode can take tens of minutes on CPU diarization
# (see audio_analytics CLAUDE.md Gotchas).
DEFAULT_TIMEOUT_S = 3600

# asr_service's beam decoder takes {word: weight}; MIN_HOTWORD_LEN=5 rejects
# shorter words (e.g. "NiKo", "rain") server-side, reason reported in
# meta.hotwords.rejected. 6.0 matches the example weight in asr/src/ctc_beam.py.
DEFAULT_HOTWORD_WEIGHT = 6.0


def flatten_hotwords(hotwords_config: Dict[str, List[str]]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for names in hotwords_config.values():
        for name in names:
            flat[name] = DEFAULT_HOTWORD_WEIGHT
    return flat


def transcribe(
    audio_path: Path,
    hotwords: Dict[str, float],
    asr_url: str = DEFAULT_ASR_URL,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> Dict:
    """Returns {"turns": [...], "meta": {...}} from asr_service."""
    with open(audio_path, "rb") as f:
        resp = requests.post(
            asr_url,
            files={"audio": f},
            data={"run_llm": "0", "hotwords": json.dumps(hotwords)},
            timeout=timeout_s,
        )
    resp.raise_for_status()
    body = resp.json()
    if "error" in body:
        # Inherited API quirk: unreadable audio returns HTTP 200 with {"error": ...}.
        raise RuntimeError(f"asr_service error: {body['error']}")
    return body
