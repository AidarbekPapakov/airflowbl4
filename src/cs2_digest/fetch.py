"""yt-dlp audio-only download + ffmpeg downmix to mono 16kHz.

The downloaded/downmixed audio is never stored long term: callers are
expected to transcribe it and then call cleanup().
"""

import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def fetch_duration_s(video_id: str) -> int:
    """Resolve a video's duration without downloading it. Used by discover.py."""
    result = subprocess.run(
        ["yt-dlp", "--print", "%(duration)s", f"https://www.youtube.com/watch?v={video_id}"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return int(float(result.stdout.strip()))


def download_audio(video_id: str, out_dir: Path) -> Path:
    """Download audio-only for a video. Returns the path to the downloaded file."""
    out_template = str(out_dir / f"{video_id}.%(ext)s")
    subprocess.run(
        [
            "yt-dlp", "-x", "--audio-format", "mp3",
            "-o", out_template,
            f"https://www.youtube.com/watch?v={video_id}",
        ],
        check=True,
        timeout=3600,
    )
    matches = list(out_dir.glob(f"{video_id}.*"))
    if not matches:
        raise RuntimeError(f"yt-dlp reported success but no file found for {video_id}")
    return matches[0]


def downmix_to_mono16k(src: Path, dst: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)],
        check=True,
        capture_output=True,
        timeout=600,
    )
    return dst


def fetch_and_downmix(video_id: str, work_dir: Path) -> Path:
    raw = download_audio(video_id, work_dir)
    mono_path = work_dir / f"{video_id}_mono16k.wav"
    downmix_to_mono16k(raw, mono_path)
    raw.unlink(missing_ok=True)
    return mono_path


def cleanup(*paths: Path) -> None:
    for path in paths:
        Path(path).unlink(missing_ok=True)
