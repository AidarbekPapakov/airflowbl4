"""
CS2 digest pipeline (Phase 3: anonymous speakers).

Task flow:
  discover_videos → fetch_audio.expand() → transcribe_video.expand()
    → summarize_video.expand() → has_new_summaries (short-circuit) → send_digest_email

One failed video is marked `failed` in the `videos` ledger and returns None
instead of raising, so it doesn't block the other mapped instances.
transcribe_video caps concurrency at 1 (max_active_tis_per_dag) since the
diarization stage is the bottleneck and shares the box's 8GB GPU with llama.cpp.

This is the origin copy. The deployed copy lives in
~/projects/airflow/dags/cs2_digest_dag.py, and src/cs2_digest is copied into
~/projects/airflow/src/cs2_digest -- see that repo's CLAUDE.md.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from airflow.sdk import Variable, dag, task
from airflow.utils.email import send_email

from src.cs2_digest import discover, fetch, store, summarize, transcribe
from src.cs2_digest.email import render_digest_html
from src.cs2_digest.transcribe import flatten_hotwords

logger = logging.getLogger(__name__)

# Relative to the src.cs2_digest package itself, not this DAG file: the two
# copies (this one and the deployed ~/projects/airflow copy) nest src/ under
# dags/ differently, see airflow/CLAUDE.md "Runtime wiring".
CONFIG_DIR = Path(discover.__file__).parent / "config"
WORK_DIR = Path("/tmp/cs2_digest")


def _clickhouse_client():
    return store.get_client(
        host=Variable.get("cs2_digest_clickhouse_host", "cs2_digest-clickhouse-1"),
        port=int(Variable.get("cs2_digest_clickhouse_port", "8123")),
        username=Variable.get("cs2_digest_clickhouse_user", "cs2_digest"),
        password=Variable.get("cs2_digest_clickhouse_password"),
    )


def _load_channels() -> List[Dict]:
    return yaml.safe_load((CONFIG_DIR / "channels.yaml").read_text())


def _load_hotwords() -> Dict[str, float]:
    raw = yaml.safe_load((CONFIG_DIR / "hotwords.yaml").read_text())
    return flatten_hotwords(raw)


@dag(
    dag_id="cs2_digest",
    schedule="@daily",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    tags=["cs2", "clickhouse", "llm", "asr"],
    max_active_runs=1,
)
def cs2_digest():

    @task
    def discover_videos() -> List[Dict]:
        client = _clickhouse_client()
        store.ensure_tables(client)
        channels = _load_channels()

        new_videos = []
        for channel in channels:
            if channel["channel_id"].startswith("TODO_"):
                logger.warning("Skipping %s: channel_id not confirmed yet", channel["name"])
                continue
            known = store.known_video_ids(client, channel["name"])
            entries = discover.discover_new_videos(
                channel_id=channel["channel_id"],
                known_video_ids=known,
                min_duration_s=channel.get("min_duration_s"),
                max_duration_s=channel.get("max_duration_s"),
                get_duration=fetch.fetch_duration_s,
            )
            for entry in entries:
                store.upsert_video(
                    client,
                    video_id=entry["video_id"],
                    channel=channel["name"],
                    title=entry["title"],
                    published_at=entry["published_at"],
                    duration_s=entry.get("duration_s", 0),
                    status="discovered",
                )
                new_videos.append({
                    "video_id": entry["video_id"],
                    "channel": channel["name"],
                    "title": entry["title"],
                })
        return new_videos

    @task(retries=2, retry_delay=timedelta(minutes=2))
    def fetch_audio(video: Dict) -> Optional[Dict]:
        client = _clickhouse_client()
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        try:
            audio_path = fetch.fetch_and_downmix(video["video_id"], WORK_DIR)
        except Exception as e:
            logger.exception("fetch failed for %s", video["video_id"])
            store.mark_status(client, video["video_id"], video["channel"], "failed", str(e))
            return None
        store.mark_status(client, video["video_id"], video["channel"], "fetched")
        return {**video, "audio_path": str(audio_path)}

    @task(retries=1, retry_delay=timedelta(minutes=5), max_active_tis_per_dag=1)
    def transcribe_video(fetched: Optional[Dict]) -> Optional[Dict]:
        if fetched is None:
            return None
        client = _clickhouse_client()
        hotwords = _load_hotwords()
        audio_path = Path(fetched["audio_path"])
        try:
            result = transcribe.transcribe(audio_path, hotwords)
        except Exception as e:
            logger.exception("transcribe failed for %s", fetched["video_id"])
            store.mark_status(client, fetched["video_id"], fetched["channel"], "failed", str(e))
            return None
        finally:
            fetch.cleanup(audio_path)

        store.insert_transcript(client, fetched["video_id"], result["turns"], result.get("meta", {}))
        store.mark_status(client, fetched["video_id"], fetched["channel"], "transcribed")
        return {
            "video_id": fetched["video_id"],
            "channel": fetched["channel"],
            "title": fetched["title"],
            "turns": result["turns"],
        }

    @task(retries=2, retry_delay=timedelta(minutes=2))
    def summarize_video(transcribed: Optional[Dict]) -> Optional[str]:
        if transcribed is None:
            return None
        client = _clickhouse_client()
        model = Variable.get("cs2_digest_llm_model", "local-llm")
        try:
            summary = summarize.summarize_turns(
                transcribed["turns"],
                base_url=Variable.get("cs2_digest_llm_base_url", "http://litellm:4000"),
                model=model,
                api_key=Variable.get("cs2_digest_llm_api_key"),
            )
        except Exception as e:
            logger.exception("summarize failed for %s", transcribed["video_id"])
            store.mark_status(client, transcribed["video_id"], transcribed["channel"], "failed", str(e))
            return None

        store.insert_summary(
            client, transcribed["video_id"], model,
            summary["tldr"], summary["positions"], summary["entities"],
        )
        store.mark_status(client, transcribed["video_id"], transcribed["channel"], "summarised")
        return transcribed["video_id"]

    @task.short_circuit
    def has_new_summaries(summarized_ids: List[Optional[str]]) -> bool:
        return any(v is not None for v in summarized_ids)

    @task
    def send_digest_email() -> None:
        client = _clickhouse_client()
        videos = store.videos_pending_email(client)
        if not videos:
            return

        send_email(
            to=[Variable.get("cs2_digest_recipient_email")],
            subject=f"CS2 Digest — {datetime.now().date()}",
            html_content=render_digest_html(videos),
        )
        for v in videos:
            store.mark_status(client, v["video_id"], v["channel"], "emailed")

    videos = discover_videos()
    fetched = fetch_audio.expand(video=videos)
    transcribed = transcribe_video.expand(fetched=fetched)
    summarized = summarize_video.expand(transcribed=transcribed)
    has_new_summaries(summarized) >> send_digest_email()


cs2_digest()
