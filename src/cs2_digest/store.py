"""ClickHouse ledger (videos) and data tables (transcripts, summaries).

JSON-valued columns are stored as String (not ClickHouse's JSON type, which
is still unsettled in 24.3) and (de)serialised in Python at the boundary.
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

import clickhouse_connect

DDL = [
    """
    CREATE TABLE IF NOT EXISTS videos (
        video_id String,
        channel String,
        title String,
        published_at DateTime,
        duration_s UInt32,
        status Enum8('discovered' = 1, 'fetched' = 2, 'transcribed' = 3,
                      'summarised' = 4, 'emailed' = 5, 'failed' = 6),
        error String DEFAULT '',
        updated_at DateTime
    ) ENGINE = ReplacingMergeTree(updated_at)
    ORDER BY video_id
    """,
    """
    CREATE TABLE IF NOT EXISTS transcripts (
        video_id String,
        turns String,
        asr_meta String,
        created_at DateTime DEFAULT now()
    ) ENGINE = MergeTree
    ORDER BY video_id
    """,
    """
    CREATE TABLE IF NOT EXISTS summaries (
        video_id String,
        model String,
        tldr String,
        positions String,
        entities String,
        created_at DateTime DEFAULT now()
    ) ENGINE = MergeTree
    ORDER BY video_id
    """,
]


def get_client(host: str, port: int, username: str, password: str, database: str = "cs2_digest"):
    return clickhouse_connect.get_client(
        host=host, port=port, username=username, password=password, database=database
    )


def ensure_tables(client) -> None:
    for ddl in DDL:
        client.command(ddl)


def known_video_ids(client, channel: str) -> set:
    """IDs discover_new_videos should treat as already seen, i.e. NOT
    re-discover and reprocess.

    Only the two terminal statuses count: 'emailed' (done) and 'failed'
    (permanent, not auto-retried). A video stuck at discovered/fetched/
    transcribed/summarised -- because the run that would have advanced it
    crashed, or infra was broken, not because the video itself is bad -- is
    treated as not-yet-seen, so the next run re-discovers and reprocesses it
    from scratch instead of leaving it orphaned forever.
    """
    rows = client.query(
        "SELECT video_id FROM videos FINAL WHERE channel = {channel:String} "
        "AND status IN ('emailed', 'failed')",
        parameters={"channel": channel},
    ).result_rows
    return {row[0] for row in rows}


def upsert_video(
    client,
    video_id: str,
    channel: str,
    title: str,
    published_at: datetime,
    duration_s: int,
    status: str,
    error: str = "",
) -> None:
    client.insert(
        "videos",
        [[video_id, channel, title, published_at, duration_s, status, error,
          datetime.now(timezone.utc)]],
        column_names=["video_id", "channel", "title", "published_at", "duration_s",
                       "status", "error", "updated_at"],
    )


def mark_status(client, video_id: str, channel: str, status: str, error: str = "") -> None:
    """Re-fetches the video's current row to carry its fields forward, then
    inserts a new version (ReplacingMergeTree dedups by video_id on merge)."""
    rows = client.query(
        "SELECT title, published_at, duration_s FROM videos FINAL WHERE video_id = {vid:String}",
        parameters={"vid": video_id},
    ).result_rows
    if not rows:
        raise ValueError(f"video {video_id} not in ledger; call upsert_video first")
    title, published_at, duration_s = rows[0]
    upsert_video(client, video_id, channel, title, published_at, duration_s, status, error)


def insert_transcript(client, video_id: str, turns: List[Dict], asr_meta: Dict) -> None:
    client.insert(
        "transcripts",
        [[video_id, json.dumps(turns), json.dumps(asr_meta)]],
        column_names=["video_id", "turns", "asr_meta"],
    )


def insert_summary(
    client, video_id: str, model: str, tldr: str, positions: Dict, entities: Dict
) -> None:
    client.insert(
        "summaries",
        [[video_id, model, tldr, json.dumps(positions), json.dumps(entities)]],
        column_names=["video_id", "model", "tldr", "positions", "entities"],
    )


def videos_pending_email(client) -> List[Dict]:
    """Videos summarised but not yet emailed, joined with their summary, for
    the email task. One row per video."""
    rows = client.query("""
        SELECT v.video_id, v.channel, v.title, s.tldr, s.positions, s.entities
        FROM videos AS v FINAL
        INNER JOIN summaries AS s ON v.video_id = s.video_id
        WHERE v.status = 'summarised'
        ORDER BY v.channel, v.published_at
    """).result_rows
    return [
        {
            "video_id": r[0],
            "channel": r[1],
            "title": r[2],
            "tldr": r[3],
            "positions": json.loads(r[4]),
            "entities": json.loads(r[5]),
        }
        for r in rows
    ]
