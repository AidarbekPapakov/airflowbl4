"""RSS discovery: find new videos for a channel that aren't in the ledger yet."""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from xml.etree import ElementTree

import requests

logger = logging.getLogger(__name__)

RSS_URL = "https://www.youtube.com/feeds/videos.xml"

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "yt": "http://www.youtube.com/xml/schemas/2015",
}


def fetch_rss(channel_id: str, timeout: int = 15) -> str:
    resp = requests.get(RSS_URL, params={"channel_id": channel_id}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_rss(xml_text: str) -> List[Dict]:
    """Parse a YouTube channel RSS feed into [{video_id, title, published_at}, ...]."""
    root = ElementTree.fromstring(xml_text)
    videos = []
    for entry in root.findall("atom:entry", _NS):
        video_id = entry.findtext("yt:videoId", namespaces=_NS)
        title = entry.findtext("atom:title", namespaces=_NS)
        published = entry.findtext("atom:published", namespaces=_NS)
        if not video_id or not published:
            continue
        videos.append({
            "video_id": video_id,
            "title": title or "",
            "published_at": datetime.fromisoformat(published),
        })
    return videos


def discover_new_videos(
    channel_id: str,
    known_video_ids: set,
    min_duration_s: Optional[int] = None,
    max_duration_s: Optional[int] = None,
    get_duration=None,
) -> List[Dict]:
    """Return RSS entries not already in known_video_ids, with duration filtering.

    get_duration(video_id) -> int seconds, injected so this stays unit-testable
    without hitting yt-dlp/network. Entries are dropped if get_duration is not
    given and a duration bound is set, duration can't be resolved, or it falls
    outside [min_duration_s, max_duration_s].
    """
    has_bounds = min_duration_s is not None or max_duration_s is not None
    if has_bounds and get_duration is None:
        logger.warning("Duration bounds set but no get_duration given; skipping all new videos")
        return []

    entries = parse_rss(fetch_rss(channel_id))
    new_entries = [e for e in entries if e["video_id"] not in known_video_ids]

    if not has_bounds:
        return new_entries

    filtered = []
    for entry in new_entries:
        try:
            duration = get_duration(entry["video_id"])
        except Exception:
            # e.g. an upcoming/live stream yt-dlp can't resolve a duration
            # for yet. Skip it this run; it'll resolve once it has one.
            logger.warning("get_duration failed for %s, skipping", entry["video_id"], exc_info=True)
            continue
        if duration is None:
            continue
        if min_duration_s is not None and duration < min_duration_s:
            continue
        if max_duration_s is not None and duration > max_duration_s:
            continue
        entry["duration_s"] = duration
        filtered.append(entry)
    return filtered
