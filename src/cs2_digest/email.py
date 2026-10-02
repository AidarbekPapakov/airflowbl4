"""Renders the digest HTML. Sending is Airflow's job (airflow.utils.email.send_email
in the DAG), kept out of this module so it has no Airflow import and stays unit-testable."""

from collections import defaultdict
from html import escape
from typing import Dict, List


def render_digest_html(videos: List[Dict]) -> str:
    """videos: [{video_id, channel, title, tldr, positions, entities}, ...]"""
    by_channel = defaultdict(list)
    for v in videos:
        by_channel[v["channel"]].append(v)

    sections = []
    for channel in sorted(by_channel):
        video_blocks = []
        for v in by_channel[channel]:
            positions_html = "".join(
                f"<li><b>{escape(speaker)}</b>: {escape(text)}</li>"
                for speaker, text in v["positions"].items()
            )
            entities_html = "".join(
                f"<li><b>{escape(kind)}</b>: {escape(', '.join(items)) if items else '-'}</li>"
                for kind, items in v["entities"].items()
            )
            video_blocks.append(f"""
            <h3>{escape(v['title'])}</h3>
            <p>{escape(v['tldr'])}</p>
            <p><i>Who argued what</i></p>
            <ul>{positions_html}</ul>
            <p><i>Entities mentioned</i></p>
            <ul>{entities_html}</ul>
            """)
        sections.append(f"<h2>{escape(channel)}</h2>" + "".join(video_blocks))

    return "<h1>cs2_digest</h1>" + "".join(sections)
