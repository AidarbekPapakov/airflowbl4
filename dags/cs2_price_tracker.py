"""
CS2 inventory price-tracking pipeline.

Task flow:
  fetch_inventory → price_item.expand() → store_observations → compute_changes_and_notify

Base price = median of CSFloat completed sales (/api/v1/history/{name}/sales) within the
last 24h, which avoids the lowball/outlier risk of a single lowest-listing price. Items with
no completed sales in that window are skipped for the day.
Sticker/charm prices still use CSFloat's lowest listing (/api/v1/listings).
Modified price = base price + sticker/charm value (SP% heuristic — see _sp_pct()).
Both base and modified prices are tracked independently; either can trigger an alert.
"""

import json
import re
import statistics
import time
from datetime import date, datetime, timedelta, timezone
from html import escape
from typing import Any, Dict, List, Optional, Tuple

import clickhouse_connect
import requests
from airflow.sdk import Variable, dag, task
from airflow.utils.email import send_email
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.ingest.csfloat_client import fetch_sales_history
from src.model.cs_item import CsItem

CS2_APP_ID = "730"

class InventoryFetchError(Exception):
    pass


class PriceFetchError(Exception):
    pass


# ---------------------------------------------------------------------------
# Sticker / charm valuation helpers
# ---------------------------------------------------------------------------

_DEFAULT_SP_TABLE = {
    "kato14_holo": 0.08,    # Katowice 2014 Holo / iBUYPOWER Holo
    "kato14_paper": 0.05,   # Katowice 2014 paper
    "old_major_holo": 0.10, # Other major Holo/Foil 2015–2018
    "modern_holo": 0.07,    # Modern Holo/Foil 2019+
    "modern_paper": 0.005,  # Regular paper modern
    "fallback": 0.02,
}


def _sp_pct(sticker_name: str, sp_table: Dict) -> float:
    """Return the SP% for a sticker based on its name."""
    name_lower = sticker_name.lower()

    is_holo = "(holo)" in name_lower or "(foil)" in name_lower or "(gold)" in name_lower
    is_kato14 = "katowice 2014" in name_lower or "ibuypower" in name_lower

    if is_kato14 and is_holo:
        return sp_table.get("kato14_holo", _DEFAULT_SP_TABLE["kato14_holo"])
    if is_kato14:
        return sp_table.get("kato14_paper", _DEFAULT_SP_TABLE["kato14_paper"])

    # Detect year from sticker name (e.g. "2016", "2023")
    year_match = re.search(r"20(\d{2})", sticker_name)
    year = int("20" + year_match.group(1)) if year_match else None

    if year and year <= 2018 and is_holo:
        return sp_table.get("old_major_holo", _DEFAULT_SP_TABLE["old_major_holo"])
    if year and year >= 2019 and is_holo:
        return sp_table.get("modern_holo", _DEFAULT_SP_TABLE["modern_holo"])
    if year and year >= 2019:
        return sp_table.get("modern_paper", _DEFAULT_SP_TABLE["modern_paper"])

    return sp_table.get("fallback", _DEFAULT_SP_TABLE["fallback"])


def _csfloat_sticker_name(steam_title: str) -> str:
    """
    Strip the "Sticker: "/"Charm: " prefix Steam already puts on parsed titles,
    so callers can rebuild CSFloat's own "Sticker | <name>"/"Charm | <name>" convention
    without doubling it up.
    """
    return re.sub(r"^(sticker|charm):\s*", "", steam_title, flags=re.IGNORECASE)


# CSFloat fetch with retry on 429
@retry(
    retry=retry_if_exception_type(PriceFetchError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _fetch_csfloat_price(market_hash_name: str, api_key: str) -> Optional[float]:
    """
    Return the lowest listed price (USD cents → dollars) for market_hash_name on CSFloat.
    Returns None if no listings found.
    """
    resp = requests.get(
        "https://csfloat.com/api/v1/listings",
        params={"market_hash_name": market_hash_name, "sort_by": "lowest_price", "limit": 1},
        headers={"Authorization": api_key},
        timeout=15,
    )
    if resp.status_code == 429:
        raise PriceFetchError(f"CSFloat rate-limited for {market_hash_name}")
    if resp.status_code != 200:
        raise PriceFetchError(f"CSFloat returned {resp.status_code} for {market_hash_name}")

    data = resp.json()
    listings = data.get("data", [])
    if not listings:
        return None
    # CSFloat returns price in USD cents
    return listings[0]["price"] / 100.0


def _fetch_median_sale_price(market_hash_name: str, api_key: str, window_hours: int = 24) -> Optional[float]:
    """
    Return the median completed-sale price (USD) for market_hash_name on CSFloat,
    over the last `window_hours`. Returns None if there were no sales in that window.
    """
    sales = fetch_sales_history(market_hash_name, paint_index=None, api_key=api_key)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    recent_prices = [
        sale["price"] / 100.0
        for sale in sales
        if datetime.fromisoformat(sale["created_at"].replace("Z", "+00:00")) >= cutoff
    ]
    if not recent_prices:
        return None
    return statistics.median(recent_prices)


# Steam inventory parsing helpers
def _parse_stickers_from_description(descriptions: List[Dict]) -> List[str]:
    """
    Extract applied sticker names from the HTML description blocks Steam returns.
    The relevant block is a 'sticker_info' div inside description value HTML.
    """
    stickers = []
    for block in descriptions:
        if block.get("type") != "html":
            continue
        soup = BeautifulSoup(block["value"], "html.parser")
        sticker_div = soup.find("div", {"id": "sticker_info"})
        if not sticker_div:
            continue
        # Each sticker image has a title attribute with the sticker name
        for img in sticker_div.find_all("img"):
            title = img.get("title", "").strip()
            if title:
                stickers.append(title)
    return stickers


def _parse_charms_from_description(descriptions: List[Dict]) -> List[str]:
    """
    Extract applied charm (keychain) names from description HTML.
    Charms appear in a 'keychain_info' div or similar pattern.
    """
    charms = []
    for block in descriptions:
        if block.get("type") != "html":
            continue
        soup = BeautifulSoup(block["value"], "html.parser")
        charm_div = soup.find("div", {"id": "keychain_info"})
        if not charm_div:
            continue
        for img in charm_div.find_all("img"):
            title = img.get("title", "").strip()
            if title:
                charms.append(title)
    return charms


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

_REPORT_CSS = """
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; color: #24292f; }
h3 { margin-bottom: 4px; }
p.summary { margin-top: 0; color: #57606a; font-size: 13px; }
table { border-collapse: collapse; font-size: 13px; }
th, td { border: 1px solid #d0d7de; padding: 4px 8px; }
thead th { background: #4a90d9; color: #fff; font-weight: 600; text-align: center; }
tbody tr:nth-child(even) { background: #f6f8fa; }
td.item { text-align: left; white-space: nowrap; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.none { text-align: center; color: #8c959f; }
.up { color: #1a7f37; }
.down { color: #cf222e; }
"""


def _pct_cell(pct: Optional[float]) -> str:
    """A percentage cell, coloured by direction; an em dash when there was no alert."""
    if pct is None:
        return '<td class="none">&mdash;</td>'
    css = "up" if pct >= 0 else "down"
    return f'<td class="num {css}">{pct:+.1f}%</td>'


def _render_report_html(rows: List[Dict], windows: List[int], threshold_pct: float, today: date) -> str:
    """
    Render the alert table: one row per item, comparison windows as column pairs.

    Built by hand rather than via DataFrame.to_html because the two-level header
    and the per-cell colouring need more control than to_html gives.
    """
    header_top = "".join(f'<th colspan="2">{w}d</th>' for w in windows)
    header_bottom = "".join('<th>base</th><th>mod</th>' for _ in windows)

    body = []
    for row in rows:
        cells = [
            # Steam names carry '&' ("Dreams & Nightmares Case") and can carry '<'
            f'<td class="item">{escape(row["item"])}</td>',
            f'<td class="num">{row["qty"]}</td>',
            f'<td class="num">{row["base_price"]:.2f}</td>',
            f'<td class="num">{row["modified_price"]:.2f}</td>',
        ]
        for w in windows:
            cells.append(_pct_cell(row["changes"].get((w, "base_price"))))
            cells.append(_pct_cell(row["changes"].get((w, "modified_price"))))
        impact_css = "up" if row["impact"] >= 0 else "down"
        cells.append(f'<td class="num {impact_css}">{row["impact"]:+.2f}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")

    net_impact = sum(row["impact"] for row in rows)
    return (
        f"<style>{_REPORT_CSS}</style>"
        f"<h3>Price moves exceeding {threshold_pct:.0f}% as of {today}</h3>"
        f'<p class="summary">{len(rows)} item(s) moved &middot; '
        f"net impact {net_impact:+.2f} USD across the shortest alerting window</p>"
        "<table><thead>"
        f'<tr><th rowspan="2">Item</th><th rowspan="2">Qty</th>'
        f'<th rowspan="2">Base $</th><th rowspan="2">Mod $</th>'
        f'{header_top}<th rowspan="2">Impact $</th></tr>'
        f"<tr>{header_bottom}</tr>"
        "</thead><tbody>"
        f"{''.join(body)}"
        "</tbody></table>"
    )


@dag(
    dag_id="report_price_changes",
    schedule="@daily",
    start_date=datetime(2025, 9, 20),
    catchup=False,
    tags=["cs2", "csfloat", "clickhouse"],
    max_active_runs=1,
    max_active_tasks=4,
)
def cs2_price_tracker():

    @task(retries=3, retry_delay=timedelta(seconds=90))
    def fetch_inventory() -> List[Dict]:
        steam_id = Variable.get("steam_inventory_secret")
        url = f"https://steamcommunity.com/inventory/{steam_id}/{CS2_APP_ID}/2?l=english"
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            raise InventoryFetchError(f"Steam inventory returned {resp.status_code}")

        payload = resp.json()
        descriptions_by_key: Dict[str, Dict] = {
            f"{d['classid']}_{d['instanceid']}": d
            for d in payload.get("descriptions", [])
        }

        assets = []
        for asset in payload.get("assets", []):
            key = f"{asset['classid']}_{asset['instanceid']}"
            desc = descriptions_by_key.get(key)
            if not desc or not desc.get("marketable"):
                continue

            stickers = _parse_stickers_from_description(desc.get("descriptions", []))
            charms = _parse_charms_from_description(desc.get("descriptions", []))

            assets.append({
                "asset_id": asset["assetid"],
                "market_hash_name": desc["market_hash_name"],
                "applied_stickers": stickers,
                "applied_charms": charms,
            })

        return assets

    @task(retries=3, retry_delay=timedelta(seconds=90))
    def price_item(record: Dict) -> Optional[Dict]:
        """
        Fetch base + modified price for one inventory asset.
        Returns None for items in the $0.03–$0.05 exclusion band.
        """
        api_key = Variable.get("csfloat_api_key")
        sp_table = json.loads(Variable.get("sticker_sp_table", json.dumps(_DEFAULT_SP_TABLE)))
        charm_retention = float(Variable.get("charm_retention", "0.9"))

        base_price = _fetch_median_sale_price(record["market_hash_name"], api_key)
        if base_price is None:
            return None

        # Exclusion: low-value items ($0.03–$0.05)
        if 0.03 <= base_price <= 0.05:
            return None

        # Price each applied sticker (cache per task run to avoid duplicate lookups)
        sticker_value = 0.0
        seen_prices: Dict[str, Optional[float]] = {}
        for sticker_name in record["applied_stickers"]:
            market_name = f"Sticker | {_csfloat_sticker_name(sticker_name)}"
            if market_name not in seen_prices:
                seen_prices[market_name] = _fetch_csfloat_price(market_name, api_key)
                time.sleep(0.5)  # light throttle within a single mapped task
            price = seen_prices[market_name]
            if price is not None:
                sticker_value += price * _sp_pct(sticker_name, sp_table)

        # Price each applied charm
        charm_value = 0.0
        for charm_name in record["applied_charms"]:
            market_name = f"Charm | {_csfloat_sticker_name(charm_name)}"
            if market_name not in seen_prices:
                seen_prices[market_name] = _fetch_csfloat_price(market_name, api_key)
                time.sleep(0.5)
            price = seen_prices[market_name]
            if price is not None:
                charm_value += price * charm_retention

        modified_price = base_price + sticker_value + charm_value

        return {
            "asset_id": record["asset_id"],
            "market_hash_name": record["market_hash_name"],
            "base_price": round(base_price, 2),
            "modified_price": round(modified_price, 2),
            "sticker_value_added": round(sticker_value + charm_value, 2),
            "applied_stickers": record["applied_stickers"],
            "applied_charms": record["applied_charms"],
        }

    @task(trigger_rule="all_done")
    def store_observations(priced_records: List[Any], **context) -> None:
        observed_date = context["data_interval_end"].date()
        records = [r for r in priced_records if r is not None]
        if not records:
            return

        client = clickhouse_connect.get_client(
            host=Variable.get("clickhouse_host", "clickhouse"),
            port=int(Variable.get("clickhouse_port", "8123")),
            username=Variable.get("clickhouse_user", "default"),
            password=Variable.get("clickhouse_password", ""),
            database="cs2",
        )

        # Insert new items into the dimension table (skip already-known ones)
        all_names = {r["market_hash_name"] for r in records}
        existing = {
            row[0]
            for row in client.query(
                "SELECT market_hash_name FROM cs2.skins FINAL"
            ).result_rows
        }
        new_skins = [
            [name, CsItem(name).category.value, observed_date]
            for name in all_names - existing
        ]
        if new_skins:
            client.insert(
                "cs2.skins",
                new_skins,
                column_names=["market_hash_name", "item_type", "first_seen"],
            )

        # Insert price observations (ReplacingMergeTree deduplicates on re-runs)
        rows = [
            [
                observed_date,
                r["asset_id"],
                r["market_hash_name"],
                r["base_price"],
                r["modified_price"],
                r["sticker_value_added"],
                r["applied_stickers"],
                r["applied_charms"],
            ]
            for r in records
        ]
        client.insert(
            "cs2.price_observations",
            rows,
            column_names=[
                "observed_date", "asset_id", "market_hash_name",
                "base_price", "modified_price", "sticker_value_added",
                "applied_stickers", "applied_charms",
            ],
        )

    @task
    def compute_changes_and_notify(**context) -> None:
        today = context["data_interval_end"].date()
        threshold_pct = float(Variable.get("price_change_threshold_pct", "15"))
        windows: List[int] = json.loads(Variable.get("comparison_windows_days", "[1, 3, 7, 14, 30]"))
        recipient = Variable.get("recipients_email")

        client = clickhouse_connect.get_client(
            host=Variable.get("clickhouse_host", "clickhouse"),
            port=int(Variable.get("clickhouse_port", "8123")),
            username=Variable.get("clickhouse_user", "default"),
            password=Variable.get("clickhouse_password", ""),
            database="cs2",
        )

        # Pull enough history to cover the widest comparison window
        cutoff = today - timedelta(days=max(windows) + 1)
        result = client.query(
            "SELECT observed_date, asset_id, market_hash_name, base_price, modified_price, "
            "applied_stickers, applied_charms "
            "FROM cs2.price_observations FINAL "
            "WHERE observed_date >= %(cutoff)s "
            "ORDER BY observed_date",
            parameters={"cutoff": cutoff},
        )

        import pandas as pd

        df = pd.DataFrame(result.result_rows, columns=[
            "observed_date", "asset_id", "market_hash_name",
            "base_price", "modified_price", "applied_stickers", "applied_charms",
        ])
        if df.empty:
            return

        df["observed_date"] = pd.to_datetime(df["observed_date"])
        df["base_price"] = df["base_price"].astype(float)
        df["modified_price"] = df["modified_price"].astype(float)

        # Collapse interchangeable copies: Steam gives every capsule its own
        # asset_id, but 59 identical capsules are one line item on the report.
        # ClickHouse returns Array(String) as a list, which is unhashable —
        # CsItem takes tuples so its identity can be used as a dict key.
        # Each identity gets an integer id: grouping and filtering on a plain
        # int avoids pandas' ambiguous handling of tuple-valued columns.
        items: Dict[int, CsItem] = {}
        item_ids: Dict[Tuple, int] = {}
        row_ids = []
        for name, stickers, charms in zip(
            df["market_hash_name"], df["applied_stickers"], df["applied_charms"]
        ):
            item = CsItem(name, tuple(stickers or ()), tuple(charms or ()))
            if item.identity not in item_ids:
                item_ids[item.identity] = len(item_ids)
                items[item_ids[item.identity]] = item
            row_ids.append(item_ids[item.identity])
        df["item_id"] = row_ids

        daily = (
            df.groupby(["item_id", "observed_date"], sort=False)
            .agg(
                qty=("asset_id", "nunique"),
                base_price=("base_price", "median"),
                modified_price=("modified_price", "median"),
            )
            .reset_index()
            .sort_values("observed_date")
        )

        today_df = daily[daily["observed_date"] == pd.Timestamp(today)]
        if today_df.empty:
            return

        report_rows = []
        for _, today_row in today_df.iterrows():
            item_id = int(today_row["item_id"])
            history = daily[daily["item_id"] == item_id]

            changes: Dict[Tuple, float] = {}
            past_by_window: Dict[int, Any] = {}
            for w in windows:
                target_date = pd.Timestamp(today - timedelta(days=w))
                past = history[history["observed_date"] <= target_date]
                if past.empty:
                    continue
                # Pick the observation closest to target_date (most recent on/before it)
                past_row = past.iloc[-1]
                past_by_window[w] = past_row

                for price_col in ("base_price", "modified_price"):
                    old_price = float(past_row[price_col])
                    new_price = float(today_row[price_col])
                    if old_price == 0:
                        continue
                    pct_change = (new_price - old_price) / old_price * 100
                    if abs(pct_change) >= threshold_pct:
                        changes[(w, price_col)] = pct_change

            if not changes:
                continue

            # Value moved, measured over the shortest window that alerted.
            shortest = min(w for w, _ in changes)
            reference = past_by_window[shortest]
            impact = int(today_row["qty"]) * (
                float(today_row["modified_price"]) - float(reference["modified_price"])
            )

            report_rows.append({
                "item": items[item_id].display_name,
                "qty": int(today_row["qty"]),
                "base_price": float(today_row["base_price"]),
                "modified_price": float(today_row["modified_price"]),
                "changes": changes,
                "impact": impact,
            })

        if not report_rows:
            return

        report_rows.sort(key=lambda r: -abs(r["impact"]))
        # Drop windows that nothing alerted on, so the table stays narrow
        windows_used = sorted({w for r in report_rows for w, _ in r["changes"]})

        send_email(
            to=[recipient],
            subject=f"CS2 Price Alert — {today}",
            html_content=_render_report_html(report_rows, windows_used, threshold_pct, today),
        )

    # Task wiring
    inventory = fetch_inventory()
    priced = price_item.expand(record=inventory)
    stored = store_observations(priced)
    stored >> compute_changes_and_notify()


cs2_price_tracker()
