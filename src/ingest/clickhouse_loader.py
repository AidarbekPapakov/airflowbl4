import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import clickhouse_connect
from pydantic import ValidationError

from src.ingest.csfloat_schemas import SaleRow

TABLE = "csfloat_sales"
QUARANTINE = Path(os.environ.get("QUARANTINE_PATH", "quarantine"))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def flatten_data(row: SaleRow, snapshot_date: str) -> dict[str, Any]:
    item = row.item
    ref = row.reference
    return {
        "id": row.id,
        "sold_at": row.sold_at,
        "price": row.price,
        "market_hash_name": item.market_hash_name,
        "def_index": item.def_index,
        "paint_index": item.paint_index,
        "paint_seed": item.paint_seed,
        "float_value": item.float_value,
        "is_stattrak": item.is_stattrak,
        "is_souvenir": item.is_souvenir,
        "rarity": item.rarity,
        "wear_name": item.wear_name,
        "stickers": [
            (s.stickerId, s.slot, s.name,
             s.reference.price if s.reference else 0,
             s.reference.quantity if s.reference else 0)
            for s in item.stickers
        ],
        "keychain_id": item.keychains[0].stickerId if item.keychains else None,
        "keychain_pattern": item.keychains[0].pattern if item.keychains else None,
        "keychain_name": item.keychains[0].name if item.keychains else None,
        "keychain_ref_price": (
            item.keychains[0].reference.price
            if item.keychains and item.keychains[0].reference else None
        ),
        "fade_pct": item.fade.percentage if item.fade else None,
        "fade_rank": item.fade.rank if item.fade else None,
        "fade_type": item.fade.type if item.fade else None,
        "blue_gem": json.dumps(item.blue_gem, default=str) if item.blue_gem else None,
        "ref_base_price": ref.base_price,
        "ref_float_factor": ref.float_factor,
        "ref_predicted_price": ref.predicted_price,
        "ref_quantity": ref.quantity,
        "ref_keychain_price": ref.keychain_price,
        "_ingested_at": now_utc(),
        "_snapshot_date": datetime.strptime(snapshot_date, "%Y-%m-%d %H:%M:%S").date(),
    }


def quarantine(raw: dict[str, Any], error: Exception) -> None:
    QUARANTINE.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = QUARANTINE / f"sale_{ts}.json"
    payload = {"error": str(error), "raw": raw}
    path.write_text(json.dumps(payload, indent=2, default=str))


def validate_and_flatten(
    raw_rows: list[dict[str, Any]], snapshot_date: str
) -> list[dict[str, Any]]:
    """Validate each row; quarantine + raise on the first schema violation."""
    flat = []
    for raw in raw_rows:
        try:
            sale = SaleRow.model_validate(raw)
        except ValidationError as exc:
            quarantine(raw, exc)
            raise
        if sale.reference is None:
            continue  # no base_price/predicted_price — unusable for training
        flat.append(flatten_data(sale, snapshot_date))
    return flat


def insert_sales(
    rows: list[dict[str, Any]],
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
) -> None:
    if not rows:
        return
    client = clickhouse_connect.get_client(host=host, port=port, database=database, username=user, password=password)
    try:
        column_names = list(rows[0].keys())
        data = [[r[col] for col in column_names] for r in rows]
        client.insert(TABLE, data, column_names=column_names)
    finally:
        client.close()
