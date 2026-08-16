"""
Pydantic models pinning the CSFloat /history/{name}/sales response shape.

extra="ignore" on all models: we only validate fields we USE. This catches
removals/renames/type-changes of depended-upon fields while tolerating new
upstream fields (the safer posture for an undocumented endpoint).
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class _Base(BaseModel):
    model_config = {"extra": "ignore"}


class StickerReference(_Base):
    price: int
    quantity: int


class Sticker(_Base):
    stickerId: int
    slot: int
    name: str
    reference: Optional[StickerReference] = None


class KeychainReference(_Base):
    price: int
    quantity: int


class Keychain(_Base):
    stickerId: int   # keychain type ID
    slot: int
    pattern: Optional[int] = None  # 0–100 000; absent on some keychain types
    name: str
    reference: Optional[KeychainReference] = None


class FadeInfo(_Base):
    percentage: float
    rank: int
    type: str


class SaleItem(_Base):
    market_hash_name: str
    def_index: int
    paint_index: int
    paint_seed: int
    float_value: float
    is_stattrak: bool
    is_souvenir: bool
    rarity: int
    wear_name: str
    stickers: list[Sticker] = Field(default_factory=list)
    keychains: list[Keychain] = Field(default_factory=list)
    # Fade: only on Fade-finish skins; exact fields confirmed from API docs
    fade: Optional[FadeInfo] = None
    # Blue gem: only on Case Hardened; sub-fields stored as-is until a live
    # response confirms the exact names (then promote to a typed model)
    blue_gem: Optional[dict[str, Any]] = None


class SaleReference(_Base):
    base_price: int
    float_factor: float = 0.0  # absent on some skins (e.g. Souvenir items)
    predicted_price: int
    quantity: int
    keychain_price: int = 0    # absent when no charm is applied


class SaleRow(_Base):
    id: str
    sold_at: str
    price: int
    item: SaleItem
    reference: Optional[SaleReference] = None  # empty {} on ultra-rare skins

    @model_validator(mode="before")
    @classmethod
    def _empty_reference_to_none(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("reference") == {}:
            data["reference"] = None
        return data
