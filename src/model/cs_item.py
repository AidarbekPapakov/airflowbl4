"""
Item identity for CS2 inventory assets.

Steam hands out one `asset_id` per physical copy, so an inventory holding 59
identical sticker capsules yields 59 rows that say the same thing. `CsItem`
collapses that: two assets are the same item when they are interchangeable on
the market.

`market_hash_name` already encodes weapon, skin, wear tier, StatTrak/Souvenir
and the star prefix, so for most categories identity is just the name. The one
thing the name cannot express is applied stickers/charms — a bare
"AK-47 | Redline (Field-Tested)" and one wearing Katowice 2014 holos share a
name but not a price. Only weapons can carry them, so only WEAPON_SKIN folds
them into its identity.

Float value is deliberately *not* part of identity: two Factory New copies of a
skin have different floats but trade as the same item, and treating floats as
distinguishing would put us back to one row per asset.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Optional, Tuple


class ItemCategory(Enum):
    WEAPON_SKIN = "weapon_skin"
    KNIFE = "knife"
    GLOVES = "gloves"
    CASE = "case"
    KEY = "key"
    STICKER_CAPSULE = "sticker_capsule"
    SOUVENIR_PACKAGE = "souvenir_package"
    STICKER = "sticker"
    CHARM = "charm"
    PATCH = "patch"
    GRAFFITI = "graffiti"
    AGENT = "agent"
    MUSIC_KIT = "music_kit"
    PIN = "pin"
    PASS = "pass"
    OTHER = "other"


# Only weapons can carry stickers/charms. Knives and gloves cannot take
# stickers, and no container category can.
_STICKER_BEARING: FrozenSet[ItemCategory] = frozenset({ItemCategory.WEAPON_SKIN})

_EMPTY: Tuple[str, ...] = ()

# The only five parentheticals that mean wear. Anything else in trailing
# parentheses — "Sticker | Reason Gaming (Holo)" — is part of the name.
_WEAR_NAMES: FrozenSet[str] = frozenset({
    "Factory New",
    "Minimal Wear",
    "Field-Tested",
    "Well-Worn",
    "Battle-Scarred",
})

_WEAR_SUFFIX_RE = re.compile(r"\s*\((?P<wear>[^()]+)\)\s*$")
_STATTRAK_RE = re.compile(r"^StatTrak™\s+")
_SOUVENIR_RE = re.compile(r"^Souvenir\s+")
_STAR_RE = re.compile(r"^★\s*")

_PREFIX_CATEGORIES: Tuple[Tuple[str, ItemCategory], ...] = (
    ("Sticker | ", ItemCategory.STICKER),
    ("Charm | ", ItemCategory.CHARM),
    ("Patch | ", ItemCategory.PATCH),
    ("Sealed Graffiti | ", ItemCategory.GRAFFITI),
    ("Graffiti | ", ItemCategory.GRAFFITI),
    ("Music Kit | ", ItemCategory.MUSIC_KIT),
)


@dataclass(frozen=True, eq=False)
class CsItem:
    """
    The market identity of an inventory asset.

    eq=False keeps the dataclass from generating an __eq__ that would compare
    raw fields; identity is defined by the `identity` tuple instead, and
    __hash__ is derived from the same tuple so the two cannot drift apart.
    Hashability is the point — aggregation is a groupby, not a pairwise scan.
    """

    market_hash_name: str
    applied_stickers: Tuple[str, ...] = _EMPTY
    applied_charms: Tuple[str, ...] = _EMPTY

    # ------------------------------------------------------------------
    # Parsed name components
    # ------------------------------------------------------------------

    @property
    def is_stattrak(self) -> bool:
        return bool(_STATTRAK_RE.match(_STAR_RE.sub("", self.market_hash_name)))

    @property
    def is_souvenir(self) -> bool:
        return bool(_SOUVENIR_RE.match(self.market_hash_name))

    @property
    def wear_name(self) -> Optional[str]:
        """The wear tier from the name suffix, or None for items without wear."""
        match = _WEAR_SUFFIX_RE.search(self.market_hash_name)
        if match and match.group("wear") in _WEAR_NAMES:
            return match.group("wear")
        return None

    @property
    def base_name(self) -> str:
        """The name with the wear suffix and quality prefixes stripped."""
        name = self.market_hash_name
        if self.wear_name is not None:
            name = _WEAR_SUFFIX_RE.sub("", name)
        name = _STAR_RE.sub("", name)
        name = _STATTRAK_RE.sub("", name)
        name = _SOUVENIR_RE.sub("", name)
        return name.strip()

    @property
    def category(self) -> ItemCategory:
        name = self.market_hash_name

        for prefix, category in _PREFIX_CATEGORIES:
            if name.startswith(prefix):
                return category

        if _STAR_RE.match(name):
            return ItemCategory.GLOVES if _is_gloves(name) else ItemCategory.KNIFE

        # A wear suffix on a "Weapon | Skin" name is what makes it a weapon skin.
        if self.wear_name is not None and " | " in name:
            return ItemCategory.GLOVES if _is_gloves(name) else ItemCategory.WEAPON_SKIN

        if name.endswith("Key"):
            return ItemCategory.KEY
        if name.endswith("Case"):
            return ItemCategory.CASE
        if "Capsule" in name:
            return ItemCategory.STICKER_CAPSULE
        if "Souvenir Package" in name:
            return ItemCategory.SOUVENIR_PACKAGE
        if name.endswith("Pin"):
            return ItemCategory.PIN
        if name.endswith("Pass"):
            return ItemCategory.PASS
        # Agents are the remaining "Name | Group" items, e.g.
        # "Sir Bloody Miami Darryl | The Professionals".
        if " | " in name:
            return ItemCategory.AGENT
        return ItemCategory.OTHER

    @property
    def display_name(self) -> str:
        """Name, plus a sticker/charm marker so broken-out rows are tellable apart."""
        parts = []
        if self.applied_stickers:
            parts.append(_plural(len(self.applied_stickers), "sticker"))
        if self.applied_charms:
            parts.append(_plural(len(self.applied_charms), "charm"))
        if not parts:
            return self.market_hash_name
        return f"{self.market_hash_name} · {', '.join(parts)}"

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def _weapon_identity(self) -> Tuple:
        """Weapons: the name, plus whatever is applied to this particular copy."""
        return (self.market_hash_name, self.applied_stickers, self.applied_charms)

    def _name_identity(self) -> Tuple:
        """Everything else: the name alone — nothing can be applied to it."""
        return (self.market_hash_name, _EMPTY, _EMPTY)

    @property
    def identity(self) -> Tuple:
        if self.category in _STICKER_BEARING:
            return self._weapon_identity()
        return self._name_identity()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CsItem):
            return NotImplemented
        return self.identity == other.identity

    def __hash__(self) -> int:
        return hash(self.identity)

    def __repr__(self) -> str:
        return f"CsItem({self.display_name!r}, {self.category.value})"


def _is_gloves(name: str) -> bool:
    return "Gloves" in name or "Hand Wraps" in name


def _plural(count: int, noun: str) -> str:
    return f"+{count} {noun}" if count == 1 else f"+{count} {noun}s"
