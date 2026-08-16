"""
Identity rules for CsItem.

The report aggregates on CsItem, so these tests pin down exactly which assets
collapse into one row and which stay apart. Run with `pytest tests/`; no
network, no Airflow.
"""

import pytest

from src.model.cs_item import CsItem, ItemCategory


@pytest.mark.parametrize("name,expected", [
    ("AWP | Atheris (Factory New)", ItemCategory.WEAPON_SKIN),
    ("StatTrak™ AK-47 | Redline (Field-Tested)", ItemCategory.WEAPON_SKIN),
    ("Souvenir AWP | Dragon Lore (Factory New)", ItemCategory.WEAPON_SKIN),
    ("★ Karambit | Doppler (Factory New)", ItemCategory.KNIFE),
    ("★ Karambit", ItemCategory.KNIFE),
    ("★ Sport Gloves | Pandora's Box (Field-Tested)", ItemCategory.GLOVES),
    ("★ Hand Wraps | Cobalt Skulls (Minimal Wear)", ItemCategory.GLOVES),
    ("Clutch Case", ItemCategory.CASE),
    ("Clutch Case Key", ItemCategory.KEY),
    ("Paris 2023 Challengers Sticker Capsule", ItemCategory.STICKER_CAPSULE),
    ("Sticker | Reason Gaming (Holo) | Katowice 2014", ItemCategory.STICKER),
    ("Charm | Die-cast AK", ItemCategory.CHARM),
    ("Patch | Fnatic (Gold)", ItemCategory.PATCH),
    ("Sealed Graffiti | Kawaii Killer (Tracer Yellow)", ItemCategory.GRAFFITI),
    ("Music Kit | Daniel Sadowski, Crimson Assault", ItemCategory.MUSIC_KIT),
    ("Sir Bloody Miami Darryl | The Professionals", ItemCategory.AGENT),
    ("Berlin 2019 Legends Pin", ItemCategory.PIN),
    ("Operation Riptide Pass", ItemCategory.PASS),
])
def test_category_detection(name, expected):
    assert CsItem(name).category is expected


def test_wear_name_parsed_from_suffix():
    assert CsItem("AWP | Atheris (Factory New)").wear_name == "Factory New"
    assert CsItem("Galil AR | Galigator (Battle-Scarred)").wear_name == "Battle-Scarred"


def test_non_wear_parenthetical_is_not_wear():
    """A sticker's "(Holo)" must not be mistaken for a wear tier."""
    sticker = CsItem("Sticker | Reason Gaming (Holo) | Katowice 2014")
    assert sticker.wear_name is None
    assert CsItem("Clutch Case").wear_name is None


def test_base_name_strips_wear_and_quality_prefixes():
    assert CsItem("StatTrak™ AK-47 | Redline (Field-Tested)").base_name == "AK-47 | Redline"
    assert CsItem("★ Karambit | Doppler (Factory New)").base_name == "Karambit | Doppler"
    assert CsItem("Souvenir AWP | Dragon Lore (Factory New)").base_name == "AWP | Dragon Lore"


def test_stattrak_and_souvenir_flags():
    assert CsItem("StatTrak™ AK-47 | Redline (Field-Tested)").is_stattrak
    assert CsItem("★ StatTrak™ Karambit | Doppler (Factory New)").is_stattrak
    assert not CsItem("AK-47 | Redline (Field-Tested)").is_stattrak
    assert CsItem("Souvenir AWP | Dragon Lore (Factory New)").is_souvenir


def test_same_wear_collapses_regardless_of_float():
    """The whole point: floats differ per copy but the items are interchangeable."""
    a = CsItem("AWP | Atheris (Factory New)")
    b = CsItem("AWP | Atheris (Factory New)")
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_different_wear_does_not_collapse():
    fn = CsItem("AWP | Atheris (Factory New)")
    ft = CsItem("AWP | Atheris (Field-Tested)")
    assert fn != ft
    assert len({fn, ft}) == 2


def test_stattrak_does_not_collapse_into_plain():
    assert CsItem("StatTrak™ AK-47 | Redline (Field-Tested)") != CsItem("AK-47 | Redline (Field-Tested)")


def test_stickers_split_weapon_skins():
    bare = CsItem("AK-47 | Redline (Field-Tested)")
    stickered = CsItem("AK-47 | Redline (Field-Tested)", ("Sticker: Titan (Holo) | Katowice 2014",))
    assert bare != stickered
    assert len({bare, stickered}) == 2


def test_charms_split_weapon_skins():
    bare = CsItem("AK-47 | Redline (Field-Tested)")
    charmed = CsItem("AK-47 | Redline (Field-Tested)", applied_charms=("Charm: Die-cast AK",))
    assert bare != charmed


def test_identical_sticker_sets_collapse():
    stickers = ("Sticker: Titan (Holo) | Katowice 2014",)
    assert CsItem("AK-47 | Redline (Field-Tested)", stickers) == CsItem("AK-47 | Redline (Field-Tested)", stickers)


def test_containers_ignore_sticker_field():
    """A case cannot bear stickers, so the field must not split the group."""
    plain = CsItem("Clutch Case")
    bogus = CsItem("Clutch Case", ("Sticker: Titan (Holo) | Katowice 2014",))
    assert plain == bogus
    assert len({plain, bogus}) == 1


def test_knives_and_gloves_ignore_sticker_field():
    """Neither can take stickers, so identity is the name alone."""
    knife = CsItem("★ Karambit | Doppler (Factory New)")
    assert knife == CsItem("★ Karambit | Doppler (Factory New)", ("Sticker: Titan",))

    gloves = CsItem("★ Sport Gloves | Pandora's Box (Field-Tested)")
    assert gloves == CsItem("★ Sport Gloves | Pandora's Box (Field-Tested)", ("Sticker: Titan",))


def test_capsules_collapse_across_copies():
    """The 59-capsule case that inflated the report to 19 pages."""
    copies = [CsItem("Paris 2023 Challengers Sticker Capsule") for _ in range(59)]
    assert len(set(copies)) == 1


def test_display_name_marks_applied_items():
    assert CsItem("Clutch Case").display_name == "Clutch Case"
    assert CsItem("AK-47 | Redline (Field-Tested)", ("a", "b")).display_name == (
        "AK-47 | Redline (Field-Tested) · +2 stickers"
    )
    assert CsItem("AK-47 | Redline (Field-Tested)", ("a",)).display_name == (
        "AK-47 | Redline (Field-Tested) · +1 sticker"
    )
    assert CsItem("AK-47 | Redline (Field-Tested)", ("a",), ("c",)).display_name == (
        "AK-47 | Redline (Field-Tested) · +1 sticker, +1 charm"
    )


def test_not_equal_to_non_csitem():
    assert CsItem("Clutch Case") != "Clutch Case"
