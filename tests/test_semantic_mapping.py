"""Tests for semantic tile mapping completeness."""

import pytest

from smb2_gym.constants.object_ids import (
    BackgroundTile,
    EnemyId,
)
from smb2_gym.constants.semantic import (
    FINE_TO_COARSE_MAPPING,
    OBJECT_ID_MAPPING,
    TILE_COLORS,
    TILE_ID_MAPPING,
    UNMAPPED_OBJECT_FINE_TYPE,
    CoarseTileType,
    FineTileType,
)


def test_all_background_tiles_are_mapped():
    """Verify that every BackgroundTile enum value has a mapping in TILE_ID_MAPPING.

    This ensures that the semantic map will never encounter an unmapped tile ID,
    preventing KeyError exceptions during gameplay.
    """
    all_tile_values = set(BackgroundTile.__members__.values())
    mapped_tile_ids = set(TILE_ID_MAPPING.keys())

    # Find any missing mappings
    missing_tiles = all_tile_values - mapped_tile_ids

    # Build error message with missing tile details
    if missing_tiles:
        missing_details = []
        for tile_value in sorted(missing_tiles):
            # Find the enum name for this value
            tile_name = next(
                name for name, val in BackgroundTile.__members__.items() if val == tile_value
            )
            missing_details.append(
                f"  0x{tile_value:02X} ({tile_value:3d}): BackgroundTile.{tile_name}"
            )

        error_msg = (
            f"\n{len(missing_tiles)} BackgroundTile(s) are not mapped in TILE_ID_MAPPING:\n"
            + "\n".join(missing_details)
            + "\n\nAll tiles must have a mapping to prevent runtime KeyError exceptions."
        )
        pytest.fail(error_msg)

    # Verify we have exactly 256 tiles (full uint8 range)
    assert len(all_tile_values) == 256, (
        f"Expected 256 BackgroundTile values, got {len(all_tile_values)}"
    )
    assert len(mapped_tile_ids) == 256, f"Expected 256 mapped tiles, got {len(mapped_tile_ids)}"


def test_no_duplicate_tile_mappings():
    """Verify that TILE_ID_MAPPING has no duplicate keys.

    This is a sanity check to ensure the mapping dictionary is well-formed.
    """
    # Count occurrences of each tile ID
    tile_ids = list(TILE_ID_MAPPING.keys())
    unique_tile_ids = set(tile_ids)

    assert len(tile_ids) == len(unique_tile_ids), (
        f"TILE_ID_MAPPING has duplicate keys: {len(tile_ids)} entries but only {len(unique_tile_ids)} unique keys"
    )


def test_all_tile_ids_are_valid_background_tiles():
    """Verify that all keys in TILE_ID_MAPPING are valid BackgroundTile enum values.

    This ensures we haven't accidentally added invalid tile IDs to the mapping.
    """
    valid_tile_values = set(BackgroundTile.__members__.values())
    mapped_tile_ids = set(TILE_ID_MAPPING.keys())

    # Find any invalid mappings
    invalid_tiles = mapped_tile_ids - valid_tile_values

    if invalid_tiles:
        error_msg = (
            f"\n{len(invalid_tiles)} invalid tile ID(s) in TILE_ID_MAPPING:\n"
            + "\n".join(f"  0x{tile:02X} ({tile:3d})" for tile in sorted(invalid_tiles))
            + "\n\nAll tile IDs must be valid BackgroundTile enum values."
        )
        pytest.fail(error_msg)


# ------------------------------------------------------------------------------
# ---- Sprite (object) mapping -------------------------------------------------
# ------------------------------------------------------------------------------

# The sprite slots hold every dynamic object, not just enemies. Ids that are not
# in OBJECT_ID_MAPPING deliberately fall back to ENEMY, so the only ones allowed
# to be absent are the undocumented ones.
UNKNOWN_OBJECT_IDS = {
    value for name, value in EnemyId.__members__.items() if name.startswith("UNKNOWN")
}


def test_all_known_object_ids_are_mapped():
    """Every documented EnemyId has an explicit FineTileType.

    Only the UNKNOWN_* ids may be missing; those fall back to ENEMY at lookup
    time. Anything else missing means a real object would be misreported.
    """
    documented = set(EnemyId.__members__.values()) - UNKNOWN_OBJECT_IDS
    missing = documented - set(OBJECT_ID_MAPPING)

    if missing:
        names = {int(v): k for k, v in EnemyId.__members__.items()}
        detail = "\n".join(f"  0x{i:02X} ({i:3d}): EnemyId.{names[i]}" for i in sorted(missing))
        pytest.fail(f"\n{len(missing)} documented object id(s) unmapped:\n{detail}")


def test_object_mapping_keys_are_valid_enemy_ids():
    """No stray ids crept into the object mapping."""
    invalid = set(OBJECT_ID_MAPPING) - set(EnemyId.__members__.values())

    assert not invalid, f"invalid object ids: {sorted(hex(i) for i in invalid)}"


def test_unknown_object_ids_fall_back_to_enemy():
    """Unrecognised sprites are treated as hostile, the safer default."""
    assert UNMAPPED_OBJECT_FINE_TYPE == FineTileType.ENEMY

    for object_id in UNKNOWN_OBJECT_IDS:
        assert OBJECT_ID_MAPPING.get(object_id, UNMAPPED_OBJECT_FINE_TYPE) == FineTileType.ENEMY


def test_boss_range_mirrors_the_regular_range():
    """BOSS_* variants (0x5C-0x7F) classify the same as their 0x1C-0x3F twins."""
    for object_id in range(0x1C, 0x40):
        if object_id not in OBJECT_ID_MAPPING:
            continue
        boss_id = object_id + 0x40
        assert OBJECT_ID_MAPPING.get(boss_id) == OBJECT_ID_MAPPING[object_id], (
            f"0x{boss_id:02X} should mirror 0x{object_id:02X}"
        )


def test_non_hostile_objects_are_not_classified_as_enemies():
    """The point of the mapping: pickups and entrances must not read as threats.

    Marking the subspace door as ENEMY tells an agent to avoid the thing it is
    supposed to walk into.
    """
    expected = {
        EnemyId.SUBSPACE_DOOR: FineTileType.DOOR,
        EnemyId.SUBSPACE_POTION: FineTileType.POTION,
        EnemyId.HAWKMOUTH_RIGHT: FineTileType.DOOR,
        EnemyId.HAWKMOUTH_LEFT: FineTileType.DOOR,
        EnemyId.HEART: FineTileType.HEART,
        EnemyId.COIN: FineTileType.COIN,
        EnemyId.MUSHROOM_1UP: FineTileType.MUSHROOM,
        EnemyId.STARMAN: FineTileType.POWERUP,
        EnemyId.STOPWATCH: FineTileType.POWERUP,
        EnemyId.KEY: FineTileType.KEY,
        EnemyId.POW_BLOCK: FineTileType.POW_BLOCK,
        EnemyId.VEGETABLE_LARGE: FineTileType.VEGETABLE,
        EnemyId.SHELL: FineTileType.SHELL,
        EnemyId.FLYING_CARPET: FineTileType.PLATFORM,
    }
    for object_id, fine_type in expected.items():
        assert OBJECT_ID_MAPPING[object_id] == fine_type, (
            f"{EnemyId(object_id).name} should be {fine_type.name}, "
            f"got {FineTileType(OBJECT_ID_MAPPING[object_id]).name}"
        )


def test_hostile_objects_stay_enemies():
    """ENEMY keeps its meaning: genuinely hostile things."""
    for object_id in (
        EnemyId.SHYGUY_RED,
        EnemyId.BIRDO,
        EnemyId.MOUSER,
        EnemyId.PIDGIT,
        EnemyId.PHANTO,
        EnemyId.WART,
    ):
        assert OBJECT_ID_MAPPING[object_id] == FineTileType.ENEMY


def test_a_sprite_and_its_background_tile_agree():
    """The same object classifies identically whichever RAM path it came from.

    A POW block read from the tile map and a POW block read from a sprite slot
    must not disagree.
    """
    assert OBJECT_ID_MAPPING[EnemyId.POW_BLOCK] == TILE_ID_MAPPING[BackgroundTile.POW_BLOCK]
    assert (
        OBJECT_ID_MAPPING[EnemyId.SUBSPACE_POTION] == TILE_ID_MAPPING[BackgroundTile.GRASS_POTION]
    )
    assert (
        OBJECT_ID_MAPPING[EnemyId.VEGETABLE_LARGE]
        == (TILE_ID_MAPPING[BackgroundTile.GRASS_LARGE_VEGGIE])
    )


def test_every_fine_type_has_a_coarse_type_and_colour():
    """New fine types are fully wired up, not half-added."""
    for fine_type in FineTileType:
        assert fine_type in FINE_TO_COARSE_MAPPING, f"{fine_type.name} has no coarse type"
        assert isinstance(FINE_TO_COARSE_MAPPING[fine_type], CoarseTileType)
        if fine_type != FineTileType.EMPTY:
            assert fine_type in TILE_COLORS, f"{fine_type.name} has no colour"
