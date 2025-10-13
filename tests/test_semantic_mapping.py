"""Tests for semantic tile mapping completeness."""

import pytest

from smb2_gym.constants.object_ids import BackgroundTile
from smb2_gym.constants.semantic import TILE_ID_MAPPING


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
            tile_name = [
                name for name, val in BackgroundTile.__members__.items() if val == tile_value
            ][0]
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
    assert len(
        all_tile_values
    ) == 256, f"Expected 256 BackgroundTile values, got {len(all_tile_values)}"
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
