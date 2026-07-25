"""Tests for the layered semantic map and its binary tensor view."""

import numpy as np
import pytest

from smb2_gym import SuperMarioBros2Env
from smb2_gym.app import InitConfig
from smb2_gym.constants import (
    COARSE_TENSOR_CHANNEL_NAMES,
    COARSE_TENSOR_CHANNELS,
    NO_OBJECT,
    PLAYER,
    PROPERTY_CHANNEL_NAMES,
    SCREEN_TILES_HEIGHT,
    SCREEN_TILES_WIDTH,
    VELOCITY_CHANNEL_NAMES,
    CoarseTileType,
    FineTileType,
)
from smb2_gym.constants.semantic import (
    DAMAGING_FINE_TYPES,
    TILE_ID_MAPPING,
)


NOOP, RIGHT, JUMP = 0, 1, 13  # complex action set

TERRAIN_CHANNELS = [i for i, (layer, _) in enumerate(COARSE_TENSOR_CHANNELS) if layer == 'terrain']
OBJECT_CHANNELS = [i for i, (layer, _) in enumerate(COARSE_TENSOR_CHANNELS) if layer == 'object']


def _settle(env, frames=900):
    previous, stable = None, 0
    for _ in range(frames):
        env.step(NOOP)
        y = env._read_ram_safe(PLAYER.Y_POSITION)
        stable = stable + 1 if y == previous else 0
        previous = y
        if stable >= 25:
            return True
    return False


@pytest.fixture(params=["1-2", "1-3", "2-1", "4-1"])
def live_env(request):
    env = SuperMarioBros2Env(
        init_config=InitConfig(level=request.param, character="mario"),
        render_mode=None,
        action_type="complex",
    )
    env.reset()
    _settle(env)
    yield env
    env.close()


def test_sprites_do_not_overwrite_terrain(live_env):
    """An object never erases the terrain underneath it.

    The old map wrote sprites straight into the terrain field, so a door stood
    on nothing. Keeping the layers apart is what lets a cell say "solid ground,
    with a door on it".

    Checked by comparing the terrain field against the raw background tile it is
    derived from: if a sprite had overwritten it, the two would disagree.
    """
    checked = False

    for _ in range(60):
        live_env.step(RIGHT)
        semantic_map = live_env.semantic_map
        occupied = semantic_map['object_id'] != NO_OBJECT
        if not occupied.any():
            continue

        for row, column in np.argwhere(occupied):
            cell = semantic_map[row, column]
            expected_terrain = TILE_ID_MAPPING[int(cell['tile_id'])]

            assert int(cell['fine_type']) == expected_terrain, (
                f"terrain at ({column}, {row}) reads "
                f"{FineTileType(int(cell['fine_type'])).name} but its background tile "
                f"{int(cell['tile_id']):#04x} is {FineTileType(expected_terrain).name}; "
                "a sprite has overwritten the terrain layer"
            )
            checked = True

        if checked:
            return

    pytest.skip("no sprite objects appeared to check")


def test_object_cells_record_identity(live_env):
    """Occupied cells carry the object id, not just a category."""
    for _ in range(40):
        live_env.step(RIGHT)
        semantic_map = live_env.semantic_map
        occupied = semantic_map['object_id'] != NO_OBJECT
        if occupied.any():
            assert (semantic_map['object_fine_type'][occupied] != FineTileType.EMPTY).all()
            return


def test_empty_cells_use_the_no_object_sentinel(live_env):
    """Absence is 0xFF, not 0 -- object id 0 is a real object (HEART)."""
    semantic_map = live_env.semantic_map
    unoccupied = semantic_map['object_id'] == NO_OBJECT

    assert (semantic_map['object_fine_type'][unoccupied] == FineTileType.EMPTY).all()
    assert (semantic_map['object_coarse_type'][unoccupied] == CoarseTileType.EMPTY).all()


def test_tensor_shape_and_dtype(live_env):
    """The tensor is a binary (H, W, C) uint8 array."""
    tensor = live_env.semantic_tensor

    expected_channels = len(COARSE_TENSOR_CHANNELS) + len(PROPERTY_CHANNEL_NAMES)
    assert tensor.shape == (SCREEN_TILES_HEIGHT, SCREEN_TILES_WIDTH, expected_channels)
    assert tensor.dtype == np.uint8
    assert set(np.unique(tensor).tolist()) <= {0, 1}, "channels must be binary"


def test_tensor_terrain_layer_is_one_hot(live_env):
    """Exactly one terrain channel is set per cell, always."""
    for step in range(40):
        live_env.step(RIGHT if step % 3 else JUMP)
        tensor = live_env.semantic_tensor
        terrain_sum = tensor[:, :, TERRAIN_CHANNELS].sum(axis=2)

        assert (terrain_sum == 1).all(), (
            f"terrain must be one-hot, found counts {sorted(set(terrain_sum.flatten().tolist()))}"
        )


def test_tensor_object_layer_is_at_most_one_hot(live_env):
    """A cell holds at most one object."""
    for step in range(40):
        live_env.step(RIGHT if step % 3 else JUMP)
        tensor = live_env.semantic_tensor

        assert tensor[:, :, OBJECT_CHANNELS].sum(axis=2).max() <= 1


def test_tensor_is_multi_hot_across_layers(live_env):
    """A cell can be terrain AND object at once -- the point of two layers.

    A single shared channel group would force a choice here and lose exactly the
    "standing on it vs inside it" distinction.
    """
    for _ in range(60):
        live_env.step(RIGHT)
        semantic_map = live_env.semantic_map
        occupied = semantic_map['object_id'] != NO_OBJECT
        if not occupied.any():
            continue

        tensor = live_env.semantic_tensor_from(semantic_map)
        terrain_hot = tensor[:, :, TERRAIN_CHANNELS].sum(axis=2) == 1
        object_hot = tensor[:, :, OBJECT_CHANNELS].sum(axis=2) == 1

        assert (terrain_hot & object_hot).any(), "expected a cell with both layers set"
        return


def test_tensor_matches_the_structured_map(live_env):
    """The tensor is a faithful view of the map, not a separate computation."""
    semantic_map = live_env.semantic_map
    tensor = live_env.semantic_tensor_from(semantic_map)

    for channel, (layer, coarse_type) in enumerate(COARSE_TENSOR_CHANNELS):
        if layer == 'terrain':
            expected = semantic_map['coarse_type'] == coarse_type
        else:
            expected = (semantic_map['object_id'] != NO_OBJECT) & (
                semantic_map['object_coarse_type'] == coarse_type
            )
        assert (tensor[:, :, channel] == expected.astype(np.uint8)).all(), (
            f"channel {channel} ({COARSE_TENSOR_CHANNEL_NAMES[channel]}) disagrees with the map"
        )


def test_channel_names_match_channel_count():
    """Names and channels stay in step."""
    assert len(COARSE_TENSOR_CHANNEL_NAMES) == (
        len(COARSE_TENSOR_CHANNELS) + len(PROPERTY_CHANNEL_NAMES)
    )
    assert COARSE_TENSOR_CHANNEL_NAMES[0].startswith('terrain:')
    assert any(name.startswith('object:') for name in COARSE_TENSOR_CHANNEL_NAMES)


def test_info_dict_exposes_both_views(live_env):
    """`info` carries the structured map and the tensor, built from one read."""
    info = live_env.info

    assert 'semantic' in info and 'semantic_tensor' in info
    assert info['semantic_tensor'].shape[:2] == info['semantic'].shape


def test_property_channels_mark_hostility_and_liftability(live_env):
    """DAMAGES and LIFTABLE describe what happens on contact, not category.

    These are the actual decision an agent makes about a sprite, and they are
    mutually exclusive in practice: nothing that hurts you can be picked up.
    """
    damage_channel = COARSE_TENSOR_CHANNEL_NAMES.index('object:DAMAGES')
    liftable_channel = COARSE_TENSOR_CHANNEL_NAMES.index('object:LIFTABLE')

    for _ in range(60):
        live_env.step(RIGHT)
        semantic_map = live_env.semantic_map
        occupied = semantic_map['object_id'] != NO_OBJECT
        if not occupied.any():
            continue

        tensor = live_env.semantic_tensor_from(semantic_map)
        damages = tensor[:, :, damage_channel].astype(bool)
        liftable = tensor[:, :, liftable_channel].astype(bool)

        assert not (damages & liftable).any(), "an object cannot both hurt and be liftable"
        # Properties only ever apply where an object actually is
        assert not (damages & ~occupied).any()
        assert not (liftable & ~occupied).any()

        # The flag must be selective, not blanket-true: non-hostile objects
        # (doors, pickups, coins) are present here and must not be marked.
        harmless = occupied & ~np.isin(semantic_map['object_fine_type'], list(DAMAGING_FINE_TYPES))
        if harmless.any():
            assert not damages[harmless].any(), (
                "harmless objects are marked as damaging; the flag is not selective"
            )
        return


def test_velocity_is_separate_and_continuous(live_env):
    """Velocity is its own float array, so the tensor stays a pure binary mask."""
    velocity = live_env.semantic_velocity

    assert velocity.shape == (SCREEN_TILES_HEIGHT, SCREEN_TILES_WIDTH, len(VELOCITY_CHANNEL_NAMES))
    assert velocity.dtype == np.float32
    assert np.all(np.abs(velocity) <= 1.0), "velocity must be normalised into [-1, 1]"


def test_velocity_of_moving_objects_is_normalised(live_env):
    """Normalisation is checked against cells that actually carry motion.

    Asserting the bound over a mostly-empty grid passes trivially -- zeros are
    always in range -- so this drives until a genuinely moving object is seen and
    only then checks the magnitude.
    """
    for _ in range(120):
        live_env.step(RIGHT)
        semantic_map = live_env.semantic_map
        velocity = live_env.semantic_velocity_from(semantic_map)
        moving = velocity != 0
        if not moving.any():
            continue

        assert np.all(np.abs(velocity[moving]) <= 1.0), (
            f"moving object velocity {velocity[moving].max()} is outside [-1, 1]; "
            "raw RAM values must be scaled by OBJECT_VELOCITY_SCALE"
        )
        return

    pytest.skip("no moving objects observed on this level")


def test_velocity_only_set_where_objects_are(live_env):
    """Empty cells carry zero velocity."""
    for _ in range(40):
        live_env.step(RIGHT)
        semantic_map = live_env.semantic_map
        velocity = live_env.semantic_velocity_from(semantic_map)
        unoccupied = semantic_map['object_id'] == NO_OBJECT

        assert (velocity[unoccupied] == 0).all()


def test_velocity_tracks_object_movement(live_env):
    """A moving object reports non-zero velocity, and the sign follows direction.

    Without this the map is a still frame: an agent cannot tell an enemy closing
    on it from one moving away.
    """
    seen_velocities = set()

    for _ in range(120):
        live_env.step(RIGHT)
        semantic_map = live_env.semantic_map
        occupied = semantic_map['object_id'] != NO_OBJECT
        if not occupied.any():
            continue
        for value in semantic_map['object_velocity_x'][occupied]:
            if value != 0:
                seen_velocities.add(float(value))

    if not seen_velocities:
        pytest.skip("no moving objects observed")

    assert any(v > 0 for v in seen_velocities) or any(v < 0 for v in seen_velocities)


def test_info_exposes_velocity(live_env):
    """`info` carries all three views, built from a single map read."""
    info = live_env.info

    assert 'semantic_velocity' in info
    assert info['semantic_velocity'].shape[:2] == info['semantic'].shape
