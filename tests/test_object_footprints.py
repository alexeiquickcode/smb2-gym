"""Tests for sprite object footprints in the semantic map.

Object sizes are recovered from the OAM sprites being drawn, since the game does
not expose dimensions in RAM. The failure mode these guard against: two objects
close together chain into one sprite cluster and both report the combined
extent. A Cobrat in its jar sits 6px from the bullet it just spat, which made an
8px bullet occupy four tiles of the observation tensor.
"""

import os

import numpy as np
import pytest

from smb2_gym.app import InitConfig
from smb2_gym.constants import (
    NO_OBJECT,
    OBJECT_ID_MAPPING,
    SINGLE_TILE_FINE_TYPES,
    FineTileType,
)
from smb2_gym.smb2_env import SuperMarioBros2Env
from smb2_gym.state.semantic_map import _is_nearer_other_object


# The Cobrat-and-bullet scene that exposed the merging bug: level 2-1, reached by
# running right for this many frames.
BUG_LEVEL = "2-1"
BUG_FRAME = 863


@pytest.fixture(scope="module")
def cobrat_env():
    """Environment loaded at the level containing the Cobrat/bullet pair."""
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = os.path.join(package_dir, "smb2_gym", "_nes", "prg0")
    rom = os.path.join(base, "super_mario_bros_2_prg0.nes")
    save = os.path.join(base, "saves", "select", f"{BUG_LEVEL}.sav")
    if not (os.path.exists(rom) and os.path.exists(save)):
        pytest.skip("ROM or save state not available")

    env = SuperMarioBros2Env(init_config=InitConfig(rom_path=rom, save_state_path=save))
    yield env
    env.close()


# ------------------------------------------------------------------------------
# ---- Sprite ownership (pure, no emulator) ------------------------------------
# ------------------------------------------------------------------------------


def test_sprite_belongs_to_its_own_object():
    """A sprite right on an object's anchor is not stolen by a distant one."""
    assert not _is_nearer_other_object(10, 80, 10, 80, [(200, 80)])


def test_sprite_nearer_another_object_is_excluded():
    """A sprite hugging a neighbour must not count toward this object's size."""
    assert _is_nearer_other_object(198, 80, 10, 80, [(200, 80)])


def test_ties_favour_the_measured_object():
    """A sprite equidistant from two objects is kept, not dropped from both."""
    assert not _is_nearer_other_object(50, 80, 40, 80, [(60, 80)])


def test_no_other_objects_excludes_nothing():
    assert not _is_nearer_other_object(10, 80, 10, 80, [])


# ------------------------------------------------------------------------------
# ---- Footprints in a live environment ----------------------------------------
# ------------------------------------------------------------------------------


def _run_to_frame(env, frames):
    """Walk right, jumping when stuck, and return the info at the last frame."""
    obs, info = env.reset()
    stuck, prev_x = 0, None
    for _ in range(frames):
        action = 6 if stuck > 6 else 1
        obs, _, terminated, truncated, info = env.step(np.int64(action))
        if terminated or truncated:
            break
        x = info['pos'].x_global
        stuck = 0 if (prev_x is not None and x != prev_x) else stuck + 1
        prev_x = x
    return info


@pytest.mark.slow
def test_bullet_beside_cobrat_stays_one_cell(cobrat_env):
    """Regression: the reported scene, where a bullet spanned four tiles.

    In level 2-1 a Cobrat rears from its jar at the screen's left edge with the
    bullet it just spat 6px away. Their OAM sprites chain into one cluster, so
    without ownership handling the 8px bullet inherited the jar's full extent.
    """
    info = _run_to_frame(cobrat_env, BUG_FRAME)
    semantic_map = cobrat_env.semantic_map
    ids = semantic_map['object_id']

    projectile_cells = [
        (x, y)
        for y in range(ids.shape[0])
        for x in range(ids.shape[1])
        if int(ids[y, x]) != NO_OBJECT
        and semantic_map['object_fine_type'][y, x] == FineTileType.PROJECTILE
    ]

    # Each on-screen projectile may claim at most one cell
    projectiles_on_screen = sum(
        1
        for _, _, e in cobrat_env._get_visible_objects()
        if OBJECT_ID_MAPPING.get(e.object_type) == FineTileType.PROJECTILE
    )
    assert len(projectile_cells) <= projectiles_on_screen, (
        f"{len(projectile_cells)} projectile cells for "
        f"{projectiles_on_screen} on-screen projectile(s): {projectile_cells}"
    )

    # And none of them may form a vertical run, which is what the bug looked like
    columns = {}
    for x, y in projectile_cells:
        columns.setdefault(x, []).append(y)
    for x, ys in columns.items():
        assert len(ys) == 1, f"projectile spans {len(ys)} rows in column {x}: {sorted(ys)}"


@pytest.mark.slow
def test_small_objects_never_exceed_one_cell(env_no_render):
    """Projectiles, coins, cherries and hearts must occupy exactly one cell.

    These are small by definition, so a multi-cell footprint means the measure
    absorbed a neighbouring object's sprites.
    """
    env = env_no_render
    obs, info = env.reset()
    stuck, prev_x = 0, None

    for _ in range(400):
        action = 6 if stuck > 6 else 1
        obs, _, terminated, truncated, info = env.step(np.int64(action))
        if terminated or truncated:
            break
        x = info['pos'].x_global
        stuck = 0 if (prev_x is not None and x != prev_x) else stuck + 1
        prev_x = x

        semantic_map = env.semantic_map
        ids = semantic_map['object_id']

        # Group by cell rather than by id: two separate fireballs share an id
        for _, _, enemy in env._get_visible_objects():
            fine_type = OBJECT_ID_MAPPING.get(enemy.object_type)
            if fine_type not in SINGLE_TILE_FINE_TYPES:
                continue
            matching = int((ids == enemy.object_type).sum())
            on_screen = sum(
                1
                for _, _, other in env._get_visible_objects()
                if other.object_type == enemy.object_type
            )
            assert matching <= on_screen, (
                f"{FineTileType(fine_type).name} occupies {matching} cells "
                f"but only {on_screen} are on screen"
            )


@pytest.mark.slow
def test_object_footprints_stay_within_bounds(env_no_render):
    """No object may claim more cells than the largest legitimate sprite."""
    env = env_no_render
    obs, info = env.reset()
    stuck, prev_x = 0, None

    for _ in range(300):
        action = 6 if stuck > 6 else 1
        obs, _, terminated, truncated, info = env.step(np.int64(action))
        if terminated or truncated:
            break
        x = info['pos'].x_global
        stuck = 0 if (prev_x is not None and x != prev_x) else stuck + 1
        prev_x = x

        visible = env._get_visible_objects()
        for i, (px, py, _) in enumerate(visible):
            others = [(ox, oy) for j, (ox, oy, _) in enumerate(visible) if j != i]
            width, height = env._measure_object_size(px, py, others)
            assert 1 <= width <= 4 and 1 <= height <= 4, f"implausible size {width}x{height}"


@pytest.mark.slow
def test_measure_shrinks_when_neighbours_are_known(env_no_render):
    """Passing the other anchors must never grow a footprint, only shrink it."""
    env = env_no_render
    obs, info = env.reset()
    stuck, prev_x = 0, None

    for _ in range(300):
        action = 6 if stuck > 6 else 1
        obs, _, terminated, truncated, info = env.step(np.int64(action))
        if terminated or truncated:
            break
        x = info['pos'].x_global
        stuck = 0 if (prev_x is not None and x != prev_x) else stuck + 1
        prev_x = x

        visible = env._get_visible_objects()
        for i, (px, py, _) in enumerate(visible):
            others = [(ox, oy) for j, (ox, oy, _) in enumerate(visible) if j != i]
            naive_w, naive_h = env._measure_object_size(px, py)
            aware_w, aware_h = env._measure_object_size(px, py, others)
            assert aware_w <= naive_w and aware_h <= naive_h
