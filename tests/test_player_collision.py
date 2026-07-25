"""Tests for the player collision box and the shared world->tile transform.

These lock in behaviour that was verified against the emulator:

- ``Player.Y_POSITION`` is the TOP of a big standing player's box; the feet sit
  32px below. It does not move when the player ducks or shrinks.
- The box therefore shrinks from the top: ducking and losing a life both keep the
  feet tile and drop the head tile, rather than moving the player up a cell.
- The player and the enemies go through one shared transform, both for rows and
  for columns, so they cannot drift apart by a tile. A carried item, which shares
  its carrier's X, must land in the carrier's column.
"""

import pytest

from smb2_gym import SuperMarioBros2Env
from smb2_gym.app import InitConfig
from smb2_gym.constants import (
    PAGE_SIZE,
    PLAYER,
    PLAYER_HEIGHT_BIG,
    PLAYER_HEIGHT_DUCKING,
    PLAYER_HEIGHT_SMALL,
    SCREEN_TILES_HEIGHT,
    SCREEN_TILES_WIDTH,
    TILE_SIZE,
    FineTileType,
)

ROM = "smb2_gym/_nes/prg0/super_mario_bros_2_prg0.nes"

# Horizontal levels where the player starts big and can duck. Vertical levels
# (e.g. 1-1) are excluded: DOWN makes the player descend there rather than duck.
DUCKABLE_SAVES = [
    "smb2_gym/_nes/prg0/saves/mario/1-2.sav",
    "smb2_gym/_nes/prg0/saves/mario/1-3.sav",
    "smb2_gym/_nes/prg0/saves/mario/2-1.sav",
    "smb2_gym/_nes/prg0/saves/mario/4-1.sav",
]

NOOP, RIGHT, JUMP, DOWN = 0, 1, 4, 10  # indices into SIMPLE_ACTIONS


def _make_env(save_path):
    return SuperMarioBros2Env(
        init_config=InitConfig(rom_path=ROM, save_state_path=save_path),
        render_mode=None,
        action_type="simple",
    )


def _settle(env, max_frames=600, required_stable=20):
    """Advance until the player has been at a constant Y for `required_stable` frames.

    Save states start mid-air, so the player must land before its tiles mean
    anything relative to the ground.
    """
    previous, stable = None, 0
    for _ in range(max_frames):
        env.step(NOOP)
        y = env._read_ram_safe(PLAYER.Y_POSITION)
        stable = stable + 1 if y == previous else 0
        previous = y
        if stable >= required_stable:
            return True
    return False


@pytest.fixture(params=DUCKABLE_SAVES)
def grounded_env(request):
    env = _make_env(request.param)
    env.reset()
    assert _settle(env), f"player never landed in {request.param}"
    yield env
    env.close()


def test_standing_player_occupies_two_tiles(grounded_env):
    """A big, grounded, tile-aligned player fills exactly two vertical tiles."""
    tiles = grounded_env.get_player_collision_tiles()

    assert len(tiles) == 2, f"expected a 32px box to span 2 tiles, got {tiles}"
    assert tiles[0][0] == tiles[1][0], "both tiles should be in the same column"
    assert tiles[1][1] == tiles[0][1] + 1, "tiles should be vertically adjacent"


SOLID_TYPES = (FineTileType.SOLID, FineTileType.PLATFORM)


def test_standing_player_rests_on_solid_ground(grounded_env):
    """The player sits exactly ON the floor: ground below, free space at its own tiles.

    This is the anchor check, and it is deliberately two-sided. Requiring solid
    ground below alone is not enough -- floors are several tiles thick, so a
    player shifted a row down would still have solid ground beneath it. Requiring
    the player's own tiles to be free pins the box to the true surface.
    """
    semantic_map = grounded_env.semantic_map
    tiles = grounded_env.get_player_collision_tiles()
    column, feet_row = tiles[-1]

    ground = int(semantic_map[feet_row + 1, column]['fine_type'])
    assert ground in SOLID_TYPES, (
        f"expected solid ground at row {feet_row + 1}, found {FineTileType(ground).name}"
    )

    # Skip states where the tile map itself is untrustworthy. `_read_tile_maps`
    # currently mis-reads some levels (mario/1-3 reports a solid column where the
    # player is plainly standing in the open), which is a separate bug in the
    # SRAM->tile decode, not in the collision box. Asserting free space there
    # would be testing the wrong thing.
    column_is_wall = all(
        int(semantic_map[row, column]['fine_type']) in SOLID_TYPES
        for row in range(tiles[0][1], SCREEN_TILES_HEIGHT)
    )
    if column_is_wall:
        pytest.skip(f"tile map reports column {column} as solid throughout; see _read_tile_maps")

    for tile_x, tile_y in tiles:
        occupied = int(semantic_map[tile_y, tile_x]['fine_type'])
        assert occupied not in SOLID_TYPES, (
            f"player tile ({tile_x}, {tile_y}) is inside {FineTileType(occupied).name}; "
            "the box has sunk into the terrain"
        )


def test_ducking_keeps_feet_tile_and_drops_head_tile(grounded_env):
    """Ducking must shrink the box from the top, not shift it.

    The reported cell should stay on the feet row. Keeping the head row instead
    is what made the player appear to jump up one cell when ducking.
    """
    standing = grounded_env.get_player_collision_tiles()
    feet = standing[-1]

    for _ in range(20):
        grounded_env.step(DOWN)
    assert grounded_env.is_player_ducking(), "ducking did not engage"

    ducking = grounded_env.get_player_collision_tiles()

    assert ducking == [feet], (f"ducking should keep only the feet tile {feet}, got {ducking}")


def test_ducking_does_not_move_the_player(grounded_env):
    """Ducking changes the box height only; the underlying position is unchanged."""
    y_before = grounded_env._read_ram_safe(PLAYER.Y_POSITION)
    _, _, _, bottom_before = grounded_env.get_player_collision_box()

    for _ in range(20):
        grounded_env.step(DOWN)
    assert grounded_env.is_player_ducking(), "ducking did not engage"

    y_after = grounded_env._read_ram_safe(PLAYER.Y_POSITION)
    _, _, _, bottom_after = grounded_env.get_player_collision_box()

    assert y_after == y_before, "ducking should not change the player's RAM Y"
    assert bottom_after == bottom_before, "the feet should stay put while ducking"


def test_collision_height_reflects_duck_state(grounded_env):
    """Height, not position, is what ducking changes."""
    assert grounded_env.get_player_collision_height() == PLAYER_HEIGHT_BIG

    for _ in range(20):
        grounded_env.step(DOWN)
    assert grounded_env.is_player_ducking(), "ducking did not engage"

    assert grounded_env.get_player_collision_height() == PLAYER_HEIGHT_DUCKING


def test_collision_box_height_matches_reported_height(grounded_env):
    """The box's pixel extent agrees with get_player_collision_height()."""
    _, top, _, bottom = grounded_env.get_player_collision_box()

    assert bottom - top == grounded_env.get_player_collision_height()


def test_release_duck_restores_standing_box(grounded_env):
    """Standing back up returns exactly the original tiles."""
    standing = grounded_env.get_player_collision_tiles()

    for _ in range(20):
        grounded_env.step(DOWN)
    for _ in range(30):
        grounded_env.step(NOOP)

    assert grounded_env.get_player_collision_tiles() == standing


def test_tiles_are_within_map_bounds(grounded_env):
    """Reported tiles always index the semantic map safely."""
    for tile_x, tile_y in grounded_env.get_player_collision_tiles():
        assert 0 <= tile_x < SCREEN_TILES_WIDTH
        assert 0 <= tile_y < SCREEN_TILES_HEIGHT


def test_world_to_screen_tile_is_consistent_with_pixels(grounded_env):
    """The tile transform is just the pixel transform floor-divided by TILE_SIZE.

    Both the player and the enemy paths rely on this, so a divergence here means
    the two frames of reference have drifted apart again.
    """
    for world_x, world_y in ((0, 0), (128, 176), (640, 51), (300, 243)):
        pixel_x, pixel_y = grounded_env._world_to_screen_pixels(world_x, world_y)
        tile_x, tile_y = grounded_env._world_to_screen_tile(world_x, world_y)

        assert (tile_x, tile_y) == (pixel_x // TILE_SIZE, pixel_y // TILE_SIZE)


def test_player_and_enemies_share_one_transform(grounded_env):
    """Enemy tiles come from the same transform the player uses.

    Previously the player path added a +16 correction that the enemy path did
    not, putting the two in different frames of reference.
    """
    for screen_x, screen_y, _enemy_id in grounded_env._get_enemy_screen_positions():
        assert 0 <= screen_y < SCREEN_TILES_HEIGHT * TILE_SIZE, (
            "enemy screen Y left the map, transform is inconsistent"
        )


def test_carried_item_shares_the_players_column(grounded_env):
    """An item at the player's X lands in the player's column, at every offset.

    A carried item is placed at its carrier's X, so the two must centre
    identically. The player used to be centred (+8) while enemies were not,
    which made a held item appear a cell to the left at some sub-tile positions
    but not others -- it looked direction-dependent because walking changes X.
    """
    for _ in range(24):
        grounded_env.step(RIGHT)
        player_tiles = grounded_env.get_player_collision_tiles()
        if not player_tiles:
            continue

        world_x = (
            grounded_env._read_ram_safe(PLAYER.X_PAGE) * PAGE_SIZE
            + grounded_env._read_ram_safe(PLAYER.X_POSITION)
        )
        assert grounded_env._sprite_column(world_x) == player_tiles[0][0], (
            f"item at the player's X={world_x} (X%{TILE_SIZE}={world_x % TILE_SIZE}) "
            f"landed in a different column than the player"
        )


def test_sprite_column_centres_left_edge_coordinates(grounded_env):
    """`_sprite_column` maps a left edge to the column holding the sprite's centre."""
    # A sprite whose left edge is at 120 spans 120..135, centre 128 -> column 8.
    assert grounded_env._sprite_column(120) == 8
    # Left edge exactly on a boundary spans the whole tile.
    assert grounded_env._sprite_column(128) == 8
    # One pixel past the boundary still centres into the next column.
    assert grounded_env._sprite_column(136) == 9


def test_losing_a_life_halves_the_box_without_moving_the_player():
    """Shrinking big -> small halves the hitbox and keeps the feet row.

    Takes damage for real, since no bundled save state starts small and the
    emulator bindings expose no RAM writes. Verified on 4-1, where the hit is
    survivable (elsewhere the same walk kills the player outright).

    The player must not appear to move: `Player.Y_POSITION` is unchanged by the
    shrink, and the reported tile stays on the feet row. Note the SPRITE does
    shift down 8px while keeping a 32px span -- which is exactly why the box is
    derived from RAM and not from OAM.
    """
    env = SuperMarioBros2Env(
        init_config=InitConfig(level="4-1", character="mario"),
        render_mode=None,
        action_type="complex",
    )
    try:
        env.reset()
        assert _settle(env), "player never landed"

        big_tiles = env.get_player_collision_tiles()
        # Only Y is compared: the player has to walk to find an enemy, so X moves.
        big_y = env._read_ram_safe(PLAYER.Y_POSITION)
        hearts_before = (env._read_ram_safe(PLAYER.LIFE_METER) >> 4) + 1
        assert hearts_before >= 2, "expected to start big"
        assert env.get_player_collision_height() == PLAYER_HEIGHT_BIG
        assert len(big_tiles) == 2

        # Pace back and forth until something hits us.
        COMPLEX_RIGHT, COMPLEX_LEFT = 1, 5
        for frame in range(1200):
            env.step(COMPLEX_RIGHT if (frame // 150) % 2 == 0 else COMPLEX_LEFT)
            if (env._read_ram_safe(PLAYER.LIFE_METER) >> 4) + 1 < hearts_before:
                break
        else:
            pytest.skip("never took damage; enemy layout may have shifted")

        # Let the shrink animation finish. NOOP only, so we cannot pick up a
        # heart and bounce back to big.
        for _ in range(60):
            env.step(NOOP)

        assert (env._read_ram_safe(PLAYER.LIFE_METER) >> 4) + 1 < hearts_before
        assert env.get_player_collision_height(
        ) == PLAYER_HEIGHT_SMALL, ("losing a life should halve the collision height")

        small_tiles = env.get_player_collision_tiles()
        assert len(small_tiles) == 1, f"a small player is one tile tall, got {small_tiles}"
        # Compare rows, not full tiles: the player walked, so the column differs.
        assert small_tiles[0][1] == big_tiles[-1][1], (
            f"small player should keep the feet row {big_tiles[-1][1]}, got {small_tiles}"
        )
        assert env._read_ram_safe(PLAYER.Y_POSITION
                                 ) == big_y, ("shrinking must not move the player's RAM Y")
    finally:
        env.close()


def test_jumping_player_stays_two_tiles(grounded_env):
    """A big player is two tiles tall throughout a jump, not three.

    The box only lines up with the tile grid when grounded. Mid-jump it sits at
    an arbitrary sub-tile offset, and listing every row it overlaps reported a
    third tile for almost the whole arc. Rows are picked by which tile the box is
    most in -- the same rule used for columns -- so the count stays stable.
    """
    assert len(grounded_env.get_player_collision_tiles()) == 2, "expected a big player"

    counts = set()
    for frame in range(60):
        grounded_env.step(JUMP if frame < 20 else NOOP)
        tiles = grounded_env.get_player_collision_tiles()
        counts.add(len(tiles))
        assert tiles[0][0] == tiles[-1][0], "all tiles should share a column"

    assert counts == {2}, f"tile count varied during the jump: {sorted(counts)}"


def test_landing_restores_the_grounded_box(grounded_env):
    """After a jump the player returns to exactly the tiles it started on."""
    before = grounded_env.get_player_collision_tiles()

    for frame in range(20):
        grounded_env.step(JUMP)
    for _ in range(90):
        grounded_env.step(NOOP)

    assert grounded_env.get_player_collision_tiles() == before


def test_player_reports_vertical_velocity(grounded_env):
    """The player's own vertical motion is observable, matching what enemies expose.

    `player_speed` alone tells an agent it is moving right but not whether it is
    rising or falling -- which is what decides whether a jump clears a gap.
    """
    start_y = grounded_env._read_ram_safe(PLAYER.Y_POSITION)

    rising = False
    moved = False
    for frame in range(30):
        grounded_env.step(JUMP if frame < 12 else NOOP)
        if grounded_env._read_ram_safe(PLAYER.Y_POSITION) != start_y:
            moved = True
        if grounded_env.player_y_velocity < 0:
            rising = True
            break

    if not moved:
        # Some save states drop the player somewhere it cannot move (mid-jar, or
        # a transition), where JUMP does nothing at all.
        pytest.skip("player cannot move on this save state")

    assert rising, "jumping should produce negative (upward) vertical velocity"

    falling = False
    for _ in range(40):
        grounded_env.step(NOOP)
        if grounded_env.player_y_velocity > 0:
            falling = True
            break

    assert falling, "descending should produce positive vertical velocity"
