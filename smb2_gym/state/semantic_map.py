"""Semantic tile map for SMB2 environment."""

import warnings
from collections.abc import Sequence
from typing import (
    Any,
)

import numpy as np
from numpy.typing import NDArray
from tetanes_py import NesEnv

from ..constants import (
    GAME_STATE,
    LEVEL_PAGE_HEIGHT,
    LEVEL_PAGE_WIDTH,
    OAM_SPRITE_COUNT,
    OAM_SPRITE_HEIGHT,
    OAM_SPRITE_WIDTH,
    OBJECT_MAX_TILES,
    OBJECT_SPRITE_MAX_HEIGHT,
    OBJECT_SPRITE_MAX_WIDTH,
    OBJECT_VELOCITY_SCALE,
    PAGE_SIZE,
    PLAYER,
    PLAYER_HEIGHT_BIG,
    PLAYER_HEIGHT_DUCKING,
    PLAYER_HEIGHT_SMALL,
    PLAYER_OAM_INDICES,
    SCREEN_HEIGHT,
    SCREEN_TILES_HEIGHT,
    SCREEN_TILES_WIDTH,
    TILE_SIZE,
    VIEWPORT,
    EnemyState,
    SpriteFlags,
)
from ..constants.semantic import (
    COARSE_LOOKUP,
    COARSE_TENSOR_CHANNELS,
    COLOR_LOOKUP,
    DAMAGING_FINE_TYPES,
    LIFTABLE_FINE_TYPES,
    NO_OBJECT,
    OBJECT_ID_MAPPING,
    PROPERTY_CHANNEL_NAMES,
    SEMANTIC_TILE_DTYPE,
    SINGLE_TILE_FINE_TYPES,
    TILE_ID_MAPPING,
    UNMAPPED_OBJECT_FINE_TYPE,
    FineTileType,
)
from ._base import (
    GameStateMixin,
    HasEnemies,
)


def _is_nearer_other_object(
    sprite_x: int,
    sprite_y: int,
    anchor_x: int,
    anchor_y: int,
    other_anchors: Sequence[tuple[int, int]],
) -> bool:
    """Whether an OAM sprite sits closer to another object than to this one.

    Distances are measured to the anchor (an object's top-left), with Y weighted
    the same as X. Ties favour the object being measured, so a sprite exactly
    between two objects is not dropped from both.

    This alone cannot separate objects whose anchors nearly coincide -- a Cobrat
    and the bullet it just spat sit 6px apart, so every nearby sprite is
    "nearest" to both. The anchor-relative bound in `_measure_object_size`
    handles that case; this handles objects that are merely adjacent.
    """
    own = (sprite_x - anchor_x) ** 2 + (sprite_y - anchor_y) ** 2
    return any((sprite_x - ox) ** 2 + (sprite_y - oy) ** 2 < own for ox, oy in other_anchors)


class SemanticMapMixin(GameStateMixin, HasEnemies):
    """Mixin providing semantic tile map for SMB2 environment.

    This mixin provides access to semantic tile information including:
    - Tile types (SOLID, ENEMY, COLLECTIBLE, etc.)
    - Tile categories (TERRAIN, HAZARD, etc.)
    - Player position on the tile grid
    - Enemy positions overlaid on the map

    Note: This mixin depends on the `enemies` property being provided by EnemiesMixin.
    """

    _nes: NesEnv  # Parent class for type checking

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    # ---- Sprites ---------------------------------------------------

    def _read_player_sprites(self) -> list[tuple[int, int, int, int]]:
        """Read player sprite data from OAM.

        Player character sprites are always at fixed OAM indices 8-11.
        Returns list of sprite entries for the player, each with (y, tile_id, attributes, x).
        """
        oam_data: list[tuple[int, int, int, int]] = []

        for sprite_index in range(8, 12):
            sprite_data = self._nes.read_oam_sprite(sprite_index)
            if sprite_data is None:
                continue

            y_pos, tile_id, attributes, x_pos = sprite_data

            oam_data.append((y_pos, tile_id, attributes, x_pos))

        return oam_data

    def get_player_sprite_position(self) -> tuple[int, int] | None:
        """Get the player character sprite position from OAM.

        Player sprites are always at fixed OAM indices 8-11.
        Returns the top-left position of the player sprite.

        Returns:
            tuple of (x_pixel, y_pixel) on screen, or None if not found
        """
        oam_sprites = self._read_player_sprites()
        if not oam_sprites:
            return None

        min_x = min(sprite[3] for sprite in oam_sprites)  # x_pos is index 3
        min_y = min(sprite[0] for sprite in oam_sprites)  # y_pos is index 0

        return (min_x, min_y)

    # ---- Coordinate transforms -------------------------------------

    def _world_to_screen_pixels(self, world_x: int, world_y: int) -> tuple[int, int]:
        """Convert world pixel coordinates to semantic-map screen pixel coordinates.

        This is the single place where the viewport offset is applied. Both the
        player and the enemies must go through it, otherwise the two end up in
        different frames of reference and drift apart by a tile.

        No status-bar correction is applied: world Y already aligns with the
        semantic map's rows, verified by checking that a grounded player's feet
        land exactly on the first SOLID row beneath them.

        Args:
            world_x: X position in world pixels
            world_y: Y position in world pixels

        Returns:
            tuple of (screen_x, screen_y) in pixels, relative to the semantic map origin
        """
        viewport_x, viewport_y = self._get_viewport_offset()

        screen_x = world_x - (viewport_x * TILE_SIZE)
        screen_y = world_y - (viewport_y * TILE_SIZE)

        return screen_x, screen_y

    def _world_to_screen_tile(self, world_x: int, world_y: int) -> tuple[int, int]:
        """Convert world pixel coordinates to a semantic-map tile index.

        Args:
            world_x: X position in world pixels
            world_y: Y position in world pixels

        Returns:
            tuple of (tile_x, tile_y) indices into the semantic map
        """
        screen_x, screen_y = self._world_to_screen_pixels(world_x, world_y)
        return screen_x // TILE_SIZE, screen_y // TILE_SIZE

    def _sprite_column(self, world_x: int) -> int:
        """Get the tile column a sprite occupies, from its left-edge world X.

        Sprites are a tile wide but rarely tile-aligned, so they usually straddle
        two columns; the one containing their centre is the column they are most
        in. Both the player and the enemies store their X as a left edge, so both
        must be centred the same way -- otherwise a carried item, which shares its
        carrier's X, lands in a different column than the carrier.

        Args:
            world_x: Left edge of the sprite, in world pixels

        Returns:
            Tile column index
        """
        screen_x, _ = self._world_to_screen_pixels(world_x + (TILE_SIZE // 2), 0)
        return screen_x // TILE_SIZE

    # ---- Player State ----------------------------------------------

    def is_player_ducking(self) -> bool:
        """Check if the player is currently ducking/crouching.

        Ducking does not move the player: its RAM Y, and its sprite bounds, are
        identical standing and ducking. What changes is the collision box height
        (see `get_player_collision_height`), which is why this is detected from
        the sprite tile rather than from any position delta.

        Returns:
            True if player is ducking, False otherwise
        """
        # Check if player has 2+ hearts (is big)
        life_meter = self._read_ram_safe(PLAYER.LIFE_METER)
        num_hearts = (life_meter >> 4) + 1

        # Only big players can duck
        if num_hearts < 2:
            return False

        # Check sprite indices 8 and 9 for the $FB tile (ducking sprite)
        for sprite_index in [8, 9]:
            sprite_data = self._nes.read_oam_sprite(sprite_index)
            if sprite_data is not None:
                _y_pos, tile_id, _attributes, _x_pos = sprite_data

                # Check if this is the ducking tile ($FB)
                if tile_id == 0xFB:
                    return True

        return False

    def get_player_collision_height(self) -> int:
        """Get the height of the player's collision box in pixels.

        The box shrinks from the top: ducking and losing a life both halve the
        height while `Player.Y_POSITION` stays where it is.

        Returns:
            Height in pixels (32 when big and standing, 16 when small or ducking)
        """
        life_meter = self._read_ram_safe(PLAYER.LIFE_METER)
        num_hearts = (life_meter >> 4) + 1

        if num_hearts < 2:
            return PLAYER_HEIGHT_SMALL
        if self.is_player_ducking():
            return PLAYER_HEIGHT_DUCKING
        return PLAYER_HEIGHT_BIG

    def get_player_collision_box(self) -> tuple[int, int, int, int]:
        """Get the player's collision box in world pixel coordinates.

        Derived from RAM rather than from OAM sprite bounds. Sprite extents are a
        rendering artefact -- they include animation frames and, because the
        player is drawn as 8x16 sprites, the lowest OAM entry is the top of the
        bottom sprite rather than the player's feet.

        Player.Y_POSITION is the TOP of the box when the player is big and
        standing, and -- crucially -- it does not move when the player ducks or
        shrinks. So the box is built downward from a fixed full-height bottom
        edge, shrinking from the top, rather than from Y downward.

        A caveat on the rendering, which is why OAM must not be used here: a small
        player is still DRAWN as a 32px-tall sprite pair, just shifted 8px down
        (measured big -> small: sprite top 80 -> 88, bottom 112 -> 120, span 32
        both times). Only the collision height actually halves.

        Returns:
            tuple of (left, top, right, bottom) in world pixels, where right/bottom
            are exclusive.
        """
        x_page = self._read_ram_safe(PLAYER.X_PAGE)
        x_pos = self._read_ram_safe(PLAYER.X_POSITION)
        y_page = self._read_ram_safe(PLAYER.Y_PAGE)
        y_pos = self._read_ram_safe(PLAYER.Y_POSITION)

        if y_page == 255:  # Screen wrap-around when moving above the top
            y_page = 0

        world_x = (x_page * PAGE_SIZE) + x_pos
        # Feet are at the bottom of the full-height box, regardless of the
        # player's current height.
        world_y_feet = (y_page * PAGE_SIZE) + y_pos + PLAYER_HEIGHT_BIG

        height = self.get_player_collision_height()
        # The box is one tile wide; RAM X is its left edge.
        return world_x, world_y_feet - height, world_x + TILE_SIZE, world_y_feet

    def get_player_collision_tiles(self) -> list[tuple[int, int]]:
        """Get the tile positions occupied by the player for collision detection.

        The box is anchored at the feet and extends upward by the current
        collision height, so ducking correctly drops the head tile while keeping
        the feet tile, rather than appearing to shift the player up a cell.

        Returns:
            List of (screen_x_tile, screen_y_tile) tuples representing tiles occupied
            by the player, ordered top to bottom.
        """
        left, top, _right, bottom = self.get_player_collision_box()

        tile_x = self._sprite_column(left)
        if not 0 <= tile_x < SCREEN_TILES_WIDTH:
            return []

        # The box covers a whole number of tiles, so report exactly that many:
        # two rows for a 32px player, one for a 16px one. Listing every row the
        # box merely overlaps would report three rows whenever the player is not
        # tile-aligned, which is most of a jump.
        row_count = max(1, round((bottom - top) / TILE_SIZE))

        # Anchor on the feet -- the row the lowest part of the box is most in --
        # and extend upward, so ducking and shrinking drop the head row while the
        # feet row stays put.
        _, feet_row = self._world_to_screen_tile(left, bottom - 1 - (TILE_SIZE // 2))

        return [
            (tile_x, row)
            for row in range(feet_row - row_count + 1, feet_row + 1)
            if 0 <= row < SCREEN_TILES_HEIGHT
        ]

    # ---- Enemy Positions (RAM-based) ------------------------------

    def _get_enemy_screen_positions(self) -> list[tuple[int, int, int]]:
        """Get enemy positions on screen from RAM.

        Reads enemy positions from RAM addresses and converts them to screen coordinates.
        Only returns visible enemies that are on screen.

        Returns:
            List of (x_pixel, y_pixel, enemy_id) tuples for visible enemies on screen.
        """
        return [
            (screen_x, screen_y, enemy.object_type)
            for screen_x, screen_y, enemy in self._get_visible_objects()
        ]

    def _get_visible_objects(self) -> list[tuple[int, int, Any]]:
        """Get on-screen objects with their screen position and full slot data.

        Returns the Enemy record rather than just the id so callers can reach
        velocity and sprite flags without re-reading the slots.

        Returns:
            List of (x_pixel, y_pixel, enemy) tuples for visible objects on screen.
        """

        enemy_positions: list[tuple[int, int, Any]] = []
        for enemy in self.enemies:
            if enemy.state != EnemyState.VISIBLE:
                continue

            if enemy.x_page is None or enemy.x_position is None:
                continue
            if enemy.y_page is None or enemy.y_position is None:
                continue
            if enemy.object_type is None:
                continue

            # Calculate world position in pixels
            # NOTE: y_position is already inverted (y=0 at bottom), but we need raw Y here
            # We need to un-invert it for screen coordinates
            world_x = (enemy.x_page * PAGE_SIZE) + enemy.x_position
            # Convert back to raw Y (top-down) for screen rendering
            world_y_raw = (SCREEN_HEIGHT - 1) - enemy.y_position
            world_y = (enemy.y_page * PAGE_SIZE) + world_y_raw

            # Convert to screen coordinates through the shared transform, so
            # enemies land in the same frame of reference as the player
            screen_x, screen_y = self._world_to_screen_pixels(world_x, world_y)

            # Only include enemies that are on screen
            if 0 <= screen_x < PAGE_SIZE and 0 <= screen_y < SCREEN_TILES_HEIGHT * TILE_SIZE:
                enemy_positions.append((screen_x, screen_y, enemy))

        return enemy_positions

    def _measure_object_size(
        self,
        screen_x: int,
        screen_y: int,
        other_anchors: Sequence[tuple[int, int]] | None = None,
    ) -> tuple[int, int]:
        """Measure a sprite object's size in tiles from the OAM entries drawing it.

        The game does not expose object dimensions in RAM -- the only size-related
        flag is WIDE_SPRITE, which applies to Mouser alone -- so the size is
        recovered from what is actually being drawn. This keeps oversized objects
        (a 1x3 Hawkmouth, a 1x2 Birdo) correct without hand-maintaining a table,
        and stays right for animation frames and boss variants.

        Args:
            screen_x: Object's left edge in screen pixels
            screen_y: Object's top edge in screen pixels
            other_anchors: Screen positions of the other on-screen objects. Sprites
                nearer one of these than to this object are excluded, so two
                objects standing close together do not merge into one footprint.

        Returns:
            tuple of (width_tiles, height_tiles), at least 1x1
        """
        # The player is drawn at fixed OAM indices, so exclude it: its sprites sit
        # right on top of whatever it is standing on and would inflate the match.
        candidates: list[tuple[int, int]] = []

        for sprite_index in range(OAM_SPRITE_COUNT):
            if sprite_index in PLAYER_OAM_INDICES:
                continue

            sprite_data = self._nes.read_oam_sprite(sprite_index)
            if sprite_data is None:
                continue

            oam_y, _tile_id, _attributes, oam_x = sprite_data
            if oam_y >= SCREEN_HEIGHT:  # Off-screen sentinel
                continue

            # OAM stores Y minus one
            sprite_top = oam_y + 1

            # Coarse window: anything that could plausibly belong to this object
            if abs(oam_x - screen_x) > OBJECT_SPRITE_MAX_WIDTH:
                continue
            if not -TILE_SIZE < (sprite_top - screen_y) < OBJECT_SPRITE_MAX_HEIGHT:
                continue

            # A sprite drawn nearer another object belongs to that object. Without
            # this, two objects close together chain into a single cluster and both
            # report the combined extent -- a Cobrat in its jar beside the bullet it
            # spat made an 8px bullet measure four tiles tall.
            if other_anchors and _is_nearer_other_object(
                oam_x, sprite_top, screen_x, screen_y, other_anchors
            ):
                continue

            candidates.append((oam_x, sprite_top))

        if not candidates:
            return 1, 1

        # An object's anchor is its top-left, so its own sprites start at the
        # anchor row and continue downward. Sprites above that row belong to
        # something else -- an object stacked overhead, whose sprites would
        # otherwise chain down into this cluster and inflate it. The tolerance is
        # deliberately under one sprite row: it absorbs the few pixels of
        # anchor/draw jitter without reaching the row above.
        top_limit = screen_y - OAM_SPRITE_HEIGHT // 2
        candidates = [c for c in candidates if c[1] >= top_limit]
        if not candidates:
            return 1, 1

        # Grow the cluster outward from the anchor, keeping only sprites that
        # actually touch what is already in it. A fixed window is not enough: a
        # nearby object's sprites fall inside it and silently inflate the size,
        # which is how a 1-tile-wide Hawkmouth measured 3 tiles wide.
        cluster = [(screen_x, screen_y)]
        remaining = list(candidates)
        grew = True
        while grew:
            grew = False
            for candidate in list(remaining):
                if any(
                    abs(candidate[0] - member[0]) <= OAM_SPRITE_WIDTH
                    and abs(candidate[1] - member[1]) <= OAM_SPRITE_HEIGHT
                    for member in cluster
                ):
                    cluster.append(candidate)
                    remaining.remove(candidate)
                    grew = True

        # OAM sprites are 8px wide and, in 8x16 mode, 16px tall
        xs = [x for x, _ in cluster]
        ys = [y for _, y in cluster]
        width_px = (max(xs) + OAM_SPRITE_WIDTH) - min(xs)
        height_px = (max(ys) + OAM_SPRITE_HEIGHT) - min(ys)

        width = max(1, round(width_px / TILE_SIZE))
        height = max(1, round(height_px / TILE_SIZE))

        # Guard against a runaway cluster swallowing nearby objects
        return min(width, OBJECT_MAX_TILES), min(height, OBJECT_MAX_TILES)

    def _build_object_maps(self) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Build the sprite-object layers: which object occupies each cell.

        The sprite slots hold every dynamic object, not just enemies, so each is
        classified by its object ID -- a subspace door reads as DOOR, a coin as
        COIN, and only genuinely hostile objects as ENEMY. Unrecognised ids fall
        back to ENEMY, since treating an unknown hazard as harmless is the more
        dangerous mistake.

        Objects fill their whole measured footprint, and are kept separate from
        the terrain rather than overwriting it: a door standing on solid ground
        must read as both, or there is no way to tell standing on it from being
        inside it.

        Returns:
            tuple of (object_id_map, object_fine_type_map, property_map, velocity_map).
            property_map is (15, 16, 2) uint8 holding the DAMAGES and LIFTABLE
            masks; velocity_map is (15, 16, 2) float32 holding normalised X and Y
            velocity.
        """
        shape = (SCREEN_TILES_HEIGHT, SCREEN_TILES_WIDTH)
        object_id_map = np.full(shape, NO_OBJECT, dtype=np.uint8)
        object_type_map = np.zeros(shape, dtype=np.uint8)
        property_map = np.zeros((*shape, 2), dtype=np.uint8)
        velocity_map = np.zeros((*shape, 2), dtype=np.float32)

        visible = self._get_visible_objects()

        # Two objects can land in the same cell -- a Cobrat and the bullet it just
        # spat sit 6px apart, well inside one 16px tile. Only one survives, so
        # write the small transient types first and let the substantial object
        # overwrite them: an agent needs to see the Cobrat more than its bullet.
        visible = sorted(
            visible,
            key=lambda item: (
                OBJECT_ID_MAPPING.get(item[2].object_type, UNMAPPED_OBJECT_FINE_TYPE)
                not in SINGLE_TILE_FINE_TYPES
            ),
        )

        for index, (x_pixel, y_pixel, enemy) in enumerate(visible):
            # Every other on-screen object, so sprite ownership can be resolved
            other_anchors = [(ox, oy) for j, (ox, oy, _) in enumerate(visible) if j != index]
            object_id = enemy.object_type
            fine_type = OBJECT_ID_MAPPING.get(object_id, UNMAPPED_OBJECT_FINE_TYPE)

            # Pseudo-objects (attack triggers, spawner control) draw nothing
            if fine_type == FineTileType.EMPTY:
                continue

            damages = fine_type in DAMAGING_FINE_TYPES
            # The runtime UNLIFTABLE flag overrides the type-based guess: some
            # otherwise-liftable objects are pinned in place.
            flags = enemy.sprite_flags or 0
            liftable = fine_type in LIFTABLE_FINE_TYPES and not flags & SpriteFlags.UNLIFTABLE

            velocity_x = np.clip((enemy.x_velocity or 0) / OBJECT_VELOCITY_SCALE, -1.0, 1.0)
            # Object Y velocity is in raw screen space (positive = downward). The
            # map's rows also increase downward, so no inversion is needed here.
            velocity_y = np.clip((enemy.y_velocity or 0) / OBJECT_VELOCITY_SCALE, -1.0, 1.0)

            if fine_type in SINGLE_TILE_FINE_TYPES:
                # Small by definition; measuring would inherit a neighbour's block
                width, height = 1, 1
            else:
                width, height = self._measure_object_size(x_pixel, y_pixel, other_anchors)

            # The X is a left edge, centred the same way the player's is -- a
            # carried item shares its carrier's X and must land in its column.
            # The anchor is the object's top-left, so the footprint extends right
            # and down from there.
            left_tile = (x_pixel + (TILE_SIZE // 2)) // TILE_SIZE
            top_tile = y_pixel // TILE_SIZE

            for row in range(top_tile, top_tile + height):
                if not 0 <= row < SCREEN_TILES_HEIGHT:
                    continue
                for column in range(left_tile, left_tile + width):
                    if not 0 <= column < SCREEN_TILES_WIDTH:
                        continue
                    object_id_map[row, column] = object_id
                    object_type_map[row, column] = fine_type
                    property_map[row, column, 0] = damages
                    property_map[row, column, 1] = liftable
                    velocity_map[row, column, 0] = velocity_x
                    velocity_map[row, column, 1] = velocity_y

        return object_id_map, object_type_map, property_map, velocity_map

    # ---- Viewport --------------------------------------------------

    def _get_viewport_offset(self) -> tuple[int, int]:
        """Get viewport offset from screen boundary registers and PPU scroll.

        The game stores the camera/viewport position using:
        - ScreenBoundaryLeftHi/Lo: Page-aligned horizontal boundary
        - ScreenYHi/Lo: Vertical position in world coordinates
        - PPU scroll registers: Fine scrolling offsets within the current nametable

        For horizontal scrolling, we need to add the PPU scroll offset to get
        smooth sub-page scrolling. For vertical scrolling, ScreenYHi/Lo already
        includes the fine offset.

        Returns:
            tuple of (viewport_x_offset, viewport_y_offset) in tiles
        """
        # Read the camera base position
        viewport_x_hi = self._read_ram_safe(VIEWPORT.SCREEN_BOUNDARY_LEFT_HI)
        viewport_x_lo = self._read_ram_safe(VIEWPORT.SCREEN_BOUNDARY_LEFT_LO)
        viewport_y_hi = self._read_ram_safe(VIEWPORT.SCREEN_Y_HI)
        viewport_y_lo = self._read_ram_safe(VIEWPORT.SCREEN_Y_LO)

        # Read PPU scroll positions for fine scrolling. Only the X mirror is
        # used: in vertical levels ScreenY already carries the fine offset.
        scroll_x = self._read_ram_safe(VIEWPORT.PPU_SCROLL_X_MIRROR)

        # Check scroll direction to determine scrolling type
        scroll_direction = self._read_ram_safe(GAME_STATE.SCROLL_DIRECTION)

        # Combine boundary and scroll positions
        # For horizontal scrolling levels (scroll_direction != 0x00):
        #   - ScreenBoundaryLeft tracks the base page
        #   - PPU scroll gives fine offset within the page
        # For vertical scrolling levels (scroll_direction == 0x00):
        #   - ScreenY already includes fine positioning
        if scroll_direction == 0x00:
            # Vertical scrolling: ScreenY is complete, but may need PPU scroll for smoothness
            viewport_x_pixels = (viewport_x_hi * PAGE_SIZE) + viewport_x_lo
            viewport_y_pixels = (viewport_y_hi * PAGE_SIZE) + viewport_y_lo
        else:
            # Horizontal scrolling: add PPU scroll offset for smooth scrolling
            viewport_x_pixels = (viewport_x_hi * PAGE_SIZE) + viewport_x_lo + scroll_x
            viewport_y_pixels = (viewport_y_hi * PAGE_SIZE) + viewport_y_lo

        # Convert to tiles
        viewport_x = viewport_x_pixels // TILE_SIZE
        viewport_y = viewport_y_pixels // TILE_SIZE

        return viewport_x, viewport_y

    # ---- Full semantic map -----------------------------------------

    def _read_tile_maps(self) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Read tile IDs and types from SRAM.

        In normal gameplay, reads from standard SRAM level data (0x0000-0x095F).
        In subspace, reads from dedicated subspace SRAM region (0x0F00-0x0FE0).

        Returns:
            tuple of (tile_id_map, tile_type_map) - both 15x16 uint8 arrays
        """
        # Initialize maps (height x width)
        tile_id_map = np.zeros((SCREEN_TILES_HEIGHT, SCREEN_TILES_WIDTH), dtype=np.uint8)
        tile_type_map = np.zeros((SCREEN_TILES_HEIGHT, SCREEN_TILES_WIDTH), dtype=np.uint8)

        # Check if in subspace (subspace_status == 2 means in subspace)
        subspace_status = self._read_ram_safe(GAME_STATE.SUBSPACE_STATUS)
        in_subspace = subspace_status == 2

        if in_subspace:
            # Subspace: read from dedicated subspace RAM region
            # 0x0700 - 0x07FF (256 bytes) contains the subspace tile layout
            # When entering subspace, the current screen is stored here (possibly reversed)
            SUBSPACE_RAM_START = 0x0700
            for y in range(SCREEN_TILES_HEIGHT):
                for x in range(SCREEN_TILES_WIDTH):
                    ram_address = SUBSPACE_RAM_START + (y * SCREEN_TILES_WIDTH + x)
                    tile_id = self._read_ram_safe(ram_address)
                    tile_id_map[y, x] = tile_id

                    # Map tile ID to type
                    if tile_id not in TILE_ID_MAPPING:
                        warnings.warn(
                            f"Unknown tile ID {tile_id} at subspace position ({x}, {y}), "
                            f"RAM address 0x{ram_address:04X}. Treating as EMPTY tile.",
                            RuntimeWarning,
                            stacklevel=3,
                        )
                        tile_type_map[y, x] = FineTileType.EMPTY
                    else:
                        tile_type_map[y, x] = TILE_ID_MAPPING[tile_id]
            return tile_id_map, tile_type_map

        # Normal gameplay: read from standard SRAM
        # Get viewport offset
        viewport_x, viewport_y = self._get_viewport_offset()

        # Check scroll direction (0x00=horizontal, 0x01=vertical)
        scroll_direction = self._read_ram_safe(GAME_STATE.SCROLL_DIRECTION)

        # SRAM contains level data
        MAX_SRAM_SIZE = 0x960  # 2400 bytes

        # Read tile data using read_sram (one byte at a time)
        for y in range(SCREEN_TILES_HEIGHT):
            for x in range(SCREEN_TILES_WIDTH):
                # Calculate the world position of this tile in the viewport
                world_x = viewport_x + x
                world_y = viewport_y + y

                # Convert world position back to SRAM address
                page_x = world_x // LEVEL_PAGE_WIDTH
                page_y = world_y // LEVEL_PAGE_HEIGHT
                tile_x_in_page = world_x % LEVEL_PAGE_WIDTH
                tile_y_in_page = world_y % LEVEL_PAGE_HEIGHT

                # Calculate which page this tile belongs to
                # NOTE: scroll_direction may be inverted from what's documented?
                if scroll_direction == 0x00:
                    # Use Y page (vertical scrolling)
                    page_number = page_y
                else:
                    # Use X page (horizontal scrolling)
                    page_number = page_x

                # Calculate SRAM address
                # Each page is 16x15 = 240 bytes (LEVEL_PAGE_WIDTH * LEVEL_PAGE_HEIGHT)
                BYTES_PER_PAGE = LEVEL_PAGE_WIDTH * LEVEL_PAGE_HEIGHT
                tile_index_in_page = tile_y_in_page * LEVEL_PAGE_WIDTH + tile_x_in_page
                sram_address = (page_number * BYTES_PER_PAGE + tile_index_in_page) % MAX_SRAM_SIZE

                # Read the tile ID from SRAM
                tile_id = self._nes.read_sram(sram_address)
                tile_id_map[y, x] = tile_id

                # Map tile ID to tile type with fallback for unknown tiles
                if tile_id not in TILE_ID_MAPPING:
                    warnings.warn(
                        f"Unknown tile ID {tile_id} at screen position ({x}, {y}), "
                        f"world position ({world_x}, {world_y}), "
                        f"SRAM address 0x{sram_address:04X}, "
                        f"page {page_number}. Treating as EMPTY tile.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    tile_type_map[y, x] = FineTileType.EMPTY
                else:
                    tile_type_map[y, x] = TILE_ID_MAPPING[tile_id]

        return tile_id_map, tile_type_map

    @property
    def semantic_map(self) -> NDArray[Any]:
        """Get full semantic map with hierarchical tile information.

        Terrain and sprite objects occupy separate fields, so a cell can report
        both -- a door standing on solid ground reads as SOLID terrain with a
        DOOR object, which is what distinguishes standing on it from being inside
        it. The terrain fields are never overwritten by a sprite.

        Returns a structured numpy array with complete tile information:
        - tile_id: Raw BackgroundTile ID
        - fine_type / coarse_type: the TERRAIN in this cell
        - object_id: EnemyId of the sprite here, or NO_OBJECT (0xFF)
        - object_fine_type / object_coarse_type: that sprite's classification
        - color_r, color_g, color_b: RGB colour, object over terrain

        Returns:
            2D structured numpy array (15 x 16) with SEMANTIC_TILE_DTYPE (height x width).
        """
        # Read terrain from SRAM, and sprite objects from the object slots
        tile_id_map, fine_type_map = self._read_tile_maps()
        object_id_map, object_type_map, property_map, velocity_map = self._build_object_maps()

        # Create structured array
        semantic_map = np.zeros(
            (SCREEN_TILES_HEIGHT, SCREEN_TILES_WIDTH), dtype=SEMANTIC_TILE_DTYPE
        )

        # Populate using vectorized operations
        semantic_map['tile_id'] = tile_id_map
        semantic_map['fine_type'] = fine_type_map
        semantic_map['coarse_type'] = COARSE_LOOKUP[fine_type_map]
        semantic_map['object_id'] = object_id_map
        semantic_map['object_fine_type'] = object_type_map
        semantic_map['object_coarse_type'] = COARSE_LOOKUP[object_type_map]
        semantic_map['object_damages'] = property_map[:, :, 0]
        semantic_map['object_liftable'] = property_map[:, :, 1]
        semantic_map['object_velocity_x'] = velocity_map[:, :, 0]
        semantic_map['object_velocity_y'] = velocity_map[:, :, 1]

        # Colour shows the object where there is one, terrain otherwise, matching
        # the render priority: objects draw over the terrain they stand on.
        has_object = object_id_map != NO_OBJECT
        colour_source = np.where(has_object, object_type_map, fine_type_map)
        semantic_map['color_r'] = COLOR_LOOKUP[colour_source, 0]
        semantic_map['color_g'] = COLOR_LOOKUP[colour_source, 1]
        semantic_map['color_b'] = COLOR_LOOKUP[colour_source, 2]

        return semantic_map

    @property
    def semantic_tensor(self) -> NDArray[np.uint8]:
        """Get the semantic map as a binary (H, W, C) tensor for learning.

        Channels are binary masks rather than category ids: ids are nominal
        labels, and feeding them as numbers would imply ENEMY (15) is "more" than
        SOLID (1). Terrain and object channels form two groups -- within a group
        the encoding is one-hot, across groups it is multi-hot -- so a door on
        solid ground sets both its TERRAIN and INTERACTIVE channels.

        Channel order is COARSE_TENSOR_CHANNELS. Layout is channels-last to match
        the RGB frame observation; PyTorch users want a single permute.

        Returns:
            (15, 16, 16) uint8 array of 0/1 values.
        """
        return self.semantic_tensor_from(self.semantic_map)

    @staticmethod
    def semantic_tensor_from(semantic_map: NDArray[Any]) -> NDArray[np.uint8]:
        """Build the binary tensor from an existing semantic map.

        Exposed separately so callers that already hold a semantic map do not pay
        to read every tile from SRAM a second time.

        Args:
            semantic_map: Structured array with SEMANTIC_TILE_DTYPE

        Returns:
            (15, 16, 16) uint8 array of 0/1 values.
        """
        terrain = semantic_map['coarse_type']
        objects = semantic_map['object_coarse_type']
        has_object = semantic_map['object_id'] != NO_OBJECT

        tensor = np.zeros(
            (
                semantic_map.shape[0],
                semantic_map.shape[1],
                len(COARSE_TENSOR_CHANNELS) + len(PROPERTY_CHANNEL_NAMES),
            ),
            dtype=np.uint8,
        )

        for channel, (layer, coarse_type) in enumerate(COARSE_TENSOR_CHANNELS):
            if layer == 'terrain':
                tensor[:, :, channel] = (terrain == coarse_type).astype(np.uint8)
            else:
                tensor[:, :, channel] = (has_object & (objects == coarse_type)).astype(np.uint8)

        # Property channels: what happens if the player touches this
        tensor[:, :, len(COARSE_TENSOR_CHANNELS)] = semantic_map['object_damages']
        tensor[:, :, len(COARSE_TENSOR_CHANNELS) + 1] = semantic_map['object_liftable']

        return tensor

    @property
    def semantic_velocity(self) -> NDArray[np.float32]:
        """Get per-cell object velocity as a (H, W, 2) float array.

        Kept separate from `semantic_tensor` so that tensor stays a pure binary
        mask: velocity is continuous, and mixing the two would force every mask
        to float and lose an invariant worth keeping. Concatenate them if a single
        input array is wanted.

        Without this the map is a still frame -- an agent cannot tell an enemy
        moving towards it from one moving away.

        Returns:
            (15, 16, 2) float32 array, normalised to ~[-1, 1]. Positive X is
            rightward, positive Y is downward (matching row order).
        """
        return self.semantic_velocity_from(self.semantic_map)

    @staticmethod
    def semantic_velocity_from(semantic_map: NDArray[Any]) -> NDArray[np.float32]:
        """Build the velocity grid from an existing semantic map.

        Args:
            semantic_map: Structured array with SEMANTIC_TILE_DTYPE

        Returns:
            (15, 16, 2) float32 array, normalised to ~[-1, 1].
        """
        return np.stack(
            [semantic_map['object_velocity_x'], semantic_map['object_velocity_y']],
            axis=-1,
        ).astype(np.float32)
