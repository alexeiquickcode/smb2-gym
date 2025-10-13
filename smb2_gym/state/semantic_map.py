"""Semantic tile map for SMB2 environment."""

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tetanes_py import NesEnv

from ..constants import (
    GAME_STATE,
    LEVEL_PAGE_HEIGHT,
    LEVEL_PAGE_WIDTH,
    PAGE_SIZE,
    PLAYER,
    SCREEN_HEIGHT,
    SCREEN_TILES_HEIGHT,
    SCREEN_TILES_WIDTH,
    TILE_SIZE,
    Enemy,
    EnemyState,
)
from ..constants.semantic import (
    COARSE_LOOKUP,
    COLOR_LOOKUP,
    SEMANTIC_TILE_DTYPE,
    TILE_ID_MAPPING,
    FineTileType,
)
from ._base import GameStateMixin


class SemanticMapMixin(GameStateMixin):
    """Mixin providing semantic tile map for SMB2 environment.

    This mixin provides access to semantic tile information including:
    - Tile types (SOLID, ENEMY, COLLECTIBLE, etc.)
    - Tile categories (TERRAIN, HAZARD, etc.)
    - Player position on the tile grid
    - Enemy positions overlaid on the map

    Note: This mixin depends on the `enemies` property being provided by EnemiesMixin.
    """

    _nes: NesEnv  # Parent class for type checking
    enemies: list[Enemy]  # Provided by EnemiesMixin for type checking

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

    # ---- Player State ----------------------------------------------

    def is_player_ducking(self) -> bool:  # TODO: This doesn't change the hitbox?? Need to review
        """Check if the player is currently ducking/crouching.

        Returns:
            True if player is ducking, False otherwise
        """
        # Check if player has 2+ hearts (is big)
        life_meter = self._read_ram_safe(PLAYER.LIFE_METER, default=0x0F)
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

    def get_player_collision_tiles(self) -> list[tuple[int, int]]:
        """Get the tile positions occupied by the player for collision detection.

        Returns:
            List of (screen_x_tile, screen_y_tile) tuples representing tiles occupied by player.
            Returns empty list if player position not found.
        """
        # Get sprites
        oam_sprites = self._read_player_sprites()
        if not oam_sprites:
            return []

        # Get sprite bounds
        min_x = min(sprite[3] for sprite in oam_sprites)
        _max_x = max(sprite[3] for sprite in oam_sprites)
        min_y = min(sprite[0] for sprite in oam_sprites)
        max_y = max(sprite[0] for sprite in oam_sprites)

        # Calculate which tiles are occupied
        tiles: list[tuple[int, int]] = []

        # Get X tile (same for all cases)
        x_tile = min_x // TILE_SIZE

        # Check if player has 2+ hearts (is big)
        life_meter = self._read_ram_safe(PLAYER.LIFE_METER, default=0x0F)
        num_hearts = (life_meter >> 4) + 1
        is_big = num_hearts >= 2

        # Check if ducking
        is_ducking = self.is_player_ducking()

        if is_big and not is_ducking:
            # Big player standing: occupies 2 vertical tiles
            # Use min and max Y to determine tiles, add 16 pixels to compensate for viewport shift
            top_tile = (min_y + 16) // TILE_SIZE
            bottom_tile = (max_y + 16) // TILE_SIZE
            tiles.append((x_tile, top_tile))
            if bottom_tile != top_tile:
                tiles.append((x_tile, bottom_tile))
        else:
            # Small player or ducking big player: occupies 1 tile
            # Use min_y (top of sprite) and add 16 pixels to compensate for viewport shift
            y_tile = (min_y + 16) // TILE_SIZE
            tiles.append((x_tile, y_tile))

        return tiles

    # ---- Enemy Positions (RAM-based) ------------------------------

    def _get_enemy_screen_positions(self) -> list[tuple[int, int, int]]:
        """Get enemy positions on screen from RAM.

        Reads enemy positions from RAM addresses and converts them to screen coordinates.
        Only returns visible enemies that are on screen.

        Returns:
            List of (x_pixel, y_pixel, enemy_id) tuples for visible enemies on screen.
        """

        # Get viewport offset to convert world coordinates to screen coordinates
        viewport_x, viewport_y = self._get_viewport_offset()
        viewport_x_pixels = viewport_x * TILE_SIZE
        viewport_y_pixels = viewport_y * TILE_SIZE

        enemy_positions: list[tuple[int, int, int]] = []
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

            # Convert to screen coordinates
            screen_x = world_x - viewport_x_pixels
            screen_y = world_y - viewport_y_pixels

            # Only include enemies that are on screen
            if 0 <= screen_x < PAGE_SIZE and 0 <= screen_y < SCREEN_TILES_HEIGHT * TILE_SIZE:
                enemy_positions.append((screen_x, screen_y, enemy.object_type))

        return enemy_positions

    def _add_enemies_to_map(self, collision_map: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Add enemy positions to the collision map using RAM data.

        Reads enemy positions from RAM and marks their occupied tiles.
        Only includes visible enemies that are on screen.

        Args:
            collision_map: Base collision map to overlay enemies on

        Returns:
            Updated collision map with enemy positions
        """
        # Get all visible enemy positions from RAM
        enemy_positions = self._get_enemy_screen_positions()

        # Mark each enemy position on the collision map
        for x_pixel, y_pixel, _enemy_id in enemy_positions:
            # Convert pixel position to tile position
            tile_x = x_pixel // TILE_SIZE
            tile_y = y_pixel // TILE_SIZE

            # Check bounds and set enemy tile
            if 0 <= tile_x < SCREEN_TILES_WIDTH and 0 <= tile_y < SCREEN_TILES_HEIGHT:
                collision_map[tile_y, tile_x] = FineTileType.ENEMY

        return collision_map

    # ---- Viewport --------------------------------------------------

    def _get_viewport_offset(self) -> tuple[int, int]:
        """Get viewport offset by calculating from player's world position and screen position.

        Returns:
            tuple of (viewport_x_offset, viewport_y_offset) in tiles
        """
        # Get player's world position from RAM (in pixels)
        player_x_page = self._read_ram_safe(PLAYER.X_PAGE, default=0)
        player_x_position = self._read_ram_safe(PLAYER.X_POSITION, default=0)
        player_y_page = self._read_ram_safe(PLAYER.Y_PAGE, default=0)
        player_y_position = self._read_ram_safe(PLAYER.Y_POSITION, default=0)

        # Calculate player's world position in pixels
        player_world_x_pixels = (player_x_page * 256) + player_x_position
        player_world_y_pixels = (player_y_page * 256) + player_y_position

        # Get player's screen position in pixels from sprite
        sprite_pos = self.get_player_sprite_position()

        if sprite_pos is None:
            # FALLBACK: assume player is centered on screen
            player_screen_x_pixels = 128
            player_screen_y_pixels = 120
        else:
            player_screen_x_pixels, player_screen_y_pixels = sprite_pos

        # Calculate viewport offset in pixels: world_pos - screen_pos = viewport_offset
        # Add 8 pixel offset to align sprite center with tile grid
        viewport_x_pixels = player_world_x_pixels - player_screen_x_pixels + 8
        viewport_y_pixels = player_world_y_pixels - player_screen_y_pixels + 16

        # Convert to tiles and shift viewport down by 1 tile
        viewport_x = max(0, viewport_x_pixels // TILE_SIZE)
        viewport_y = max(0, (viewport_y_pixels // TILE_SIZE) - 1)

        return viewport_x, viewport_y

    # ---- Full semantic map -----------------------------------------

    def _read_tile_maps(self) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Read tile IDs and types from SRAM.

        Returns:
            tuple of (tile_id_map, tile_type_map) - both 15x16 uint8 arrays
        """
        # Initialize maps (height x width)
        tile_id_map = np.zeros((SCREEN_TILES_HEIGHT, SCREEN_TILES_WIDTH), dtype=np.uint8)
        tile_type_map = np.zeros((SCREEN_TILES_HEIGHT, SCREEN_TILES_WIDTH), dtype=np.uint8)

        # Get viewport offset
        viewport_x, viewport_y = self._get_viewport_offset()

        # Check scroll direction (0x00=horizontal, 0x01=vertical)
        scroll_direction = self._read_ram_safe(GAME_STATE.SCROLL_DIRECTION, default=0)

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
                        stacklevel=3
                    )
                    tile_type_map[y, x] = FineTileType.EMPTY
                else:
                    tile_type_map[y, x] = TILE_ID_MAPPING[tile_id]

        return tile_id_map, tile_type_map

    @property
    def semantic_map(self) -> NDArray[Any]:
        """Get full semantic map with hierarchical tile information.

        Returns a structured numpy array with complete tile information:
        - tile_id: Raw game object ID (BackgroundTile/EnemyId)
        - fine_type: Fine-grained FineTileType (SOLID, ENEMY, etc.)
        - coarse_type: Coarse-grained CoarseTileType (TERRAIN, ENEMY, etc.)
        - color_r, color_g, color_b: RGB visualization color

        Returns:
            2D structured numpy array (15 x 16) with SEMANTIC_TILE_DTYPE (height x width).
        """
        # Read tile maps from SRAM
        tile_id_map, fine_type_map = self._read_tile_maps()

        # Add enemy sprites from RAM (modifies fine_type_map)
        fine_type_map = self._add_enemies_to_map(fine_type_map)

        # Create structured array
        semantic_map = np.zeros(
            (SCREEN_TILES_HEIGHT, SCREEN_TILES_WIDTH), dtype=SEMANTIC_TILE_DTYPE
        )

        # Populate using vectorized operations
        semantic_map['tile_id'] = tile_id_map
        semantic_map['fine_type'] = fine_type_map
        semantic_map['coarse_type'] = COARSE_LOOKUP[fine_type_map]
        semantic_map['color_r'] = COLOR_LOOKUP[fine_type_map, 0]
        semantic_map['color_g'] = COLOR_LOOKUP[fine_type_map, 1]
        semantic_map['color_b'] = COLOR_LOOKUP[fine_type_map, 2]

        return semantic_map
