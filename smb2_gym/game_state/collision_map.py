"""Collision and interaction map for SMB2 environment."""

from enum import IntEnum
from typing import (
    Dict,
    Optional,
    Tuple,
)

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ..constants import (
    CURRENT_PAGE_POSITION,
    ENEMY_ID,
    ENEMY_NOT_PRESENT,
    ENEMY_VISIBILITY,
    ENEMY_VISIBLE,
    ENEMY_X_POS,
    ENEMY_Y_POS,
    LEVEL_PAGE_HEIGHT,
    LEVEL_PAGE_WIDTH,
    PAGE_SIZE,
    PLAYER_STATE,
    PLAYER_X_PAGE,
    PLAYER_X_POSITION,
    PLAYER_Y_PAGE,
    PLAYER_Y_POSITION,
    SCREEN_TILES_HEIGHT,
    SCREEN_TILES_WIDTH,
)
from .base import GameStateMixin


class PlayerState(IntEnum):
    """Player state constants from SMB2."""
    NORMAL = 0x00
    CLIMBING = 0x01
    LIFTING = 0x02
    CLIMBING_AREA_TRANSITION = 0x03
    GOING_DOWN_JAR = 0x04
    EXITING_JAR = 0x05
    HAWKMOUTH_EATING = 0x06
    DYING = 0x07
    CHANGING_SIZE = 0x08


class TileType(IntEnum):
    """Tile types for collision/interaction map."""
    EMPTY = 0  # Air, can move through
    SOLID = 1  # Solid ground/wall, cannot pass
    PLATFORM = 2  # Semi-solid platform (can jump through from below)
    VINE = 3  # Climbable vine
    DOOR = 4  # Door (can enter)
    JAR = 5  # Jar/pipe (can enter)
    SPIKES = 6  # Damaging spikes
    QUICKSAND = 7  # Quicksand (sinks, slower movement)
    CONVEYOR_LEFT = 8  # Conveyor belt moving left
    CONVEYOR_RIGHT = 9  # Conveyor belt moving right
    LADDER = 10  # Climbable ladder
    CHAIN = 11  # Climbable chain
    CHERRY = 12  # Collectible cherry
    VEGETABLE = 13  # Pullable vegetable
    POTION = 14  # Sub-space potion
    MUSHROOM = 15  # Size mushroom
    HEART = 16  # Health pickup
    STAR = 17  # Invincibility star
    ENEMY = 18  # Enemy occupying tile
    BOSS = 19  # Boss enemy
    PROJECTILE = 20  # Enemy projectile
    WATER = 21  # Water (different physics)
    ICE = 22  # Ice (slippery)
    CLOUD = 23  # Cloud platform
    CARPET = 24  # Flying carpet
    HAWK_MOUTH = 25  # Hawk mouth entrance
    CRYSTAL = 26  # End-level crystal
    MASK = 27  # Mask gate/boss gate
    POW_BLOCK = 28  # POW block
    BOMB = 29  # Bomb
    KEY = 30  # Key


class CollisionMapMixin(GameStateMixin):
    """Mixin providing collision and interaction map for SMB2 environment."""

    def __init__(self, *args, **kwargs):
        """Initialize mixin."""
        super().__init__(*args, **kwargs)

    # Mapping from SMB2 tile IDs to our TileType enum
    # Based on actual SRAM values observed
    TILE_ID_MAPPING: Dict[int, TileType] = {
        # Empty/Sky tiles
        0x00: TileType.EMPTY,  # Air/empty space

        # Solid ground/wall tiles (confirmed from SRAM observation)
        0xCA: TileType.SOLID,  # Solid block at $6090
        0xCC: TileType.SOLID,  # Solid block at $6091
        0xCE: TileType.SOLID,  # Solid block at $6092

        # Door tiles (confirmed from SRAM observation)
        0x4F: TileType.DOOR,  # Door tile at $60B1
        0x51: TileType.DOOR,  # Door tile at $60C1

        # Vine tile (confirmed from Data Crystal)
        0xC2: TileType.VINE,  # Vine (Color #4)

        # Additional common solid tiles (need verification)
        0xC8: TileType.SOLID,  # Likely solid
        0xC9: TileType.SOLID,  # Likely solid
        0xCB: TileType.SOLID,  # Likely solid
        0xCD: TileType.SOLID,  # Likely solid
        0xCF: TileType.SOLID,  # Likely solid
        0xD0: TileType.SOLID,  # Likely solid
        0xD1: TileType.SOLID,  # Likely solid
        0xD2: TileType.SOLID,  # Likely solid

        # Platform tiles (need verification)
        0x10: TileType.PLATFORM,  # Jump-through platform
        0x11: TileType.PLATFORM,  # Jump-through platform variant
        0x12: TileType.PLATFORM,  # Jump-through platform variant

        # Special tiles (need verification)
        0x80: TileType.LADDER,  # Potential ladder
        0x81: TileType.LADDER,  # Potential ladder variant

        # Will expand this mapping as we discover more tile values
    }

    def _read_oam_sprite_safe(self, sprite_index: int) -> Optional[Tuple[int, int, int, int]]:
        """Safely read a sprite from OAM RAM.

        Args:
            sprite_index: Sprite index (0-63)

        Returns:
            Tuple of (y_pos, tile_id, attributes, x_pos) or None if read fails
        """
        try:
            return self._nes.read_oam_sprite(sprite_index)
        except:
            return None

    def _read_player_sprites(self) -> list:
        """Read player sprite data from OAM.

        Player character sprites are always at fixed OAM indices 7-10.
        Returns list of sprite entries for the player, each with (y, tile_id, attributes, x).
        """
        oam_data = []

        # Player sprites are at fixed indices 7-10 (4 sprites for big character, subset for small)
        for sprite_index in range(7, 11):
            sprite_data = self._read_oam_sprite_safe(sprite_index)

            if sprite_data is None:
                continue

            y_pos, tile_id, attributes, x_pos = sprite_data

            # Skip empty/offscreen sprites (Y >= 0xEF means hidden)
            if y_pos < 0xEF and tile_id != 0xFF:
                oam_data.append((y_pos, tile_id, attributes, x_pos))

        return oam_data

    def get_player_sprite_position(self) -> Optional[Tuple[int, int]]:
        """Get the player character sprite position from OAM.

        Player sprites are always at fixed OAM indices 7-10.
        Returns the top-left position of the player sprite.

        Returns:
            Tuple of (x_pixel, y_pixel) on screen, or None if not found
        """
        # Read player sprite data (fixed indices 7-10)
        oam_sprites = self._read_player_sprites()

        if not oam_sprites:
            return None

        # Get the top-left position from all player sprites
        # (minimum X and Y coordinates)
        min_x = min(sprite[3] for sprite in oam_sprites)  # x_pos is index 3
        min_y = min(sprite[0] for sprite in oam_sprites)  # y_pos is index 0

        return (min_x, min_y)

    @property
    def player_state(self) -> PlayerState:
        """Get current player state."""
        state = self._read_ram_safe(PLAYER_STATE, default=0)
        try:
            return PlayerState(state)
        except ValueError:
            return PlayerState.NORMAL

    def _get_viewport_offset(self) -> Tuple[int, int]:
        """Get viewport offset by calculating from player's world position and screen position.

        Returns:
            Tuple of (viewport_x_offset, viewport_y_offset) in tiles
        """
        # Get player's world position from RAM (in pixels)
        player_x_page = self._read_ram_safe(PLAYER_X_PAGE, default=0)
        player_x_position = self._read_ram_safe(PLAYER_X_POSITION, default=0)
        player_y_page = self._read_ram_safe(PLAYER_Y_PAGE, default=0)
        player_y_position = self._read_ram_safe(PLAYER_Y_POSITION, default=0)

        # Calculate player's world position in pixels
        player_world_x_pixels = (player_x_page * 256) + player_x_position
        player_world_y_pixels = (player_y_page * 256) + player_y_position

        # Get player's screen position in pixels from sprite
        sprite_pos = self.get_player_sprite_position()

        if sprite_pos is None:
            # Fallback: assume player is centered on screen
            player_screen_x_pixels = 128
            player_screen_y_pixels = 120
        else:
            player_screen_x_pixels, player_screen_y_pixels = sprite_pos

        # Calculate viewport offset in pixels: world_pos - screen_pos = viewport_offset
        viewport_x_pixels = player_world_x_pixels - player_screen_x_pixels
        viewport_y_pixels = player_world_y_pixels - player_screen_y_pixels

        # Convert to tiles
        viewport_x = max(0, viewport_x_pixels // 16)
        viewport_y = max(0, viewport_y_pixels // 16)

        return viewport_x, viewport_y

    def get_player_screen_position(self) -> Optional[Tuple[int, int]]:
        """Get player's position on screen in tiles (0-15 for both X and Y).

        Returns:
            Tuple of (screen_x_tile, screen_y_tile) or None if not found
        """
        # Get sprite position in pixels
        sprite_pos = self.get_player_sprite_position()

        if sprite_pos is None:
            return None

        x_pixels, y_pixels = sprite_pos

        # Convert to tiles
        screen_x_tile = x_pixels // 16
        screen_y_tile = y_pixels // 16

        # Clamp to screen bounds (0-15)
        screen_x_tile = max(0, min(screen_x_tile, 15))
        screen_y_tile = max(0, min(screen_y_tile, 15))

        return screen_x_tile, screen_y_tile

    @property
    def collision_map(self) -> np.ndarray:
        """Get collision/interaction map from SRAM.

        Returns a 16x16 tile grid (256 tiles total) based on current viewport.

        Returns:
            2D numpy array (16 x 16) with TileType values.
        """
        # Initialize 16x16 map
        COLLISION_MAP_SIZE = 16
        collision_map = np.zeros((COLLISION_MAP_SIZE, COLLISION_MAP_SIZE), dtype=np.uint8)

        # Get viewport offset
        viewport_x, viewport_y = self._get_viewport_offset()

        # SRAM contains level data, but we need to figure out how it's organized
        # For now, assume it's a large 2D array and we're extracting a 16x16 window
        MAX_SRAM_SIZE = 0x960  # 2400 bytes

        # Estimate level dimensions based on typical SMB2 levels
        # This may need adjustment based on actual level format
        ESTIMATED_LEVEL_WIDTH = 160  # tiles (10 pages * 16 tiles)
        ESTIMATED_LEVEL_HEIGHT = 15  # tiles

        # Read tile data using read_sram (one byte at a time)
        for y in range(COLLISION_MAP_SIZE):
            for x in range(COLLISION_MAP_SIZE):
                # Calculate the world position of this tile in the viewport
                world_x = viewport_x + x
                world_y = viewport_y + y

                # Convert world position back to SRAM address
                # Assume level data is stored as pages (16x16 tiles each)
                page_x = world_x // 16
                page_y = world_y // 16
                tile_x_in_page = world_x % 16
                tile_y_in_page = world_y % 16

                # Calculate which page this tile belongs to
                # Assuming pages are stored sequentially in SRAM
                page_number = page_x  # For now, assume horizontal paging only

                # Calculate SRAM address
                # Each page is 16x15 = 240 bytes, but we've been using 256 for 16x16
                # Let's stick with 256 for now
                BYTES_PER_PAGE = 256
                tile_index_in_page = tile_y_in_page * 16 + tile_x_in_page
                sram_address = (page_number * BYTES_PER_PAGE + tile_index_in_page) % MAX_SRAM_SIZE

                # Read the tile ID from SRAM
                try:
                    tile_id = self._nes.read_sram(sram_address)
                except:
                    tile_id = 0  # Default to empty if read fails

                # Map tile ID to TileType
                if tile_id in self.TILE_ID_MAPPING:
                    collision_map[y, x] = self.TILE_ID_MAPPING[tile_id]
                else:
                    # For unknown tiles, try to classify them
                    if tile_id == 0x00:
                        collision_map[y, x] = TileType.EMPTY
                    elif tile_id == 0xC2:
                        # Vine tile confirmed from Data Crystal
                        collision_map[y, x] = TileType.VINE
                    else:
                        # Default to empty for unknown tiles
                        # We'll expand the mapping as we discover more tile types
                        collision_map[y, x] = TileType.EMPTY

        return collision_map

    def _add_enemies_to_map(self, collision_map: np.ndarray) -> np.ndarray:
        """Add enemy positions to the collision map.

        Args:
            collision_map: Base collision map to overlay enemies on

        Returns:
            Updated collision map with enemy positions
        """
        for i in range(len(ENEMY_ID)):
            # Check if enemy is visible
            visibility = self._read_ram_safe(ENEMY_VISIBILITY[i], default=0)
            if visibility != ENEMY_VISIBLE:
                continue

            # Get enemy position
            enemy_x = self._read_ram_safe(ENEMY_X_POS[i], default=ENEMY_NOT_PRESENT)
            enemy_y = self._read_ram_safe(ENEMY_Y_POS[i], default=ENEMY_NOT_PRESENT)

            if enemy_x == ENEMY_NOT_PRESENT or enemy_y == ENEMY_NOT_PRESENT:
                continue

            # Convert pixel position to tile position
            tile_x = enemy_x // 16  # 16 pixels per tile
            tile_y = enemy_y // 16

            # Check bounds and set enemy tile
            if 0 <= tile_x < SCREEN_TILES_WIDTH and 0 <= tile_y < SCREEN_TILES_HEIGHT:
                collision_map[tile_y, tile_x] = TileType.ENEMY

        return collision_map

    def _add_sprites_to_map(self, collision_map: np.ndarray) -> np.ndarray:
        """Add sprite-based objects like vines and ladders to the collision map.

        In SMB2, vines and ladders might be implemented as sprites rather than
        background tiles. We can try to detect them by looking for sprite patterns
        or specific sprite IDs.

        Args:
            collision_map: Base collision map to overlay sprites on

        Returns:
            Updated collision map with sprite positions
        """
        # TODO: Implement sprite detection for vines/ladders
        # This would require:
        # 1. Reading sprite data from OAM (Object Attribute Memory)
        # 2. Identifying vine/ladder sprite patterns
        # 3. Converting sprite positions to tile positions
        # 4. For now, we return the map unchanged

        return collision_map

    def get_tile_at(self, x: int, y: int) -> TileType:
        """Get the tile type at a specific pixel position.

        Args:
            x: X pixel position
            y: Y pixel position

        Returns:
            TileType at the given position
        """
        tile_x = x // 16
        tile_y = y // 16

        collision_map = self.collision_map

        if 0 <= tile_x < SCREEN_TILES_WIDTH and 0 <= tile_y < SCREEN_TILES_HEIGHT:
            return TileType(collision_map[tile_y, tile_x])

        return TileType.EMPTY

    def is_solid(self, x: int, y: int) -> bool:
        """Check if a position contains a solid tile.

        Args:
            x: X pixel position
            y: Y pixel position

        Returns:
            True if the tile is solid, False otherwise
        """
        tile_type = self.get_tile_at(x, y)
        return tile_type == TileType.SOLID

    def is_climbable(self, x: int, y: int) -> bool:
        """Check if a position contains a climbable tile.

        Args:
            x: X pixel position
            y: Y pixel position

        Returns:
            True if the tile is climbable (ladder, vine, chain), False otherwise
        """
        tile_type = self.get_tile_at(x, y)
        return tile_type in [TileType.LADDER, TileType.VINE, TileType.CHAIN]

    def visualize_collision_map(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Visualize the collision map with colored tiles.

        Args:
            ax: Optional matplotlib axes to draw on

        Returns:
            The matplotlib axes with the visualization
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Get collision map
        collision_map = self.collision_map

        # Define colors for each tile type
        colors = {
            TileType.EMPTY: 'white',
            TileType.SOLID: 'brown',
            TileType.PLATFORM: 'tan',
            TileType.VINE: 'green',
            TileType.LADDER: 'gray',
            TileType.CHAIN: 'silver',
            TileType.ENEMY: 'red',
            TileType.DOOR: 'blue',
            TileType.JAR: 'purple',
            # Add more colors as needed
        }

        # Create color map
        color_array = np.zeros((*collision_map.shape, 3))
        for tile_type, color in colors.items():
            mask = collision_map == tile_type.value
            if np.any(mask):
                rgb = mcolors.to_rgb(color)
                color_array[mask] = rgb

        # Display the map
        ax.imshow(color_array, aspect='equal')
        ax.set_title('Collision Map')
        ax.set_xlabel('Tile X')
        ax.set_ylabel('Tile Y')

        # Add grid
        ax.set_xticks(np.arange(-0.5, SCREEN_TILES_WIDTH, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, SCREEN_TILES_HEIGHT, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

        # Add legend
        patches = []
        for tile_type, color in colors.items():
            if np.any(collision_map == tile_type.value):
                patches.append(mpatches.Patch(color=color, label=tile_type.name))

        if patches:
            ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.15, 1))

        return ax

    @property
    def interaction_matrix(self) -> np.ndarray:
        """Alias for collision_map for backward compatibility."""
        return self.collision_map

    @property
    def can_climb(self) -> bool:
        """Check if player is on a climbable tile at current position."""
        player_x = self._read_ram_safe(PLAYER_X_POSITION, default=0)
        player_y = self._read_ram_safe(PLAYER_Y_POSITION, default=0)

        # Player position is already in screen coordinates (0-255 for X, 0-239 for Y)
        # No coordinate inversion needed - both use Y=0 at top
        return self.is_climbable(player_x, player_y)

    @property
    def is_on_special_surface(self) -> bool:
        """Check if player is on any special surface (platform, ice, etc.)."""
        player_x = self._read_ram_safe(PLAYER_X_POSITION, default=0)
        player_y = self._read_ram_safe(PLAYER_Y_POSITION, default=0)

        # Player position is already in screen coordinates (0-255 for X, 0-239 for Y)
        # No coordinate inversion needed - both use Y=0 at top
        tile_type = self.get_tile_at(player_x, player_y)
        special_surfaces = [
            TileType.PLATFORM, TileType.ICE, TileType.CONVEYOR_LEFT, TileType.CONVEYOR_RIGHT,
            TileType.CLOUD, TileType.CARPET
        ]

        return tile_type in special_surfaces
