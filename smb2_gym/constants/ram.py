"""
RAM addresses and constants for Super Mario Bros 2 (Europe).

Detailed RAM map available at:

https://datacrystal.tcrf.net/wiki/Super_Mario_Bros._2_(NES)/RAM_map

or 

https://github.com/Xkeeper0/smb2/blob/master/src/ram.asm

NOTE: Some fields in these classes do not directly align with the above references
"""

from dataclasses import dataclass
from enum import IntEnum

# ------------------------------------------------------------------------------
# ---- Main RAM Properties -----------------------------------------------------
# ------------------------------------------------------------------------------


@dataclass
class GlobalCoordinate:
    """Global coordinate system combining level structure with player position.

    Represents a 4-tuple coordinate system: (Area, Sub-area, Global_X, Global_Y)

    This coordinate system provides a unified way to track the player's location
    within the level structure and their exact position in the game world:

    Level Structure:
    - Area: Current level
    - Sub-area: Subdivisions within an level

    World Position:
    - Global X: Absolute horizontal position in the sub-area
    - Global Y: Absolute vertical position in the sub-area (y=0 at bottom, increasing upward)

    Args:
        area: Current area from memory address $04E7
        sub_area: Current sub-area from memory address $04E8
        global_x: Player's global X position (x_page * PAGE_SIZE + x_position)
        global_y: Player's global Y position (inverted: MAX_Y_GLOBAL - (y_page * PAGE_SIZE + y_position))

    Example:
        coord = GlobalCoordinate(area=1, sub_area=0, global_x=1024, global_y=192)
        # Represents area 1, sub-area 0, position (1024, 192)
    """
    area: int
    sub_area: int
    global_x: int
    global_y: int


@dataclass
class EnemySlot:
    """RAM addresses for a single object slot.

    """
    slot_number: int  # 0-8
    x_position: int  # ObjectXLo
    y_position: int  # ObjectYLo
    x_page: int  # ObjectXHi
    y_page: int  # ObjectYHi
    object_type: int  # ObjectType
    health: int  # EnemyHP
    state: int  # EnemyState
    x_velocity: int  # ObjectXVelocity (signed)
    y_velocity: int  # ObjectYVelocity (signed)
    direction: int  # EnemyMovementDirection
    collision: int  # EnemyCollision (bitfield)
    object_timer: int  # ObjectTimer1 (only 8 slots: $0086-$008D)
    sprite_flags: int  # SpriteFlags46E (bitfield)


@dataclass
class Enemy:
    """Runtime data for a single enemy/object slot with computed properties."""
    slot_number: int
    x_position: int | None  # Local X on current page
    y_position: int | None  # Local Y (inverted: y=0 at bottom)
    x_page: int | None
    y_page: int | None
    object_type: int | None
    health: int | None
    state: int
    x_velocity: int | None  # Horizontal velocity (signed)
    y_velocity: int | None  # Vertical velocity (signed)
    direction: int | None  # Movement direction
    collision: int | None  # Collision flags bitfield
    object_timer: int | None  # ObjectTimer1
    sprite_flags: int | None  # Sprite property flags bitfield

    @property
    def is_visible(self) -> bool:
        """Check if enemy is visible."""
        return self.state == 1  # ENEMY_VISIBLE

    @property
    def is_dead(self) -> bool:
        """Check if enemy is dead."""
        return self.state == 2  # ENEMY_DEAD

    @property
    def global_x(self) -> int | None:
        """Get global X position."""
        if self.x_page is None or self.x_position is None:
            return None
        return (self.x_page * PAGE_SIZE) + self.x_position

    @property
    def global_y(self) -> int | None:
        """Get global Y position (y=0 at bottom)."""
        if self.y_page is None or self.y_position is None:
            return None
        # y_position is already inverted, so we need to calculate global
        # MAX_Y_GLOBAL - (y_page * PAGE_SIZE + raw_y_pos)
        # But since y_position is already inverted at local level, we use it directly
        return (self.y_page * PAGE_SIZE) + self.y_position

    def relative_x(self, player_global_x: int) -> int | None:
        """Get X position relative to player."""
        if self.global_x is None:
            return None
        return player_global_x - self.global_x

    def relative_y(self, player_global_y: int) -> int | None:
        """Get Y position relative to player."""
        if self.global_y is None:
            return None
        return player_global_y - self.global_y


@dataclass
class Player:
    """Player RAM addresses."""
    X_PAGE: int = 0x0014
    X_POSITION: int = 0x0028
    Y_PAGE: int = 0x001E
    Y_POSITION: int = 0x0032
    STATE: int = 0x0050
    SPEED: int = 0x003C
    COLLISION: int = 0x005A
    ON_VINE: int = 0x0050
    HOLDING_ITEM: int = 0x009C
    ITEM_PULLED: int = 0x0096
    LIVES: int = 0x04ED
    LIFE_METER: int = 0x04C2
    CHERRIES: int = 0x062A
    SUBSPACE_COINS: int = 0x062B
    BIG_VEGETABLES_PULLED: int = 0x062C  # Large vegetables pulled (required for stopwatch)
    CONTINUES: int = 0x05C5
    ENEMIES_DEFEATED: int = 0x04AD  # For heart spawning
    LEVELS_FINISHED_MARIO: int = 0x062D
    LEVELS_FINISHED_PEACH: int = 0x062E
    LEVELS_FINISHED_TOAD: int = 0x062F
    LEVELS_FINISHED_LUIGI: int = 0x0630


@dataclass
class Timers:
    """Timer RAM addresses."""
    FRAMERULE: int = 0x0010
    INVULNERABILITY: int = 0x0085
    STARMAN: int = 0x04E0
    SUBSPACE: int = 0x04B7
    DOOR_TRANSITION: int = 0x04BD
    STOPWATCH: int = 0x04FF
    FLOAT: int = 0x0553
    PIDGET_CARPET: int = 0x008E


@dataclass
class GameState:
    """Game state RAM addresses."""
    CHARACTER: int = 0x008F
    CURRENT_LEVEL: int = 0x0531
    WORLD_NUMBER: int = 0x0635
    LEVEL_TILESET: int = 0x06F7
    AREA: int = 0x04E7
    SUB_AREA: int = 0x04E8
    PAGE: int = 0x04E9
    CURRENT_PAGE_POSITION: int = 0x0535
    TOTAL_PAGES_IN_SUB_AREA: int = 0x053F
    SCROLL_DIRECTION: int = 0x00EC
    LEVEL_TRANSITION: int = 0x04EC
    SUBSPACE_STATUS: int = 0x0628


@dataclass
class Viewport:
    """Viewport and camera position RAM addresses."""
    SCREEN_BOUNDARY_LEFT_HI: int = 0x04BE  # High byte of left screen boundary (page)
    SCREEN_BOUNDARY_LEFT_LO: int = 0x04BF  # Low byte of left screen boundary (offset)
    SCREEN_Y_HI: int = 0x00CA  # High byte of vertical screen position
    SCREEN_Y_LO: int = 0x00CB  # Low byte of vertical screen position
    PPU_SCROLL_X_MIRROR: int = 0x00FD  # Horizontal scroll position (within nametable)
    PPU_SCROLL_Y_MIRROR: int = 0x00FC  # Vertical scroll position (within nametable)


ENEMY_SLOTS = [ \
    EnemySlot(slot_number=0, x_position=0x0029, y_position=0x0033, x_page=0x0015, y_page=0x001F, object_type=0x0090, health=0x0465, state=0x0051, x_velocity=0x003D, y_velocity=0x0047, direction=0x006F, collision=0x005B, object_timer=0x0086, sprite_flags=0x046E),
    EnemySlot(slot_number=1, x_position=0x002A, y_position=0x0034, x_page=0x0016, y_page=0x0020, object_type=0x0091, health=0x0466, state=0x0052, x_velocity=0x003E, y_velocity=0x0048, direction=0x0070, collision=0x005C, object_timer=0x0087, sprite_flags=0x046F),
    EnemySlot(slot_number=2, x_position=0x002B, y_position=0x0035, x_page=0x0017, y_page=0x0021, object_type=0x0092, health=0x0467, state=0x0053, x_velocity=0x003F, y_velocity=0x0049, direction=0x0071, collision=0x005D, object_timer=0x0088, sprite_flags=0x0470),
    EnemySlot(slot_number=3, x_position=0x002C, y_position=0x0036, x_page=0x0018, y_page=0x0022, object_type=0x0093, health=0x0468, state=0x0054, x_velocity=0x0040, y_velocity=0x004A, direction=0x0072, collision=0x005E, object_timer=0x0089, sprite_flags=0x0471),
    EnemySlot(slot_number=4, x_position=0x002D, y_position=0x0037, x_page=0x0019, y_page=0x0023, object_type=0x0094, health=0x0469, state=0x0055, x_velocity=0x0041, y_velocity=0x004B, direction=0x0073, collision=0x005F, object_timer=0x008A, sprite_flags=0x0472),
    EnemySlot(slot_number=5, x_position=0x002E, y_position=0x0038, x_page=0x001A, y_page=0x0024, object_type=0x0095, health=0x046A, state=0x0056, x_velocity=0x0042, y_velocity=0x004C, direction=0x0074, collision=0x0060, object_timer=0x008B, sprite_flags=0x0473),
    EnemySlot(slot_number=6, x_position=0x002F, y_position=0x0039, x_page=0x001B, y_page=0x0025, object_type=0x0096, health=0x046B, state=0x0057, x_velocity=0x0043, y_velocity=0x004D, direction=0x0075, collision=0x0061, object_timer=0x008C, sprite_flags=0x0474),
    EnemySlot(slot_number=7, x_position=0x0030, y_position=0x003A, x_page=0x001C, y_page=0x0026, object_type=0x0097, health=0x046C, state=0x0058, x_velocity=0x0044, y_velocity=0x004E, direction=0x0076, collision=0x0062, object_timer=0x008D, sprite_flags=0x0475),
    EnemySlot(slot_number=8, x_position=0x0031, y_position=0x003B, x_page=0x001D, y_page=0x0027, object_type=0x0098, health=0x046D, state=0x0059, x_velocity=0x0045, y_velocity=0x004F, direction=0x0077, collision=0x0063, object_timer=0x0086, sprite_flags=0x046E), # NOTE: ?? timer 8, shares sprite_flags with slot 0
              ]

# ------------------------------------------------------------------------------
# ---- Display/Rendering/Controls ----------------------------------------------
# ------------------------------------------------------------------------------


class Buttons(IntEnum):
    """NES controller button indices."""
    A = 0
    B = 1
    SELECT = 2
    START = 3
    UP = 4
    DOWN = 5
    LEFT = 6
    RIGHT = 7


# Display/Rendering constants
SCREEN_WIDTH = 256
SCREEN_HEIGHT = 240
DEFAULT_SCALE = 3
FONT_SIZE_BASE = 18
WINDOW_CAPTION = "Super Mario Bros 2"

# Tilemap level dims
TILE_SIZE = 16  # Pixels per tile (both width and height)
LEVEL_PAGE_WIDTH = 16  # Tiles per page horizontally
LEVEL_PAGE_HEIGHT = 15  # Tiles per page vertically
SCREEN_TILES_WIDTH = 16  # Visible tiles horizontally
SCREEN_TILES_HEIGHT = 15  # Visible tiles vertically (status bar excluded)

# Game mechanics
PAGE_SIZE = 256  # Memory page size for position calculations
GAME_INIT_FRAMES = 300  # Frames to wait for game initialisation

# Save state slots
MAX_SAVE_SLOTS = 10  # 0-9

# Game limits
MAX_CHERRIES = 20
MAX_COINS = 99
MAX_CONTINUES = 9
MAX_LIVES = 9
MAX_HEARTS = 4

# Collision flags - See CollisionFlags enum in object_ids.py for bitfield values
# PLAYER_COLLISION at Player.COLLISION (0x005A)
# ENEMY_COLLISION at EnemySlot.collision (0x005B-0x0063)

# Character names for display
CHARACTER_NAMES = {0: "Mario", 1: "Peach", 2: "Toad", 3: "Luigi"}

# Level names
LEVEL_NAMES = {
    0x00: "1-1",
    0x01: "1-2",
    0x02: "1-3",
    0x03: "2-1",
    0x04: "2-2",
    0x05: "2-3",
    0x06: "3-1",
    0x07: "3-2",
    0x08: "3-3",
    0x09: "4-1",
    0x0A: "4-2",
    0x0B: "4-3",
    0x0C: "5-1",
    0x0D: "5-2",
    0x0E: "5-3",
    0x0F: "6-1",
    0x10: "6-2",
    0x11: "6-3",
    0x12: "7-1",
    0x13: "7-2"
}

# ------------------------------------------------------------------------------
# ---- Instances ---------------------------------------------------------------
# ------------------------------------------------------------------------------

PLAYER = Player()
TIMERS = Timers()
GAME_STATE = GameState()
VIEWPORT = Viewport()

# ------------------------------------------------------------------------------
# ---- Other / Not Used --------------------------------------------------------
# ------------------------------------------------------------------------------

# Collision and Level Data
DECODED_LEVEL_DATA = 0x6000  # Decoded level data (tile grid used by CPU
RAW_LEVEL_DATA = 0x7800  # Raw/compressed level data
RAW_JAR_DATA = 0x7A00  # Raw jar/pipe data
RAW_ENEMY_DATA = 0x7B00  # Raw enemy data

# PPU Nametable addresses (for display, not collision)
PPU_NAMETABLE_0 = 0x2000  # Top-left nametable
PPU_NAMETABLE_1 = 0x2400  # Top-right nametable
PPU_NAMETABLE_2 = 0x2800  # Bottom-left nametable
PPU_NAMETABLE_3 = 0x2C00  # Bottom-right nametable
