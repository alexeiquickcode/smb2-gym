"""Semantic tile definitions for SMB2 environment.

This module contains all semantic tile types, categories, colours, and mappings
used for the semantic map observation space.
"""

from enum import IntEnum

import numpy as np

from .object_ids import (
    BackgroundTile,
    EnemyId,
)


# ------------------------------------------------------------------------------
# ---- Types -------------------------------------------------------------------
# ------------------------------------------------------------------------------


class CoarseTileType(IntEnum):
    """Coarse-grained semantic categories for tiles.

    These categories provide high-level semantic groupings that help
    RL agents learn general behaviors (e.g., 'avoid ENEMY', 'pick up PICKABLE').
    """

    EMPTY = 0  # Air/background
    TERRAIN = 1  # Static environment geometry
    INTERACTIVE = 2  # Things you can enter/activate
    PICKABLE = 3  # Things you can pick up and throw
    COLLECTIBLE = 4  # Things that auto-collect on touch
    ENEMY = 5  # Dynamic threats/moving objects
    HAZARD = 6  # Environmental dangers


class FineTileType(IntEnum):
    """Fine-grained tile types for semantic map.

    These provide detailed semantic information while maintaining hierarchical
    structure through CoarseTileType. Designed for RL agent observation space.
    """

    # EMPTY category
    EMPTY = 0

    # TERRAIN category - static environment geometry
    SOLID = 1
    PLATFORM = 2  # Semi-solid platform (can jump through from below)
    CLIMBABLE = 3

    # INTERACTIVE category - things you can enter/activate
    DOOR = 6
    JAR = 7

    # PICKABLE category - things you can pick up and throw
    VEGETABLE = 8
    BOMB = 9
    POW_BLOCK = 10
    KEY = 11
    POTION = 12
    MUSHROOM = 13

    # COLLECTIBLE category - things that auto-collect on touch
    CHERRY = 14

    # ENEMY category - dynamic threats (from enemy slots in RAM).
    # This means genuinely hostile: the many non-hostile sprites that share the
    # same RAM slots (hearts, coins, subspace doors, ...) classify as their own
    # type via OBJECT_ID_MAPPING rather than collapsing in here.
    ENEMY = 15

    # HAZARD category - environmental dangers
    SPIKES = 16
    QUICKSAND = 17
    CONVEYOR_LEFT = 18
    CONVEYOR_RIGHT = 19

    # Types below exist only as sprites (enemy slots), never as background tiles
    COIN = 20  # COLLECTIBLE - subspace coin
    HEART = 21  # COLLECTIBLE - refills the life meter
    POWERUP = 22  # COLLECTIBLE - starman, stopwatch, crystal ball
    SHELL = 23  # PICKABLE - koopa shell, slides when thrown
    PROJECTILE = 24  # ENEMY - fireballs, bullets, sparks: hostile but not defeatable


# RGB colours for tile types
TILE_COLORS: dict[FineTileType, tuple[int, int, int]] = {
    # EMPTY
    FineTileType.EMPTY: (200, 200, 200),  # light gray
    # TERRAIN - browns/tans for ground
    FineTileType.SOLID: (101, 67, 33),  # brown
    FineTileType.PLATFORM: (139, 69, 19),  # saddle brown
    FineTileType.CLIMBABLE: (34, 139, 34),  # forest green
    # INTERACTIVE - gold for entrances
    FineTileType.DOOR: (255, 215, 0),  # gold
    FineTileType.JAR: (255, 215, 0),  # gold
    # PICKABLE - distinct colours for each type
    FineTileType.VEGETABLE: (50, 205, 50),  # lime green
    FineTileType.BOMB: (128, 0, 0),  # maroon
    FineTileType.POW_BLOCK: (255, 255, 224),  # light yellow
    FineTileType.KEY: (255, 215, 0),  # gold
    # COLLECTIBLE - bright colours
    FineTileType.CHERRY: (255, 20, 147),  # deep pink
    FineTileType.POTION: (186, 85, 211),  # medium orchid
    FineTileType.MUSHROOM: (255, 140, 0),  # dark orange
    # ENEMY - red for threats
    FineTileType.ENEMY: (255, 0, 0),  # red
    FineTileType.PROJECTILE: (255, 99, 71),  # tomato - hostile but not defeatable
    # HAZARD - reds/oranges for danger
    FineTileType.SPIKES: (220, 20, 60),  # crimson
    FineTileType.QUICKSAND: (244, 164, 96),  # sandy brown
    FineTileType.CONVEYOR_LEFT: (105, 105, 105),  # dim gray
    FineTileType.CONVEYOR_RIGHT: (105, 105, 105),  # dim gray
    # Sprite-only types
    FineTileType.COIN: (255, 223, 0),  # golden yellow
    FineTileType.HEART: (255, 105, 180),  # hot pink
    FineTileType.POWERUP: (0, 255, 255),  # cyan
    FineTileType.SHELL: (46, 139, 87),  # sea green
}

# ------------------------------------------------------------------------------
# ---- Lookup Tables & Dtypes --------------------------------------------------
# ------------------------------------------------------------------------------

# Human-readable names derived from enum names
FINE_TILE_NAMES: dict[FineTileType, str] = {tile_type: tile_type.name for tile_type in FineTileType}
COARSE_TILE_NAMES: dict[CoarseTileType, str] = {
    category: category.name for category in CoarseTileType
}

# Sentinel for "no sprite object in this cell". 0x00 is a real object id (HEART),
# so absence needs its own value; 0xFF is not a valid EnemyId.
NO_OBJECT = 0xFF

# Structured dtype for semantic map tiles with hierarchical information.
#
# Terrain and sprite objects are kept in SEPARATE fields. A cell can hold both --
# a subspace door standing on solid ground -- and collapsing them into one value
# would lose exactly the distinction an agent needs ("am I on top of the door or
# inside it"). The terrain fields therefore describe the background tile only and
# are never overwritten by a sprite.
SEMANTIC_TILE_DTYPE = np.dtype(
    [
        ('tile_id', np.uint8),  # Raw BackgroundTile ID
        ('fine_type', np.uint8),  # Fine-grained FineTileType of the TERRAIN
        ('coarse_type', np.uint8),  # Coarse-grained CoarseTileType of the TERRAIN
        ('object_id', np.uint8),  # EnemyId of the sprite here, or NO_OBJECT
        ('object_fine_type', np.uint8),  # Fine-grained FineTileType of that sprite
        ('object_coarse_type', np.uint8),  # Coarse-grained CoarseTileType of that sprite
        ('object_damages', np.uint8),  # 1 if touching this sprite hurts the player
        ('object_liftable', np.uint8),  # 1 if it can be picked up and thrown
        ('object_velocity_x', np.float32),  # Normalised to ~[-1, 1]
        ('object_velocity_y', np.float32),  # Normalised to ~[-1, 1], positive = down
        ('color_r', np.uint8),  # RGB colour for visualisation (object over terrain)
        ('color_g', np.uint8),
        ('color_b', np.uint8),
    ]
)

# Pre-computed lookup tables for fast vectorized operations
COARSE_LOOKUP = np.zeros(max(FineTileType) + 1, dtype=np.uint8)
COLOR_LOOKUP = np.zeros((max(FineTileType) + 1, 3), dtype=np.uint8)

# ------------------------------------------------------------------------------
# ---- Tensor channel layout ---------------------------------------------------
# ------------------------------------------------------------------------------

# Channels for the (H, W, C) binary tensor. Every coarse category appears TWICE:
# once for the terrain layer and once for the sprite layer. The two layers are
# what make a cell able to say "solid ground, with a door standing on it", and
# the same category legitimately occurs in either -- a cherry is a background
# tile (COLLECTIBLE terrain) while a coin is a sprite (COLLECTIBLE object). One
# shared channel per category would merge them and lose the distinction.
#
# Within each layer the encoding is one-hot; across layers it is multi-hot.
#
# Category ids are NOT used as channel values: they are nominal labels, and
# feeding them as numbers would imply ENEMY (15) is "more" than SOLID (1). Binary
# channels carry no such false ordering.
_COARSE_CATEGORIES: tuple[CoarseTileType, ...] = tuple(CoarseTileType)

# (layer, category) per channel, where layer is 'terrain' or 'object'
COARSE_TENSOR_CHANNELS: tuple[tuple[str, CoarseTileType], ...] = tuple(
    [('terrain', category) for category in _COARSE_CATEGORIES]
    + [('object', category) for category in _COARSE_CATEGORIES]
)

# Extra binary property channels appended after the category channels. These are
# per-object behaviour, not category: they answer "what happens if I touch this",
# which is the actual decision an agent makes about a sprite.
#
# Deliberately a small, curated set. Most SpriteFlags bits (TILEMAP2,
# DOUBLE_SPEED, MIRROR_ANIMATION) are rendering metadata and would just feed the
# net the sprite tilemap index; those stay in `info['enemies']` for debugging.
PROPERTY_CHANNEL_NAMES: tuple[str, ...] = (
    'object:DAMAGES',  # Touching this hurts the player
    'object:LIFTABLE',  # Can be picked up and thrown
)

# Human-readable channel names, e.g. 'terrain:SOLID', 'object:ENEMY'
COARSE_TENSOR_CHANNEL_NAMES: tuple[str, ...] = tuple(
    [f"{layer}:{category.name}" for layer, category in COARSE_TENSOR_CHANNELS]
    + list(PROPERTY_CHANNEL_NAMES)
)

# Names of the two velocity planes, returned separately as float32
VELOCITY_CHANNEL_NAMES: tuple[str, ...] = ('object:VELOCITY_X', 'object:VELOCITY_Y')

FINE_TENSOR_CHANNELS: tuple[FineTileType, ...] = tuple(FineTileType)

# Draw order for visualisation only: later entries render on top. Objects sit
# above terrain so a door standing on ground shows as a door, while the terrain
# stays visible underneath as a border.
RENDER_PRIORITY: tuple[CoarseTileType, ...] = (
    CoarseTileType.EMPTY,
    CoarseTileType.TERRAIN,
    CoarseTileType.HAZARD,
    CoarseTileType.COLLECTIBLE,
    CoarseTileType.PICKABLE,
    CoarseTileType.INTERACTIVE,
    CoarseTileType.ENEMY,
)

# ------------------------------------------------------------------------------
# ---- Mappings ----------------------------------------------------------------
# ------------------------------------------------------------------------------

# fine-grained -> to coarse-grained
FINE_TO_COARSE_MAPPING: dict[FineTileType, CoarseTileType] = {
    FineTileType.EMPTY: CoarseTileType.EMPTY,
    # TERRAIN
    FineTileType.SOLID: CoarseTileType.TERRAIN,
    FineTileType.PLATFORM: CoarseTileType.TERRAIN,
    FineTileType.CLIMBABLE: CoarseTileType.TERRAIN,
    # INTERACTIVE
    FineTileType.DOOR: CoarseTileType.INTERACTIVE,
    FineTileType.JAR: CoarseTileType.INTERACTIVE,
    # PICKABLE
    FineTileType.VEGETABLE: CoarseTileType.PICKABLE,
    FineTileType.BOMB: CoarseTileType.PICKABLE,
    FineTileType.POW_BLOCK: CoarseTileType.PICKABLE,
    FineTileType.KEY: CoarseTileType.PICKABLE,
    FineTileType.POTION: CoarseTileType.PICKABLE,
    FineTileType.MUSHROOM: CoarseTileType.PICKABLE,
    FineTileType.SHELL: CoarseTileType.PICKABLE,
    # COLLECTIBLE
    FineTileType.CHERRY: CoarseTileType.COLLECTIBLE,
    FineTileType.COIN: CoarseTileType.COLLECTIBLE,
    FineTileType.HEART: CoarseTileType.COLLECTIBLE,
    FineTileType.POWERUP: CoarseTileType.COLLECTIBLE,
    # ENEMY
    FineTileType.ENEMY: CoarseTileType.ENEMY,
    FineTileType.PROJECTILE: CoarseTileType.ENEMY,
    # HAZARD
    FineTileType.SPIKES: CoarseTileType.HAZARD,
    FineTileType.QUICKSAND: CoarseTileType.HAZARD,
    FineTileType.CONVEYOR_LEFT: CoarseTileType.HAZARD,
    FineTileType.CONVEYOR_RIGHT: CoarseTileType.HAZARD,
}

for fine_type, coarse_type in FINE_TO_COARSE_MAPPING.items():
    COARSE_LOOKUP[fine_type] = coarse_type
    if fine_type in TILE_COLORS:
        COLOR_LOOKUP[fine_type] = TILE_COLORS[fine_type]

# Mapping from raw BackgroundTile IDs to semantic FineTileType
TILE_ID_MAPPING: dict[int, FineTileType] = {
    BackgroundTile.BLACK: FineTileType.EMPTY,
    BackgroundTile.SKY: FineTileType.EMPTY,
    BackgroundTile.BG_CLOUD_LEFT: FineTileType.EMPTY,
    BackgroundTile.BG_CLOUD_RIGHT: FineTileType.EMPTY,
    BackgroundTile.BG_CLOUD_SMALL: FineTileType.EMPTY,
    BackgroundTile.STAR_BG_1: FineTileType.EMPTY,
    BackgroundTile.STAR_BG_2: FineTileType.EMPTY,
    BackgroundTile.BACKGROUND_BRICK: FineTileType.EMPTY,
    # Decorative background elements (non-interactive)
    BackgroundTile.WATERFALL_TOP: FineTileType.EMPTY,
    BackgroundTile.WATERFALL: FineTileType.EMPTY,
    BackgroundTile.WATERFALL_SPLASH: FineTileType.EMPTY,
    BackgroundTile.HOUSE_LEFT: FineTileType.EMPTY,
    BackgroundTile.HOUSE_MIDDLE: FineTileType.EMPTY,
    BackgroundTile.HOUSE_RIGHT: FineTileType.EMPTY,
    BackgroundTile.PALM_TREE_TRUNK: FineTileType.EMPTY,
    BackgroundTile.PALM_TREE_TOP: FineTileType.EMPTY,
    BackgroundTile.TREE_BACKGROUND_LEFT: FineTileType.EMPTY,
    BackgroundTile.TREE_BACKGROUND_MIDDLE_LEFT: FineTileType.EMPTY,
    BackgroundTile.TREE_BACKGROUND_RIGHT: FineTileType.EMPTY,
    BackgroundTile.TREE_BACKGROUND_MIDDLE_RIGHT: FineTileType.EMPTY,
    BackgroundTile.WHALE: FineTileType.EMPTY,
    BackgroundTile.WHALE_EYE: FineTileType.EMPTY,
    BackgroundTile.WHALE_TOP_LEFT: FineTileType.EMPTY,
    BackgroundTile.WHALE_TOP: FineTileType.EMPTY,
    BackgroundTile.WHALE_TOP_RIGHT: FineTileType.EMPTY,
    BackgroundTile.WHALE_TAIL: FineTileType.EMPTY,
    BackgroundTile.PHANTO: FineTileType.EMPTY,
    BackgroundTile.DRAW_BRIDGE_CHAIN: FineTileType.EMPTY,
    BackgroundTile.WINDOW_TOP: FineTileType.EMPTY,
    BackgroundTile.DOORWAY_TOP: FineTileType.EMPTY,
    BackgroundTile.JAR_OUTSIDE_BACKGROUND: FineTileType.EMPTY,
    BackgroundTile.LIGHT_TRAIL_LEFT: FineTileType.EMPTY,
    BackgroundTile.LIGHT_TRAIL: FineTileType.EMPTY,
    BackgroundTile.LIGHT_TRAIL_RIGHT: FineTileType.EMPTY,
    BackgroundTile.HORN_TOP_LEFT: FineTileType.EMPTY,
    BackgroundTile.HORN_TOP_RIGHT: FineTileType.EMPTY,
    BackgroundTile.HORN_BOTTOM_LEFT: FineTileType.EMPTY,
    BackgroundTile.HORN_BOTTOM_RIGHT: FineTileType.EMPTY,
    # Solid ground/wall tiles
    BackgroundTile.SOLID_GRASS: FineTileType.SOLID,
    BackgroundTile.SOLID_SAND: FineTileType.SOLID,
    BackgroundTile.SOLID_BRICK_0: FineTileType.SOLID,
    BackgroundTile.SOLID_BRICK_1: FineTileType.SOLID,
    BackgroundTile.SOLID_BRICK_2: FineTileType.SOLID,
    BackgroundTile.SOLID_BRICK_3: FineTileType.SOLID,
    BackgroundTile.GROUND_BRICK_0: FineTileType.SOLID,
    BackgroundTile.GROUND_BRICK_2: FineTileType.SOLID,
    BackgroundTile.GROUND_BRICK_3: FineTileType.SOLID,
    BackgroundTile.SOLID_ROUND_BRICK_0: FineTileType.SOLID,
    BackgroundTile.SOLID_ROUND_BRICK_2: FineTileType.SOLID,
    BackgroundTile.SOLID_BLOCK: FineTileType.SOLID,
    BackgroundTile.SOLID_WOOD: FineTileType.SOLID,
    BackgroundTile.FROZEN_ROCK: FineTileType.SOLID,
    BackgroundTile.CLOUD_LEFT: FineTileType.SOLID,
    BackgroundTile.CLOUD_MIDDLE: FineTileType.SOLID,
    BackgroundTile.CLOUD_RIGHT: FineTileType.SOLID,
    BackgroundTile.MUSHROOM_TOP_LEFT: FineTileType.SOLID,
    BackgroundTile.MUSHROOM_TOP_MIDDLE: FineTileType.SOLID,
    BackgroundTile.MUSHROOM_TOP_RIGHT: FineTileType.SOLID,
    BackgroundTile.GREEN_PLATFORM_TOP: FineTileType.SOLID,
    BackgroundTile.GREEN_PLATFORM_TOP_LEFT: FineTileType.SOLID,
    BackgroundTile.GREEN_PLATFORM_TOP_RIGHT: FineTileType.SOLID,
    BackgroundTile.GREEN_PLATFORM_MIDDLE: FineTileType.SOLID,
    BackgroundTile.GREEN_PLATFORM_LEFT: FineTileType.SOLID,
    BackgroundTile.GREEN_PLATFORM_RIGHT: FineTileType.SOLID,
    BackgroundTile.GREEN_PLATFORM_TOP_LEFT_OVERLAP: FineTileType.SOLID,
    BackgroundTile.GREEN_PLATFORM_TOP_RIGHT_OVERLAP: FineTileType.SOLID,
    BackgroundTile.GREEN_PLATFORM_TOP_LEFT_OVERLAP_EDGE: FineTileType.SOLID,
    BackgroundTile.GREEN_PLATFORM_TOP_RIGHT_OVERLAP_EDGE: FineTileType.SOLID,
    BackgroundTile.LOG_LEFT: FineTileType.SOLID,
    BackgroundTile.LOG_MIDDLE: FineTileType.SOLID,
    BackgroundTile.LOG_RIGHT: FineTileType.SOLID,
    BackgroundTile.LOG_PILLAR_TOP_0: FineTileType.SOLID,
    BackgroundTile.LOG_PILLAR_MIDDLE_0: FineTileType.SOLID,
    BackgroundTile.LOG_PILLAR_TOP_1: FineTileType.SOLID,
    BackgroundTile.LOG_PILLAR_MIDDLE_1: FineTileType.SOLID,
    BackgroundTile.PYRAMID_LEFT_ANGLE: FineTileType.SOLID,
    BackgroundTile.PYRAMID_LEFT: FineTileType.SOLID,
    BackgroundTile.PYRAMID_RIGHT: FineTileType.SOLID,
    BackgroundTile.PYRAMID_RIGHT_ANGLE: FineTileType.SOLID,
    BackgroundTile.ROCK_WALL_ANGLE: FineTileType.SOLID,
    BackgroundTile.ROCK_WALL: FineTileType.SOLID,
    BackgroundTile.ROCK_WALL_OFFSET: FineTileType.SOLID,
    BackgroundTile.ROCK_WALL_EYE_LEFT: FineTileType.SOLID,
    BackgroundTile.ROCK_WALL_EYE_RIGHT: FineTileType.SOLID,
    BackgroundTile.ROCK_WALL_MOUTH: FineTileType.SOLID,
    BackgroundTile.COLUMN_PILLAR_TOP_2: FineTileType.SOLID,
    BackgroundTile.COLUMN_PILLAR_MIDDLE_2: FineTileType.SOLID,
    BackgroundTile.JAR_WALL: FineTileType.SOLID,
    BackgroundTile.SOLID_BRICK_2_WALL: FineTileType.SOLID,
    BackgroundTile.CACTUS_TOP: FineTileType.SOLID,
    BackgroundTile.CACTUS_MIDDLE: FineTileType.SOLID,
    BackgroundTile.BRIDGE: FineTileType.SOLID,
    BackgroundTile.BRIDGE_SHADOW: FineTileType.SOLID,
    BackgroundTile.MUSHROOM_BLOCK: FineTileType.SOLID,
    BackgroundTile.LOG_RIGHT_TREE: FineTileType.SOLID,
    BackgroundTile.CLAW_GRIP_ROCK: FineTileType.SOLID,
    BackgroundTile.BOMBABLE_BRICK: FineTileType.SOLID,
    BackgroundTile.TILE_98: FineTileType.SOLID,
    BackgroundTile.TILE_9A: FineTileType.SOLID,
    # Door tiles
    BackgroundTile.DOOR_TOP: FineTileType.DOOR,
    BackgroundTile.DOOR_BOTTOM_LOCK: FineTileType.DOOR,
    BackgroundTile.DOOR_BOTTOM: FineTileType.DOOR,
    BackgroundTile.DOOR_BOTTOM_LOCK_STUCK: FineTileType.DOOR,
    BackgroundTile.LIGHT_DOOR: FineTileType.DOOR,
    BackgroundTile.LIGHT_DOOR_END_LEVEL: FineTileType.DOOR,
    BackgroundTile.DARK_DOOR: FineTileType.DOOR,
    # Climbable tiles (vines, ladders, chains)
    BackgroundTile.VINE_TOP: FineTileType.CLIMBABLE,
    BackgroundTile.VINE: FineTileType.CLIMBABLE,
    BackgroundTile.VINE_BOTTOM: FineTileType.CLIMBABLE,
    BackgroundTile.VINE_STANDABLE: FineTileType.CLIMBABLE,
    BackgroundTile.LADDER: FineTileType.CLIMBABLE,
    BackgroundTile.LADDER_STANDABLE: FineTileType.CLIMBABLE,
    BackgroundTile.CHAIN: FineTileType.CLIMBABLE,
    BackgroundTile.CHAIN_STANDABLE: FineTileType.CLIMBABLE,
    BackgroundTile.LADDER_SHADOW: FineTileType.CLIMBABLE,
    BackgroundTile.LADDER_STANDABLE_SHADOW: FineTileType.CLIMBABLE,
    BackgroundTile.CLIMBABLE_SKY: FineTileType.CLIMBABLE,
    # Platform tiles (jump-through)
    BackgroundTile.JUMP_THROUGH_BLOCK: FineTileType.PLATFORM,
    BackgroundTile.JUMP_THROUGH_ICE: FineTileType.PLATFORM,
    BackgroundTile.JUMP_THROUGH_MACHINE_BLOCK: FineTileType.PLATFORM,
    BackgroundTile.JUMPTHROUGH_WOOD_BLOCK: FineTileType.PLATFORM,
    BackgroundTile.JUMPTHROUGH_SAND_BLOCK: FineTileType.PLATFORM,
    BackgroundTile.JUMPTHROUGH_BRICK: FineTileType.PLATFORM,
    BackgroundTile.JUMPTHROUGH_SAND: FineTileType.PLATFORM,
    # Spikes
    BackgroundTile.SPIKES: FineTileType.SPIKES,
    # Quicksand
    BackgroundTile.QUICKSAND_FAST: FineTileType.QUICKSAND,
    BackgroundTile.QUICKSAND_SLOW: FineTileType.QUICKSAND,
    BackgroundTile.DIGGABLE_SAND: FineTileType.QUICKSAND,
    # Conveyor
    BackgroundTile.CONVEYOR_LEFT: FineTileType.CONVEYOR_LEFT,
    BackgroundTile.CONVEYOR_RIGHT: FineTileType.CONVEYOR_RIGHT,
    # Water (visual background only, not actionable)
    BackgroundTile.WATER: FineTileType.EMPTY,
    BackgroundTile.WATER_TOP: FineTileType.EMPTY,
    # Whale is solid platform
    BackgroundTile.WATER_WHALE: FineTileType.SOLID,
    BackgroundTile.WATER_WHALE_TAIL: FineTileType.SOLID,
    # Collectibles (auto-collect on touch)
    BackgroundTile.CHERRY: FineTileType.CHERRY,
    BackgroundTile.GRASS_COIN: FineTileType.CHERRY,
    BackgroundTile.GRASS_POTION: FineTileType.POTION,
    BackgroundTile.SUBSPACE_MUSHROOM_1: FineTileType.MUSHROOM,
    BackgroundTile.SUBSPACE_MUSHROOM_2: FineTileType.MUSHROOM,
    # Pickable items (can pick up and throw)
    BackgroundTile.GRASS_LARGE_VEGGIE: FineTileType.VEGETABLE,
    BackgroundTile.GRASS_SMALL_VEGGIE: FineTileType.VEGETABLE,
    BackgroundTile.GRASS_BOMB: FineTileType.BOMB,
    BackgroundTile.GRASS_BOB_OMB: FineTileType.BOMB,
    BackgroundTile.GRASS_POW: FineTileType.POW_BLOCK,
    BackgroundTile.POW_BLOCK: FineTileType.POW_BLOCK,
    BackgroundTile.GRASS_ROCKET: FineTileType.VEGETABLE,
    BackgroundTile.GRASS_SHELL: FineTileType.VEGETABLE,
    BackgroundTile.GRASS_1UP: FineTileType.MUSHROOM,
    BackgroundTile.GRASS_INACTIVE: FineTileType.EMPTY,
    # Jar
    BackgroundTile.JAR_TOP_GENERIC: FineTileType.JAR,
    BackgroundTile.JAR_TOP_NON_ENTERABLE: FineTileType.JAR,
    BackgroundTile.JAR_TOP_POINTER: FineTileType.JAR,
    BackgroundTile.JAR_MIDDLE: FineTileType.JAR,
    BackgroundTile.JAR_BOTTOM: FineTileType.JAR,
    BackgroundTile.JAR_SMALL: FineTileType.JAR,
    # Unused/unknown tiles (treated as empty/non-interactive)
    BackgroundTile.UNUSED_0E: FineTileType.EMPTY,
    BackgroundTile.UNUSED_0F: FineTileType.EMPTY,
    BackgroundTile.UNUSED_10: FineTileType.EMPTY,
    BackgroundTile.UNUSED_23: FineTileType.EMPTY,
    BackgroundTile.UNUSED_24: FineTileType.EMPTY,
    BackgroundTile.UNUSED_25: FineTileType.EMPTY,
    BackgroundTile.UNUSED_26: FineTileType.EMPTY,
    BackgroundTile.UNUSED_27: FineTileType.EMPTY,
    BackgroundTile.UNUSED_28: FineTileType.EMPTY,
    BackgroundTile.UNUSED_29: FineTileType.EMPTY,
    BackgroundTile.UNUSED_2A: FineTileType.EMPTY,
    BackgroundTile.UNUSED_2B: FineTileType.EMPTY,
    BackgroundTile.UNUSED_2C: FineTileType.EMPTY,
    BackgroundTile.UNUSED_2D: FineTileType.EMPTY,
    BackgroundTile.UNUSED_2E: FineTileType.EMPTY,
    BackgroundTile.UNUSED_2F: FineTileType.EMPTY,
    BackgroundTile.UNUSED_30: FineTileType.EMPTY,
    BackgroundTile.UNUSED_31: FineTileType.EMPTY,
    BackgroundTile.UNUSED_32: FineTileType.EMPTY,
    BackgroundTile.UNUSED_33: FineTileType.EMPTY,
    BackgroundTile.UNUSED_34: FineTileType.EMPTY,
    BackgroundTile.UNUSED_35: FineTileType.EMPTY,
    BackgroundTile.UNUSED_36: FineTileType.EMPTY,
    BackgroundTile.UNUSED_37: FineTileType.EMPTY,
    BackgroundTile.UNUSED_38: FineTileType.EMPTY,
    BackgroundTile.UNUSED_39: FineTileType.EMPTY,
    BackgroundTile.UNUSED_3A: FineTileType.EMPTY,
    BackgroundTile.UNUSED_3B: FineTileType.EMPTY,
    BackgroundTile.UNUSED_3C: FineTileType.EMPTY,
    BackgroundTile.UNUSED_3D: FineTileType.EMPTY,
    BackgroundTile.UNUSED_3E: FineTileType.EMPTY,
    BackgroundTile.UNUSED_3F: FineTileType.EMPTY,
    BackgroundTile.UNUSED_6A_MUSHROOM_BLOCK: FineTileType.EMPTY,
    BackgroundTile.UNUSED_6B_MUSHROOM_BLOCK: FineTileType.EMPTY,
    BackgroundTile.UNUSED_6D: FineTileType.EMPTY,
    BackgroundTile.UNUSED_7B: FineTileType.EMPTY,
    BackgroundTile.UNUSED_7C: FineTileType.EMPTY,
    BackgroundTile.UNUSED_7D: FineTileType.EMPTY,
    BackgroundTile.UNUSED_7E: FineTileType.EMPTY,
    BackgroundTile.UNUSED_7F: FineTileType.EMPTY,
    BackgroundTile.UNUSED_AC: FineTileType.EMPTY,
    BackgroundTile.UNUSED_AD: FineTileType.EMPTY,
    BackgroundTile.UNUSED_AE: FineTileType.EMPTY,
    BackgroundTile.UNUSED_AF: FineTileType.EMPTY,
    BackgroundTile.UNUSED_B0: FineTileType.EMPTY,
    BackgroundTile.UNUSED_B1: FineTileType.EMPTY,
    BackgroundTile.UNUSED_B2: FineTileType.EMPTY,
    BackgroundTile.UNUSED_B3: FineTileType.EMPTY,
    BackgroundTile.UNUSED_B4: FineTileType.EMPTY,
    BackgroundTile.UNUSED_B5: FineTileType.EMPTY,
    BackgroundTile.UNUSED_B6: FineTileType.EMPTY,
    BackgroundTile.UNUSED_B7: FineTileType.EMPTY,
    BackgroundTile.UNUSED_B8: FineTileType.EMPTY,
    BackgroundTile.UNUSED_B9: FineTileType.EMPTY,
    BackgroundTile.UNUSED_BA: FineTileType.EMPTY,
    BackgroundTile.UNUSED_BB: FineTileType.EMPTY,
    BackgroundTile.UNUSED_BC: FineTileType.EMPTY,
    BackgroundTile.UNUSED_BD: FineTileType.EMPTY,
    BackgroundTile.UNUSED_BE: FineTileType.EMPTY,
    BackgroundTile.UNUSED_BF: FineTileType.EMPTY,
    BackgroundTile.UNUSED_C5: FineTileType.EMPTY,
    BackgroundTile.UNUSED_D8: FineTileType.EMPTY,
    BackgroundTile.UNUSED_D9: FineTileType.EMPTY,
    BackgroundTile.UNUSED_DA: FineTileType.EMPTY,
    BackgroundTile.UNUSED_DB: FineTileType.EMPTY,
    BackgroundTile.UNUSED_DC: FineTileType.EMPTY,
    BackgroundTile.UNUSED_DD: FineTileType.EMPTY,
    BackgroundTile.UNUSED_DE: FineTileType.EMPTY,
    BackgroundTile.UNUSED_DF: FineTileType.EMPTY,
    BackgroundTile.UNUSED_E0: FineTileType.EMPTY,
    BackgroundTile.UNUSED_E1: FineTileType.EMPTY,
    BackgroundTile.UNUSED_E2: FineTileType.EMPTY,
    BackgroundTile.UNUSED_E3: FineTileType.EMPTY,
    BackgroundTile.UNUSED_E4: FineTileType.EMPTY,
    BackgroundTile.UNUSED_E5: FineTileType.EMPTY,
    BackgroundTile.UNUSED_E6: FineTileType.EMPTY,
    BackgroundTile.UNUSED_E7: FineTileType.EMPTY,
    BackgroundTile.UNUSED_E8: FineTileType.EMPTY,
    BackgroundTile.UNUSED_E9: FineTileType.EMPTY,
    BackgroundTile.UNUSED_EA: FineTileType.EMPTY,
    BackgroundTile.UNUSED_EB: FineTileType.EMPTY,
    BackgroundTile.UNUSED_EC: FineTileType.EMPTY,
    BackgroundTile.UNUSED_ED: FineTileType.EMPTY,
    BackgroundTile.UNUSED_EE: FineTileType.EMPTY,
    BackgroundTile.UNUSED_EF: FineTileType.EMPTY,
    BackgroundTile.UNUSED_F0: FineTileType.EMPTY,
    BackgroundTile.UNUSED_F1: FineTileType.EMPTY,
    BackgroundTile.UNUSED_F2: FineTileType.EMPTY,
    BackgroundTile.UNUSED_F3: FineTileType.EMPTY,
    BackgroundTile.UNUSED_F4: FineTileType.EMPTY,
    BackgroundTile.UNUSED_F5: FineTileType.EMPTY,
    BackgroundTile.UNUSED_F6: FineTileType.EMPTY,
    BackgroundTile.UNUSED_F7: FineTileType.EMPTY,
    BackgroundTile.UNUSED_F8: FineTileType.EMPTY,
    BackgroundTile.UNUSED_F9: FineTileType.EMPTY,
    BackgroundTile.UNUSED_FA: FineTileType.EMPTY,
    BackgroundTile.UNUSED_FB: FineTileType.EMPTY,
    BackgroundTile.UNUSED_FC: FineTileType.EMPTY,
    BackgroundTile.UNUSED_FD: FineTileType.EMPTY,
    BackgroundTile.UNUSED_FE: FineTileType.EMPTY,
    BackgroundTile.UNUSED_FF: FineTileType.EMPTY,
}

# ------------------------------------------------------------------------------
# ---- Sprite (object) mappings ------------------------------------------------
# ------------------------------------------------------------------------------

# Mapping from EnemyId (the sprite/object slots at $0090-$0098) to FineTileType.
#
# The enum is named EnemyId, but the slots hold every dynamic object in the game:
# hearts, coins, vegetables, POW blocks, subspace doors and potions all live here
# alongside the actual enemies. Classifying them all as ENEMY would tell an agent
# to avoid the door it is supposed to walk into, so each is mapped to the same
# FineTileType its background-tile equivalent uses -- a POW block reads as
# POW_BLOCK whether it came from the tile map or from a sprite slot.
#
# Only genuinely hostile objects map to ENEMY (or PROJECTILE, for the hostile
# things that cannot be defeated).
OBJECT_ID_MAPPING: dict[int, FineTileType] = {
    # ---- COLLECTIBLE - auto-collect on touch ----
    EnemyId.HEART: FineTileType.HEART,
    EnemyId.COIN: FineTileType.COIN,
    EnemyId.MUSHROOM: FineTileType.MUSHROOM,
    EnemyId.MUSHROOM_1UP: FineTileType.MUSHROOM,
    EnemyId.STARMAN: FineTileType.POWERUP,
    EnemyId.STOPWATCH: FineTileType.POWERUP,
    EnemyId.CRYSTAL_BALL: FineTileType.POWERUP,
    # ---- PICKABLE - can be picked up and thrown ----
    EnemyId.VEGETABLE_SMALL: FineTileType.VEGETABLE,
    EnemyId.VEGETABLE_LARGE: FineTileType.VEGETABLE,
    EnemyId.VEGETABLE_WART: FineTileType.VEGETABLE,
    EnemyId.SHELL: FineTileType.SHELL,
    EnemyId.BOMB: FineTileType.BOMB,
    EnemyId.BOB_OMB: FineTileType.BOMB,  # walks, but is picked up and thrown
    EnemyId.ROCKET: FineTileType.VEGETABLE,
    EnemyId.MUSHROOM_BLOCK: FineTileType.POW_BLOCK,
    EnemyId.POW_BLOCK: FineTileType.POW_BLOCK,
    EnemyId.KEY: FineTileType.KEY,
    EnemyId.SUBSPACE_POTION: FineTileType.POTION,
    # ---- INTERACTIVE - enter/activate ----
    EnemyId.SUBSPACE_DOOR: FineTileType.DOOR,
    # Hawkmouth is the level-exit mouth: an entrance, not a threat.
    EnemyId.HAWKMOUTH_RIGHT: FineTileType.DOOR,
    EnemyId.HAWKMOUTH_LEFT: FineTileType.DOOR,
    # ---- TERRAIN - ridable / standable objects ----
    EnemyId.FLYING_CARPET: FineTileType.PLATFORM,
    EnemyId.FALLING_LOGS: FineTileType.PLATFORM,
    # ---- ENEMY - genuinely hostile ----
    EnemyId.SHYGUY_RED: FineTileType.ENEMY,
    EnemyId.SHYGUY_PINK: FineTileType.ENEMY,
    EnemyId.TWEETER: FineTileType.ENEMY,
    EnemyId.PORCUPO: FineTileType.ENEMY,
    EnemyId.SNIFIT_RED: FineTileType.ENEMY,
    EnemyId.SNIFIT_GRAY: FineTileType.ENEMY,
    EnemyId.SNIFIT_PINK: FineTileType.ENEMY,
    EnemyId.OSTRO: FineTileType.ENEMY,
    EnemyId.ALBATOSS_CARRYING_BOB_OMB: FineTileType.ENEMY,
    EnemyId.ALBATOSS_START_RIGHT: FineTileType.ENEMY,
    EnemyId.ALBATOSS_START_LEFT: FineTileType.ENEMY,
    EnemyId.NINJI_RUNNING: FineTileType.ENEMY,
    EnemyId.NINJI_JUMPING: FineTileType.ENEMY,
    EnemyId.BEEZO_DIVING: FineTileType.ENEMY,
    EnemyId.BEEZO_STRAIGHT: FineTileType.ENEMY,
    EnemyId.PIDGIT: FineTileType.ENEMY,
    EnemyId.TROUTER: FineTileType.ENEMY,
    EnemyId.HOOPSTAR: FineTileType.ENEMY,
    EnemyId.JAR_GENERATOR_SHYGUY: FineTileType.ENEMY,
    EnemyId.JAR_GENERATOR_BOB_OMB: FineTileType.ENEMY,
    EnemyId.PHANTO: FineTileType.ENEMY,
    EnemyId.COBRAT_JAR: FineTileType.ENEMY,
    EnemyId.COBRAT_SAND: FineTileType.ENEMY,
    EnemyId.POKEY: FineTileType.ENEMY,
    EnemyId.BIRDO: FineTileType.ENEMY,
    EnemyId.MOUSER: FineTileType.ENEMY,
    EnemyId.TRYCLYDE: FineTileType.ENEMY,
    EnemyId.CLAWGRIP: FineTileType.ENEMY,
    EnemyId.PANSER_STATIONARY_FIRES_ANGLED: FineTileType.ENEMY,
    EnemyId.PANSER_WALKING: FineTileType.ENEMY,
    EnemyId.PANSER_STATIONARY_FIRES_UP: FineTileType.ENEMY,
    EnemyId.AUTOBOMB: FineTileType.ENEMY,
    EnemyId.FLURRY: FineTileType.ENEMY,
    EnemyId.FRYGUY: FineTileType.ENEMY,
    EnemyId.FRYGUY_SPLIT: FineTileType.ENEMY,
    EnemyId.WART: FineTileType.ENEMY,
    EnemyId.HAWKMOUTH_BOSS: FineTileType.ENEMY,
    EnemyId.VEGETABLE_THROWER: FineTileType.ENEMY,
    # ---- PROJECTILE - hostile, cannot be defeated, so worth distinguishing ----
    EnemyId.WART_BUBBLE: FineTileType.PROJECTILE,
    EnemyId.BULLET: FineTileType.PROJECTILE,
    EnemyId.EGG: FineTileType.PROJECTILE,  # thrown by Birdo; becomes ridable
    EnemyId.FIREBALL: FineTileType.PROJECTILE,
    EnemyId.CLAWGRIP_ROCK: FineTileType.PROJECTILE,
    EnemyId.AUTOBOMB_FIRE: FineTileType.PROJECTILE,
    EnemyId.WHALE_SPOUT: FineTileType.PROJECTILE,
    EnemyId.SPARK1: FineTileType.PROJECTILE,
    EnemyId.SPARK2: FineTileType.PROJECTILE,
    EnemyId.SPARK3: FineTileType.PROJECTILE,
    EnemyId.SPARK4: FineTileType.PROJECTILE,
    # ---- Control/spawner pseudo-objects: nothing is drawn, so nothing to mark ----
    EnemyId.ATTACK_ALBATOSS_CARRYING_BOB_OMB: FineTileType.EMPTY,
    EnemyId.ATTACK_BEEZO: FineTileType.EMPTY,
    EnemyId.STOP_ATTACK: FineTileType.EMPTY,
}

# The BOSS_* range (0x5C-0x7F) mirrors the regular range (0x1C-0x3F) exactly --
# same objects, boss-room variants -- so derive it rather than duplicating it.
_BOSS_RANGE_OFFSET = 0x40
for _object_id, _fine_type in list(OBJECT_ID_MAPPING.items()):
    if 0x1C <= _object_id <= 0x3F:
        OBJECT_ID_MAPPING[_object_id + _BOSS_RANGE_OFFSET] = _fine_type

# Unrecognised sprites (the UNKNOWN_* ids) default to ENEMY: treating an unknown
# hazard as harmless is the more dangerous failure, so it is not in this dict and
# callers fall back to ENEMY.
UNMAPPED_OBJECT_FINE_TYPE = FineTileType.ENEMY

# Fine types whose objects hurt the player on contact. Derived from the semantic
# classification rather than from SpriteFlags: the DAMAGE_FROM_ABOVE bit only
# marks the narrower "hurts even when stomped" case, so it under-reports.
DAMAGING_FINE_TYPES: frozenset[FineTileType] = frozenset(
    {
        FineTileType.ENEMY,
        FineTileType.PROJECTILE,
    }
)

# Fine types the player can pick up and throw. The UNLIFTABLE sprite flag refines
# this at runtime -- some otherwise-liftable objects are pinned -- so the flag
# takes precedence when set.
LIFTABLE_FINE_TYPES: frozenset[FineTileType] = frozenset(
    {
        FineTileType.VEGETABLE,
        FineTileType.BOMB,
        FineTileType.POW_BLOCK,
        FineTileType.KEY,
        FineTileType.POTION,
        FineTileType.MUSHROOM,
        FineTileType.SHELL,
    }
)

# Fine types that are never larger than a single tile. Object footprints are
# measured from the sprites being drawn, which cannot separate two objects at the
# same position -- a Cobrat and the bullet it just spat share an anchor, so the
# bullet would otherwise inherit the Cobrat's 1x2 sprite block. These types are
# small by definition, so their footprint is capped rather than measured.
SINGLE_TILE_FINE_TYPES: frozenset[FineTileType] = frozenset(
    {
        FineTileType.PROJECTILE,
        FineTileType.COIN,
        FineTileType.CHERRY,
        FineTileType.HEART,
    }
)
