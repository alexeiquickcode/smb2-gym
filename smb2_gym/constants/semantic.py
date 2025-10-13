"""Semantic tile definitions for SMB2 environment.

This module contains all semantic tile types, categories, colors, and mappings
used for the semantic map observation space.
"""

from enum import IntEnum

import numpy as np

from .object_ids import BackgroundTile

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

    # ENEMY category - dynamic threats (from enemy slots in RAM)
    ENEMY = 15

    # HAZARD category - environmental dangers
    SPIKES = 16
    QUICKSAND = 17
    CONVEYOR_LEFT = 18
    CONVEYOR_RIGHT = 19


# RGB colors for tile types
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

    # PICKABLE - distinct colors for each type
    FineTileType.VEGETABLE: (50, 205, 50),  # lime green
    FineTileType.BOMB: (128, 0, 0),  # maroon
    FineTileType.POW_BLOCK: (255, 255, 224),  # light yellow
    FineTileType.KEY: (255, 215, 0),  # gold

    # COLLECTIBLE - bright colors
    FineTileType.CHERRY: (255, 20, 147),  # deep pink
    FineTileType.POTION: (186, 85, 211),  # medium orchid
    FineTileType.MUSHROOM: (255, 140, 0),  # dark orange

    # ENEMY - red for threats
    FineTileType.ENEMY: (255, 0, 0),  # red

    # HAZARD - reds/oranges for danger
    FineTileType.SPIKES: (220, 20, 60),  # crimson
    FineTileType.QUICKSAND: (244, 164, 96),  # sandy brown
    FineTileType.CONVEYOR_LEFT: (105, 105, 105),  # dim gray
    FineTileType.CONVEYOR_RIGHT: (105, 105, 105),  # dim gray
}

# ------------------------------------------------------------------------------
# ---- Lookup Tables & Dtypes --------------------------------------------------
# ------------------------------------------------------------------------------

# Human-readable names derived from enum names
FINE_TILE_NAMES: dict[FineTileType, str] = {tile_type: tile_type.name for tile_type in FineTileType}
COARSE_TILE_NAMES: dict[CoarseTileType, str] = {
    category: category.name for category in CoarseTileType
}

# Structured dtype for semantic map tiles with hierarchical information
SEMANTIC_TILE_DTYPE = np.dtype(
    [
        ('tile_id', np.uint8),  # Raw game object ID (BackgroundTile/EnemyId value)
        ('fine_type', np.uint8),  # Fine-grained FineTileType (SOLID, ENEMY, etc.)
        ('coarse_type', np.uint8),  # Coarse-grained CoarseTileType (TERRAIN, ENEMY, etc.)
        ('color_r', np.uint8),  # RGB color for visualization
        ('color_g', np.uint8),
        ('color_b', np.uint8),
    ]
)

# Pre-computed lookup tables for fast vectorized operations
COARSE_LOOKUP = np.zeros(max(FineTileType) + 1, dtype=np.uint8)
COLOR_LOOKUP = np.zeros((max(FineTileType) + 1, 3), dtype=np.uint8)

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

    # COLLECTIBLE
    FineTileType.CHERRY: CoarseTileType.COLLECTIBLE,

    # ENEMY
    FineTileType.ENEMY: CoarseTileType.ENEMY,

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
TILE_ID_MAPPING: dict[int, FineTileType] = { \
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
