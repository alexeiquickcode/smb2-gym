"""
All Constants for SMB2 gym environment. Including RAM locations as well as 
additional constants used within this library. Many of these constants have been
pulled from:

https://datacrystal.tcrf.net/wiki/Super_Mario_Bros._2_(NES)/RAM_map
and 
https://github.com/Xkeeper0/smb2

However some fields in these classes do not directly align with the above 
references and there have been some additional fields added.
"""

from .character_stats import (
    CharacterStats,
    get_character_stats,
)
from .object_ids import (
    BackgroundTile,
    CollisionFlags,
    EnemyId,
    EnemyState,
    PlayerState,
    SpriteFlags,
)
from .ram import *
from .semantic import (
    COARSE_LOOKUP,
    COARSE_TENSOR_CHANNEL_NAMES,
    COARSE_TENSOR_CHANNELS,
    COARSE_TILE_NAMES,
    COLOR_LOOKUP,
    DAMAGING_FINE_TYPES,
    FINE_TENSOR_CHANNELS,
    FINE_TILE_NAMES,
    FINE_TO_COARSE_MAPPING,
    LIFTABLE_FINE_TYPES,
    NO_OBJECT,
    OBJECT_ID_MAPPING,
    PROPERTY_CHANNEL_NAMES,
    RENDER_PRIORITY,
    SEMANTIC_TILE_DTYPE,
    TILE_COLORS,
    TILE_ID_MAPPING,
    UNMAPPED_OBJECT_FINE_TYPE,
    VELOCITY_CHANNEL_NAMES,
    CoarseTileType,
    FineTileType,
)


__all__ = [
    'BackgroundTile',
    'CollisionFlags',
    'EnemyId',
    'EnemyState',
    'PlayerState',
    'SpriteFlags',
    'CharacterStats',
    'get_character_stats',
    'CoarseTileType',
    'FineTileType',
    'FINE_TO_COARSE_MAPPING',
    'FINE_TILE_NAMES',
    'COARSE_TILE_NAMES',
    'TILE_COLORS',
    'SEMANTIC_TILE_DTYPE',
    'TILE_ID_MAPPING',
    'OBJECT_ID_MAPPING',
    'UNMAPPED_OBJECT_FINE_TYPE',
    'NO_OBJECT',
    'COARSE_TENSOR_CHANNELS',
    'FINE_TENSOR_CHANNELS',
    'COARSE_TENSOR_CHANNEL_NAMES',
    'PROPERTY_CHANNEL_NAMES',
    'VELOCITY_CHANNEL_NAMES',
    'DAMAGING_FINE_TYPES',
    'LIFTABLE_FINE_TYPES',
    'RENDER_PRIORITY',
    'COARSE_LOOKUP',
    'COLOR_LOOKUP',
]
