"""App utilities for Super Mario Bros 2 gymnasium environment."""

import os
from typing import (
    Optional,
    Union,
)

from ..constants import (
    CHARACTER_NAMES,
    LEVEL_NAMES,
)
from .info_display import (
    create_info_panel,
    draw_info,
    get_required_info_height,
)


class InitConfig:
    """Configuration for initializing SMB2 environment.

    Supports two main modes:

    1. Built-in ROM mode (default):
       - Uses the included SMB2 ROM with pre-made save states
       - Specify level and character to start at that point
       - Example: InitConfig(level="1-1", character="luigi")

    2. Custom ROM mode:
       - Load your own ROM file and optional save state
       - Example: InitConfig(rom_path="/path/to/rom.nes", save_state_path="/path/to/save.sav")

    Note: The CLI supports additional syntax like --rom prg0_edited for built-in 
    ROM variants, which gets converted to custom ROM paths internally.
    """

    def __init__(
        self,
        level: Optional[str] = None,
        character: Optional[Union[str, int]] = None,
        rom_path: Optional[str] = None,
        save_state_path: Optional[str] = None,
    ):
        """Initialize configuration.

        Args:
            level: Level to play (e.g., "1-1", "7-2") 
            character: Character to play as (name or ID)
            rom_path: Path to custom ROM file
            save_state_path: Path to save state file (optional)
        """
        self.level = level or "1-1"
        self.character = character or "luigi"
        self.rom_path = rom_path
        self.save_state_path = save_state_path

        # Convert character name to ID if needed
        if isinstance(self.character, str):
            self.character_id = self._character_name_to_id(self.character)
        else:
            self.character_id = self.character

        # Validate level
        self.level_id = self._validate_level(self.level)

    def _character_name_to_id(self, name: str) -> int:
        """Convert character name to ID."""
        char_map = {v.lower(): k for k, v in CHARACTER_NAMES.items()}
        name_lower = name.lower()

        if name_lower not in char_map:
            valid_names = list(char_map.keys())
            raise ValueError(f"Invalid character '{name}'. Valid: {', '.join(valid_names)}")

        return char_map[name_lower]

    def _validate_level(self, level: str) -> int:
        """Validate and convert level string to level ID."""
        level_str_to_id = {v: k for k, v in LEVEL_NAMES.items()}

        if level not in level_str_to_id:
            valid_levels = sorted(LEVEL_NAMES.values())
            raise ValueError(f"Invalid level '{level}'. Valid: {', '.join(valid_levels)}")

        return level_str_to_id[level]

    def get_rom_path(self) -> str:
        """Get ROM file path."""
        if self.rom_path:
            if not os.path.exists(self.rom_path):
                raise FileNotFoundError(f"ROM file not found: {self.rom_path}")
            return os.path.abspath(self.rom_path)
        else:
            # Use built-in ROM
            base_dir = os.path.dirname(os.path.dirname(__file__))
            rom_path = os.path.join(base_dir, '_nes', 'prg0', 'super_mario_bros_2_prg0.nes')
            return os.path.abspath(rom_path)

    def get_save_state_path(self) -> Optional[str]:
        """Get save state file path."""
        if self.save_state_path:
            if not os.path.exists(self.save_state_path):
                raise FileNotFoundError(f"Save state not found: {self.save_state_path}")
            return os.path.abspath(self.save_state_path)
        elif not self.rom_path:
            # Use built-in save state for level/character mode
            base_dir = os.path.dirname(os.path.dirname(__file__))
            character_name = CHARACTER_NAMES[self.character_id].lower()
            save_path = os.path.join(
                base_dir, '_nes', 'prg0', 'saves', character_name, f'{self.level}.sav'
            )
            return os.path.abspath(save_path) if os.path.exists(save_path) else None
        else:
            return None

    def describe(self) -> str:
        """Get description of configuration."""
        if self.rom_path:
            desc = f"Custom ROM: {self.rom_path}"
            if self.save_state_path:
                desc += f"\nSave state: {self.save_state_path}"
            return desc
        else:
            char_name = CHARACTER_NAMES[self.character_id]
            return f"Playing as {char_name} on level {self.level}"


__all__ = [
    'InitConfig',
    'create_info_panel',
    'get_required_info_height',
    'draw_info',
]
