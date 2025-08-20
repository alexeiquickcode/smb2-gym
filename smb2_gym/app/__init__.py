"""App utilities for Super Mario Bros 2 gymnasium environment."""

import os
from typing import (
    Any,
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

    Supports three initialization modes:
    1. Character/Level mode: Specify character and level to load from prg0 ROM
    2. Built-in ROM mode: Specify ROM variant and save state file
    3. Custom ROM mode: Specify custom ROM and save state file paths
    """

    def __init__(
        self,
        level: Optional[str] = None,
        character: Optional[Union[str, int]] = None,
        rom: Optional[str] = None,
        save_state: Optional[str] = None,
        rom_path: Optional[str] = None,
        save_state_path: Optional[str] = None,
    ):
        """Initialize configuration.

        Args:
            level: Level to play (e.g., "1-1", "7-2")
            character: Character to play as (name or ID)
            rom: ROM variant ("prg0" or "prg0_edited")
            save_state: Save state file name
            rom_path: Custom ROM file path
            save_state_path: Custom save state file path
        """
        self.level = level
        self.character = character
        self.rom = rom
        self.save_state = save_state
        self.rom_path = rom_path
        self.save_state_path = save_state_path

        # These will be set during validation
        self.level_id: Optional[int] = None
        self.character_id: Optional[int] = None
        self.rom_variant: Optional[str] = None
        self.mode: str  # Will be set by _detect_mode

        # Detect and validate mode
        self.mode = self._detect_mode()
        self._validate_mode_requirements()
        self._set_defaults()
        self._process_parameters()

    def _detect_mode(self) -> str:
        """Detect which initialization mode is being used.

        Returns:
            Mode name: "character_level", "built_in_rom", "custom_rom", or "default"

        Raises:
            ValueError: If multiple modes are specified
        """
        mode_counts = [
            bool(self.level or self.character),  # Character/Level mode
            bool(self.rom or self.save_state),  # Built-in ROM mode
            bool(self.rom_path or self.save_state_path)  # Custom ROM mode
        ]

        if sum(mode_counts) > 1:
            raise ValueError(
                "Cannot specify multiple initialization modes. Use only one of: "
                "(level, character), (rom, save_state), or (rom_path, save_state_path)."
            )

        if self.level or self.character:
            return "character_level"
        elif self.rom or self.save_state:
            return "built_in_rom"
        elif self.rom_path or self.save_state_path:
            return "custom_rom"
        else:
            return "default"

    def _validate_mode_requirements(self) -> None:
        """Validate that required parameters are provided for each mode.

        Raises:
            ValueError: If required parameters are missing
        """
        if self.mode == "built_in_rom":
            if not self.rom or not self.save_state:
                raise ValueError("Built-in ROM mode requires both 'rom' and 'save_state'")
        elif self.mode == "custom_rom":
            if not self.rom_path or not self.save_state_path:
                raise ValueError("Custom ROM mode requires both 'rom_path' and 'save_state_path'")
            if not os.path.exists(self.rom_path):
                raise FileNotFoundError(f"Custom ROM file not found: {self.rom_path}")
            if not os.path.exists(self.save_state_path):
                raise FileNotFoundError(f"Custom save state file not found: {self.save_state_path}")

    def _set_defaults(self) -> None:
        """Set default values based on mode."""
        if self.mode == "character_level" or self.mode == "default":
            if self.level is None:
                self.level = "1-1"
            if self.character is None:
                self.character = "luigi"

        if self.mode == "default":
            # Default mode uses character/level with prg0
            self.mode = "character_level"

    def _process_parameters(self) -> None:
        """Process and validate parameters based on mode."""
        if self.mode == "character_level":
            # Validate level (guaranteed to be non-None after _set_defaults)
            assert self.level is not None
            self.level_id = validate_level(self.level)

            # Convert character name to ID if needed (guaranteed to be non-None after _set_defaults)
            assert self.character is not None
            if isinstance(self.character, str):
                self.character_id = character_name_to_id(self.character)
            else:
                self.character_id = validate_character(self.character)

            # Set ROM variant for character/level mode
            self.rom_variant = "prg0"

        elif self.mode == "built_in_rom":
            # Validate ROM variant (guaranteed to be non-None after validation)
            assert self.rom is not None
            self.rom_variant = validate_rom(self.rom)

        elif self.mode == "custom_rom":
            # Convert to absolute paths (guaranteed to be non-None after validation)
            assert self.rom_path is not None and self.save_state_path is not None
            self.rom_path = os.path.abspath(self.rom_path)
            self.save_state_path = os.path.abspath(self.save_state_path)
            self.rom_variant = None

    def get_rom_path(self) -> str:
        """Get the ROM file path based on mode.

        Returns:
            Absolute path to ROM file
        """
        if self.mode == "custom_rom":
            assert self.rom_path is not None  # Validated in _process_parameters
            return self.rom_path
        else:
            # Built-in ROM
            if self.rom_variant == "prg0":
                rom_filename = 'super_mario_bros_2_prg0.nes'
            elif self.rom_variant == "prg0_edited":
                rom_filename = 'super_mario_bros_2_prg0_edited.nes'
            else:
                raise ValueError(f"Unknown ROM variant: {self.rom_variant}")

            # Go up one directory from app to smb2_gym
            base_dir = os.path.dirname(os.path.dirname(__file__))
            rom_path = os.path.join(base_dir, '_nes', self.rom_variant, rom_filename)
            return os.path.abspath(rom_path)

    def get_save_state_path(self) -> Optional[str]:
        """Get the save state file path based on mode.

        Returns:
            Absolute path to save state file, or None if not using save states
        """
        # Go up one directory from app to smb2_gym
        base_dir = os.path.dirname(os.path.dirname(__file__))

        if self.mode == "custom_rom":
            assert self.save_state_path is not None  # Validated in _process_parameters
            return self.save_state_path
        elif self.mode == "built_in_rom":
            assert self.rom_variant is not None and self.save_state is not None  # Validated in _process_parameters
            save_path = os.path.join(base_dir, '_nes', self.rom_variant, 'saves', self.save_state)
            return os.path.abspath(save_path)
        elif self.mode == "character_level":
            if self.character_id is not None:
                character_name = CHARACTER_NAMES[self.character_id].lower()
                level_filename = f'{self.level}.sav'
                assert self.rom_variant is not None  # Set in _process_parameters
                save_path = os.path.join(
                    base_dir, '_nes', self.rom_variant, 'saves', character_name, level_filename
                )
                return os.path.abspath(save_path)
        return None


    def describe(self) -> str:
        """Get human-readable description of configuration.

        Returns:
            Description string
        """
        if self.mode == "custom_rom":
            return f"Using custom ROM: {self.rom_path}\nUsing custom save state: {self.save_state_path}"
        elif self.mode == "built_in_rom":
            return f"Using ROM variant: {self.rom}\nUsing save state: {self.save_state}"
        else:  # character_level
            if self.character_id is not None:
                char_name = CHARACTER_NAMES[self.character_id]
                return f"Playing as {char_name} on level {self.level}"
            return f"Playing on level {self.level}"


def validate_level(level: str) -> int:
    """Validate and convert level string to level ID.

    Args:
        level: Level string like "1-1" or "7-2"

    Returns:
        Level ID for RAM

    Raises:
        ValueError: If level is invalid
    """
    # Reverse mapping
    level_str_to_id = {v: k for k, v in LEVEL_NAMES.items()}

    if level not in level_str_to_id:
        valid_levels = sorted(LEVEL_NAMES.values())
        raise ValueError(f"Invalid level '{level}'. Valid levels are: {', '.join(valid_levels)}")

    return level_str_to_id[level]


def validate_character(character: int) -> int:
    """Validate character ID.

    Args:
        character: Character ID

    Returns:
        Validated character ID

    Raises:
        ValueError: If character is invalid
    """
    if character not in [0, 1, 2, 3]:
        raise ValueError(
            f"Invalid character {character}. Must be 0-3 (0=Mario, 1=Princess, 2=Toad, 3=Luigi)"
        )
    return character


def validate_rom(rom: str) -> str:
    """Validate ROM variant.

    Args:
        rom: ROM variant name

    Returns:
        Validated ROM variant

    Raises:
        ValueError: If ROM variant is invalid
    """
    valid_roms = ["prg0", "prg0_edited"]
    if rom not in valid_roms:
        raise ValueError(
            f"Invalid ROM variant '{rom}'. Valid variants are: {', '.join(valid_roms)}"
        )

    # Check if ROM directory exists
    base_dir = os.path.dirname(os.path.dirname(__file__))
    rom_dir = os.path.join(base_dir, '_nes', rom)
    if not os.path.exists(rom_dir):
        raise FileNotFoundError(f"ROM directory not found: {rom_dir}")

    return rom


def character_name_to_id(name: str) -> int:
    """Convert character name to ID.

    Args:
        name: Character name (case-insensitive)

    Returns:
        Character ID

    Raises:
        ValueError: If character name is invalid
    """
    char_map = {v.lower(): k for k, v in CHARACTER_NAMES.items()}
    name_lower = name.lower()

    # Handle common aliases
    if name_lower == "princess":
        name_lower = "peach"

    if name_lower not in char_map:
        valid_names = list(char_map.keys())
        raise ValueError(
            f"Invalid character name '{name}'. Valid names are: {', '.join(valid_names)}"
        )

    return char_map[name_lower]


def character_id_to_name(char_id: int) -> str:
    """Convert character ID to name.

    Args:
        char_id: Character ID

    Returns:
        Character name

    Raises:
        ValueError: If character ID is invalid
    """
    if char_id not in CHARACTER_NAMES:
        raise ValueError(f"Invalid character ID {char_id}")
    return CHARACTER_NAMES[char_id]


__all__ = [
    'InitConfig',
    'validate_level',
    'validate_character',
    'validate_rom',
    'character_name_to_id',
    'character_id_to_name',
    'create_info_panel',
    'get_required_info_height',
    'draw_info',
]
