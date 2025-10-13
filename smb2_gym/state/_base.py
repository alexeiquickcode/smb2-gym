"""Base class for game state mixins."""

from abc import (
    ABC,
    abstractmethod,
)
from typing import Optional


class GameStateMixin(ABC):
    """Abstract base class for game state mixins.

    Defines the interface that all game state mixins expect to be available
    from the main environment class.
    """

    @abstractmethod
    def _read_ram_safe(self, address: int, default: int = 0) -> int:
        """Safely read from RAM with fallback.

        Args:
            address: RAM address to read
            default: Default value if RAM reading is not available

        Returns:
            Value at RAM address or default
        """
        pass

    @abstractmethod
    def _get_y_position(self, address: int) -> int:
        """Safely read Y position from RAM and clamp to valid range.

        Args:
            address: RAM address to read Y position from

        Returns:
            Y position clamped to 0-239 range
        """
        pass

    @abstractmethod
    def _read_ppu(self, address: int, default: int = 0) -> int:
        """Safely read from PPU memory with fallback.

        Args:
            address: PPU address to read (0x0000-0x3FFF)
            default: Default value if PPU reading is not available

        Returns:
            Value at PPU address or default
        """
        pass

    # Attributes that mixins expect to exist
    # These are typically set in _init_state_tracking() in the main class
    AREA_TRANSITION_FRAMES: int
    _previous_sub_area: Optional[int]
    _previous_x_global: Optional[int]
    _previous_y_global: Optional[int]
    _transition_frame_count: int
    _previous_levels_finished: Optional[dict[str, int]]

