"""Base class for game state mixins."""

from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Optional,
    Protocol,
)

from ..constants import Enemy


class GameStateMixin(ABC):
    """Abstract base class for game state mixins.

    Defines the interface that all game state mixins expect to be available
    from the main environment class.
    """

    # Attributes that mixins expect to exist
    AREA_TRANSITION_FRAMES: int

    # Tracking variables
    _previous_sub_area: Optional[int]
    _previous_x_global: Optional[int]
    _previous_y_global: Optional[int]
    _transition_frame_count: int
    _previous_levels_finished: Optional[dict[str, int]]

    @abstractmethod
    def _read_ram_safe(self, address: int) -> int:
        """Read from RAM.

        Args:
            address: RAM address to read

        Returns:
            Value at RAM address
        """
        pass

    @abstractmethod
    def _read_ppu(self, address: int) -> int:
        """Read from PPU memory.

        Args:
            address: PPU address to read (0x0000-0x3FFF)

        Returns:
            Value at PPU address
        """
        pass


class HasEnemies(Protocol):
    """Protocol for classes that provide enemy tracking.

    This protocol defines the interface that SemanticMapMixin expects
    to be provided by EnemiesMixin.
    """

    @property
    def enemies(self) -> list[Enemy]:
        """Get all enemy slots with their current runtime data."""
        ...
