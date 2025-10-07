"""Player state properties for SMB2 environment."""

from .base import GameStateMixin
from ..constants import (
    CHARACTER,
    CHARACTER_NAMES,
    CHERRIES,
    CONTINUES,
    CURRENT_LEVEL,
    ITEM_HOLDING,
    ITEM_PULLED,
    LEVEL_NAMES,
    LEVEL_TRANSITION,
    LEVELS_FINISHED_LUIGI,
    LEVELS_FINISHED_MARIO,
    LEVELS_FINISHED_PEACH,
    LEVELS_FINISHED_TOAD,
    LIFE_METER,
    LIVES,
    MAX_CHERRIES,
    MAX_COINS,
    MAX_CONTINUES,
    MAX_HEARTS,
    MAX_LIVES,
    ON_VINE,
    PLAYER_SPEED,
    SUBSPACE_COINS,
    SUBSPACE_STATUS,
    VEGETABLES_PULLED,
)


class PlayerStateMixin(GameStateMixin):
    """Mixin class providing player state properties for SMB2 environment."""

    @property
    def lives(self) -> int:
        """Get current lives."""
        lives = self._read_ram_safe(LIVES, default=2)
        # Validate the value - SMB2 has 2-5 lives typically
        if 0 <= lives <= MAX_LIVES:
            return lives
        return 2  # Default if invalid

    @property
    def character(self) -> int:
        """Get selected character (0=Mario, 1=Princess, 2=Toad, 3=Luigi)."""
        char = self._read_ram_safe(CHARACTER, default=0)
        if 0 <= char <= 3:
            return char
        return 0

    @property
    def hearts(self) -> int:
        """Get current hearts (1-4)."""
        life_meter = self._read_ram_safe(LIFE_METER, default=0x1F)  # Default 2 hearts
        # Convert: 0x0F=1, 0x1F=2, 0x2F=3, 0x3F=4
        # The upper nibble indicates hearts - 1
        if life_meter == 0x0F:
            return 1
        elif life_meter == 0x1F:
            return 2
        elif life_meter == 0x2F:
            return 3
        elif life_meter == 0x3F:
            return 4
        else:
            # If value doesn't match expected pattern, use upper nibble + 1
            hearts = ((life_meter & 0xF0) >> 4) + 1
            if 1 <= hearts <= MAX_HEARTS:
                return hearts
            return 2  # Default

    @property
    def cherries(self) -> int:
        """Get cherries collected."""
        cherries = self._read_ram_safe(CHERRIES, default=0)
        if 0 <= cherries <= MAX_CHERRIES:
            return cherries
        return 0

    @property
    def coins(self) -> int:
        """Get coins collected in Subspace."""
        coins = self._read_ram_safe(SUBSPACE_COINS, default=0)
        if 0 <= coins <= MAX_COINS:
            return coins
        return 0

    @property
    def holding_item(self) -> bool:
        """Check if character is holding an item."""
        return self._read_ram_safe(ITEM_HOLDING, default=0) == 1

    @property
    def item_pulled(self) -> int:
        """Get item pulled from ground."""
        return self._read_ram_safe(ITEM_PULLED, default=0)

    @property
    def continues(self) -> int:
        """Get number of continues."""
        continues = self._read_ram_safe(CONTINUES, default=0)
        if 0 <= continues <= MAX_CONTINUES:
            return continues
        return 0

    @property
    def player_speed(self) -> int:
        """Get player horizontal speed (signed: positive=right, negative=left)."""
        speed = self._read_ram_safe(PLAYER_SPEED, default=0)
        return speed if speed < 128 else speed - 256

    @property
    def on_vine(self) -> bool:
        """Check if character is on a vine."""
        return self._read_ram_safe(ON_VINE, default=0) == 1

    @property
    def levels_finished(self) -> dict[str, int]:
        """Get levels finished per character."""
        return {
            'mario': self._read_ram_safe(LEVELS_FINISHED_MARIO, default=0),
            'peach': self._read_ram_safe(LEVELS_FINISHED_PEACH, default=0),
            'toad': self._read_ram_safe(LEVELS_FINISHED_TOAD, default=0),
            'luigi': self._read_ram_safe(LEVELS_FINISHED_LUIGI, default=0),
        }

    @property
    def vegetables_pulled(self) -> int:
        """Get total vegetables pulled."""
        return self._read_ram_safe(VEGETABLES_PULLED, default=0)

    @property
    def subspace_status(self) -> int:
        """Get subspace status (0=not in subspace, 2=in subspace)."""
        return self._read_ram_safe(SUBSPACE_STATUS, default=0)

    @property
    def level(self) -> str:
        """Get current level string."""
        level_id = self._read_ram_safe(CURRENT_LEVEL, default=0)
        return LEVEL_NAMES.get(level_id, f"L-{level_id:02X}")

    @property
    def level_transition(self) -> int:
        """Get level transition state.

        NOTE: This value at 0x04EC appears to change for less than a frame.
        The game sets it to non-zero and immediately clears it back to 0
        within the same frame's CPU execution (as seen in disassembly at
        $E66D: STA $04EC). Therefore, we cannot reliably detect transitions
        by polling this value once per frame.
        For reliable level completion detection, use the increase in
        'levels_finished' counter instead.

        Values (theoretical):
        0 - normal gameplay
        1 - restart same level
        2 - game over
        3 - end level, go to bonus game (level completed)
        4 - warp
        """
        # This will almost always return 0 due to sub-frame clearing? TODO: Delete
        return self._read_ram_safe(LEVEL_TRANSITION, default=0)

    @property
    def level_completed(self) -> bool:
        """Detect if a level was just completed.

        Returns True if any character's levels_finished counter has increased
        since the last step.
        """
        if not hasattr(self, '_previous_levels_finished'):
            return False

        current_levels_finished = self.levels_finished
        for char_name in ['mario', 'peach', 'toad', 'luigi']:
            if current_levels_finished[char_name] > self._previous_levels_finished.get(
                char_name, 0
            ):
                return True
        return False