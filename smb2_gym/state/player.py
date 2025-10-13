"""Player state properties for SMB2 environment."""

from ..constants import (
    GAME_STATE,
    MAX_CHERRIES,
    MAX_COINS,
    MAX_CONTINUES,
    MAX_HEARTS,
    MAX_LIVES,
    PLAYER,
    TIMERS,
)
from ..constants.character_stats import get_character_stats
from ._base import GameStateMixin


class PlayerStateMixin(GameStateMixin):
    """Mixin class providing player state helper properties for SMB2 environment."""

    @property
    def lives(self) -> int:
        """Get current lives."""
        lives = self._read_ram_safe(PLAYER.LIVES, default=2)
        if 0 <= lives <= MAX_LIVES:
            return lives
        return 2  # Default if invalid

    @property
    def character(self) -> int:
        """Get selected character (0=Mario, 1=Princess, 2=Toad, 3=Luigi)."""
        char = self._read_ram_safe(GAME_STATE.CHARACTER, default=0)
        if 0 <= char <= 3:
            return char
        return 0

    @property
    def hearts(self) -> int:
        """Get current hearts (1-4)."""
        life_meter = self._read_ram_safe(PLAYER.LIFE_METER, default=0x1F)
        hearts = ((life_meter & 0xF0) >> 4) + 1
        if 1 <= hearts <= MAX_HEARTS:
            return hearts
        return 2  # Default

    @property
    def cherries(self) -> int:
        """Get cherries collected."""
        cherries = self._read_ram_safe(PLAYER.CHERRIES, default=0)
        if 0 <= cherries <= MAX_CHERRIES:
            return cherries
        return 0

    @property
    def coins(self) -> int:
        """Get coins collected in Subspace."""
        coins = self._read_ram_safe(PLAYER.SUBSPACE_COINS, default=0)
        if 0 <= coins <= MAX_COINS:
            return coins
        return 0

    @property
    def holding_item(self) -> bool:
        """Check if character is holding an item."""
        return self._read_ram_safe(PLAYER.HOLDING_ITEM, default=0) == 1

    @property
    def item_pulled(self) -> int:
        """Get item pulled from ground."""
        return self._read_ram_safe(PLAYER.ITEM_PULLED, default=0)

    @property
    def continues(self) -> int:
        """Get number of continues."""
        continues = self._read_ram_safe(PLAYER.CONTINUES, default=0)
        if 0 <= continues <= MAX_CONTINUES:
            return continues
        return 0

    @property
    def player_speed(self) -> int:
        """Get player horizontal speed (signed: positive=right, negative=left)."""
        speed = self._read_ram_safe(PLAYER.SPEED, default=0)
        return speed if speed < 128 else speed - 256

    @property
    def on_vine(self) -> bool:
        """Check if character is on a vine."""
        return self._read_ram_safe(PLAYER.ON_VINE, default=0) == 1

    @property
    def big_vegetables_pulled(self) -> int:
        """Get total big vegetables pulled (required for stopwatch)."""
        return self._read_ram_safe(PLAYER.BIG_VEGETABLES_PULLED, default=0)

    @property
    def subspace_status(self) -> int:
        """Get subspace status (0=not in subspace, 2=in subspace)."""
        return self._read_ram_safe(GAME_STATE.SUBSPACE_STATUS, default=0)

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
        return self._read_ram_safe(GAME_STATE.LEVEL_TRANSITION, default=0)

    @property
    def character_stats(self):
        """Get current character's statistics and abilities."""
        character_id = self._read_ram_safe(GAME_STATE.CHARACTER, default=0)
        return get_character_stats(character_id)

    @property
    def starman_timer(self) -> int:
        """Get starman timer."""
        return self._read_ram_safe(TIMERS.STARMAN, default=0)

    @property
    def subspace_timer(self) -> int:
        """Get subspace timer."""
        return self._read_ram_safe(TIMERS.SUBSPACE, default=0)

    @property
    def stopwatch_timer(self) -> int:
        """Get stopwatch timer."""
        return self._read_ram_safe(TIMERS.STOPWATCH, default=0)

    @property
    def invulnerability_timer(self) -> int:
        """Get invulnerability timer (time left until character becomes vulnerable)."""
        return self._read_ram_safe(TIMERS.INVULNERABILITY, default=0)

    @property
    def framerule_timer(self) -> int:
        """Get general 256 frames framerule counter."""
        return self._read_ram_safe(TIMERS.FRAMERULE, default=0)

    @property
    def pidget_carpet_timer(self) -> int:
        """Get time left to use Pidget's carpet."""
        return self._read_ram_safe(TIMERS.PIDGET_CARPET, default=0)

    @property
    def float_timer(self) -> int:
        """Get Princess float timer (available float time, max 60 frames = 1 second)."""
        return self._read_ram_safe(TIMERS.FLOAT, default=0)

    @property
    def door_transition_timer(self) -> int:
        """Get door transition timer."""
        return self._read_ram_safe(TIMERS.DOOR_TRANSITION, default=0)

    @property
    def player_state(self) -> int:
        """Get player state/animation."""
        return self._read_ram_safe(PLAYER.STATE, default=0)

    @property
    def levels_finished(self) -> dict[str, int]:
        """Get levels finished per character."""
        return {
            'mario': self._read_ram_safe(PLAYER.LEVELS_FINISHED_MARIO, default=0),
            'peach': self._read_ram_safe(PLAYER.LEVELS_FINISHED_PEACH, default=0),
            'toad': self._read_ram_safe(PLAYER.LEVELS_FINISHED_TOAD, default=0),
            'luigi': self._read_ram_safe(PLAYER.LEVELS_FINISHED_LUIGI, default=0),
        }

    @property
    def level_completed(self) -> bool:
        """Detect if a level was just completed."""
        if self._previous_levels_finished is None:
            return False

        current_levels_finished = self.levels_finished
        for char_name in ['mario', 'peach', 'toad', 'luigi']:
            if current_levels_finished[char_name] > self._previous_levels_finished.get(char_name, 0):
                return True
        return False

