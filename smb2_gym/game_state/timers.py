"""Timer-related properties for SMB2 environment."""

from .base import GameStateMixin
from ..constants import (
    BOB_OMB_1_TIMER,
    BOB_OMB_2_TIMER,
    BOB_OMB_3_TIMER,
    BOB_OMB_4_TIMER,
    BOB_OMB_5_TIMER,
    BOMB_1_TIMER,
    BOMB_2_TIMER,
    DOOR_TRANSITION_TIMER,
    FLOAT_TIMER,
    FRAMERULE_TIMER,
    INVULNERABILITY_TIMER,
    PIDGET_CARPET_TIMER,
    STARMAN_TIMER,
    STOPWATCH_TIMER,
    SUBSPACE_TIMER,
    UNKNOWN_TIMER,
)


class TimersMixin(GameStateMixin):
    """Mixin class providing timer-related properties for SMB2 environment."""

    @property
    def starman_timer(self) -> int:
        """Get starman timer."""
        return self._read_ram_safe(STARMAN_TIMER, default=0)

    @property
    def subspace_timer(self) -> int:
        """Get subspace timer."""
        return self._read_ram_safe(SUBSPACE_TIMER, default=0)

    @property
    def stopwatch_timer(self) -> int:
        """Get stopwatch timer."""
        return self._read_ram_safe(STOPWATCH_TIMER, default=0)

    @property
    def invulnerability_timer(self) -> int:
        """Get invulnerability timer (time left until character becomes vulnerable)."""
        return self._read_ram_safe(INVULNERABILITY_TIMER, default=0)

    @property
    def framerule_timer(self) -> int:
        """Get general 256 frames framerule counter."""
        return self._read_ram_safe(FRAMERULE_TIMER, default=0)

    @property
    def unknown_timer(self) -> int:
        """Get unknown timer."""
        return self._read_ram_safe(UNKNOWN_TIMER, default=0)

    @property
    def bob_omb_5_timer(self) -> int:
        """Get time left before Bob Omb 5 explodes."""
        return self._read_ram_safe(BOB_OMB_5_TIMER, default=0)

    @property
    def bob_omb_4_timer(self) -> int:
        """Get time left before Bob Omb 4 explodes."""
        return self._read_ram_safe(BOB_OMB_4_TIMER, default=0)

    @property
    def bob_omb_3_timer(self) -> int:
        """Get time left before Bob Omb 3 explodes."""
        return self._read_ram_safe(BOB_OMB_3_TIMER, default=0)

    @property
    def bob_omb_2_timer(self) -> int:
        """Get time left before Bob Omb 2 explodes."""
        return self._read_ram_safe(BOB_OMB_2_TIMER, default=0)

    @property
    def bob_omb_1_timer(self) -> int:
        """Get time left before Bob Omb 1 explodes."""
        return self._read_ram_safe(BOB_OMB_1_TIMER, default=0)

    @property
    def bomb_1_timer(self) -> int:
        """Get time left before Bomb 1 explodes."""
        return self._read_ram_safe(BOMB_1_TIMER, default=0)

    @property
    def bomb_2_timer(self) -> int:
        """Get time left before Bomb 2 explodes."""
        return self._read_ram_safe(BOMB_2_TIMER, default=0)

    @property
    def pidget_carpet_timer(self) -> int:
        """Get time left to use Pidget's carpet."""
        return self._read_ram_safe(PIDGET_CARPET_TIMER, default=0)

    @property
    def float_timer(self) -> int:
        """Get Princess float timer (available float time, max 60 frames = 1 second)."""
        return self._read_ram_safe(FLOAT_TIMER, default=0)

    @property
    def door_transition_timer(self) -> int:
        """Get door transition timer."""
        return self._read_ram_safe(DOOR_TRANSITION_TIMER, default=0)