"""Enemy-related properties for SMB2 environment."""

from ..constants import (
    ENEMY_SLOTS,
    PLAYER,
    SCREEN_HEIGHT,
    Enemy,
    EnemyState,
)
from ._base import GameStateMixin


class EnemiesMixin(GameStateMixin):
    """Mixin class providing enemy-related properties for SMB2 environment."""

    @property
    def enemies_defeated(self) -> int:
        """Get count of enemies defeated (for heart spawning)."""
        return self._read_ram_safe(PLAYER.ENEMIES_DEFEATED)

    @property
    def enemies(self) -> list[Enemy]:
        """Get all 9 enemy slots with their current runtime data.

        Returns:
            List of 9 Enemy objects (index 0-8 = slots 0-8)
            Invisible/dead slots have None for most fields except state
        """
        enemies_data = []
        for slot in ENEMY_SLOTS:
            state = self._read_ram_safe(slot.state)

            if state in [EnemyState.INVISIBLE, EnemyState.DEAD]:
                enemies_data.append(
                    Enemy(
                        slot_number=slot.slot_number,
                        x_position=None,
                        y_position=None,
                        x_page=None,
                        y_page=None,
                        object_type=None,
                        health=None,
                        state=state,
                        x_velocity=None,
                        y_velocity=None,
                        direction=None,
                        collision=None,
                        object_timer=None,
                        sprite_flags=None
                    )
                )
            else:
                # Read Y position and invert it (y=0 at bottom)
                y_pos_raw = self._read_ram_safe(slot.y_position)
                y_pos_inverted = SCREEN_HEIGHT - 1 - y_pos_raw

                # Read velocities and convert to signed
                x_vel_raw = self._read_ram_safe(slot.x_velocity)
                x_vel_signed = x_vel_raw if x_vel_raw < 128 else x_vel_raw - 256
                y_vel_raw = self._read_ram_safe(slot.y_velocity)
                y_vel_signed = y_vel_raw if y_vel_raw < 128 else y_vel_raw - 256

                enemy = Enemy(
                    slot_number=slot.slot_number,
                    x_position=self._read_ram_safe(slot.x_position),
                    y_position=y_pos_inverted,
                    x_page=self._read_ram_safe(slot.x_page),
                    y_page=self._read_ram_safe(slot.y_page),
                    object_type=self._read_ram_safe(slot.object_type),
                    health=self._read_ram_safe(slot.health),
                    state=state,
                    x_velocity=x_vel_signed,
                    y_velocity=y_vel_signed,
                    direction=self._read_ram_safe(slot.direction),
                    collision=self._read_ram_safe(slot.collision),
                    object_timer=self._read_ram_safe(slot.object_timer),
                    sprite_flags=self._read_ram_safe(slot.sprite_flags)
                )
                enemies_data.append(enemy)
        return enemies_data
