"""Enemy-related properties for SMB2 environment."""

from .base import GameStateMixin
from ..constants import (
    ENEMIES_DEFEATED,
    ENEMY_DEAD,
    ENEMY_HEALTH,
    ENEMY_INVISIBLE,
    ENEMY_NOT_PRESENT,
    ENEMY_SPEED,
    ENEMY_VISIBILITY,
    ENEMY_X_PAGE,
    ENEMY_X_POS,
    ENEMY_Y_PAGE,
    ENEMY_Y_POS,
    PAGE_SIZE,
    SCREEN_HEIGHT,
)


class EnemiesMixin(GameStateMixin):
    """Mixin class providing enemy-related properties for SMB2 environment."""

    @property
    def enemies_defeated(self) -> int:
        """Get count of enemies defeated (for heart spawning)."""
        return self._read_ram_safe(ENEMIES_DEFEATED, default=0)

    @property
    def enemy_visibility_states(self) -> list[int]:
        """Get visibility states of enemies 1-5.

        Returns:
            List of 5 enemy visibility states (index 0 = enemy 5, index 4 = enemy 1)
            0 = Invisible, 1 = Visible, 2 = Dead
        """
        return [self._read_ram_safe(addr, default=0) for addr in ENEMY_VISIBILITY]

    @property
    def enemy_x_positions(self) -> list[int]:
        """Get X positions of enemies 1-5 on the current page.

        Returns:
            List of 5 enemy X positions (index 0 = enemy 5, index 4 = enemy 1)
            Returns ENEMY_NOT_PRESENT for invisible or dead enemies
        """
        positions = []
        visibility_states = self.enemy_visibility_states
        for i, addr in enumerate(ENEMY_X_POS):
            if visibility_states[i] in [ENEMY_INVISIBLE, ENEMY_DEAD]:
                positions.append(ENEMY_NOT_PRESENT)
            else:
                positions.append(self._read_ram_safe(addr, default=0))
        return positions

    @property
    def enemy_y_positions(self) -> list[int]:
        """Get Y positions of enemies 1-5 on the current page (with y=0 at bottom).

        Returns:
            List of 5 enemy Y positions (index 0 = enemy 5, index 4 = enemy 1)
            Returns ENEMY_NOT_PRESENT for invisible or dead enemies
        """
        positions = []
        visibility_states = self.enemy_visibility_states
        for i, addr in enumerate(ENEMY_Y_POS):
            if visibility_states[i] in [ENEMY_INVISIBLE, ENEMY_DEAD]:
                positions.append(ENEMY_NOT_PRESENT)
            else:
                y_pos = self._get_y_position(addr)
                # Invert the y-coordinate within the screen space
                positions.append(SCREEN_HEIGHT - 1 - y_pos)
        return positions

    @property
    def enemy_speeds(self) -> list[int]:
        """Get horizontal speeds of enemies 1-5 (signed: positive=right, negative=left).

        Returns:
            List of 5 enemy speeds (index 0 = enemy 5, index 4 = enemy 1)
            Returns ENEMY_NOT_PRESENT for invisible or dead enemies
        """
        speeds = []
        visibility_states = self.enemy_visibility_states
        for i, addr in enumerate(ENEMY_SPEED):
            if visibility_states[i] in [ENEMY_INVISIBLE, ENEMY_DEAD]:
                speeds.append(ENEMY_NOT_PRESENT)
            else:
                speed = self._read_ram_safe(addr, default=0)
                signed_speed = speed if speed < 128 else speed - 256
                speeds.append(signed_speed)
        return speeds

    @property
    def enemy_x_pages(self) -> list[int]:
        """Get X pages of enemies 1-5.

        Returns:
            List of 5 enemy X pages (index 0 = enemy 5, index 4 = enemy 1)
            Returns ENEMY_NOT_PRESENT for invisible or dead enemies
        """
        pages = []
        visibility_states = self.enemy_visibility_states
        for i, addr in enumerate(ENEMY_X_PAGE):
            if visibility_states[i] in [ENEMY_INVISIBLE, ENEMY_DEAD]:
                pages.append(ENEMY_NOT_PRESENT)
            else:
                pages.append(self._read_ram_safe(addr, default=0))
        return pages

    @property
    def enemy_y_pages(self) -> list[int]:
        """Get Y pages of enemies 1-5.

        Returns:
            List of 5 enemy Y pages (index 0 = enemy 5, index 4 = enemy 1)
            Returns ENEMY_NOT_PRESENT for invisible or dead enemies
        """
        pages = []
        visibility_states = self.enemy_visibility_states
        for i, addr in enumerate(ENEMY_Y_PAGE):
            if visibility_states[i] in [ENEMY_INVISIBLE, ENEMY_DEAD]:
                pages.append(ENEMY_NOT_PRESENT)
            else:
                pages.append(self._read_ram_safe(addr, default=0))
        return pages

    @property
    def enemy_x_positions_global(self) -> list[int]:
        """Get global X positions of enemies 1-5.

        Returns:
            List of 5 enemy global X positions (index 0 = enemy 5, index 4 = enemy 1)
            Returns ENEMY_NOT_PRESENT for invisible or dead enemies
        """
        global_positions = []
        visibility_states = self.enemy_visibility_states

        for i, (x_pos_addr, x_page_addr) in enumerate(zip(ENEMY_X_POS, ENEMY_X_PAGE)):
            if visibility_states[i] in [ENEMY_INVISIBLE, ENEMY_DEAD]:
                global_positions.append(ENEMY_NOT_PRESENT)
            else:
                x_page = self._read_ram_safe(x_page_addr, default=0)
                x_pos = self._read_ram_safe(x_pos_addr, default=0)
                global_x = (x_page * PAGE_SIZE) + x_pos
                global_positions.append(global_x)
        return global_positions

    @property
    def enemy_y_positions_global(self) -> list[int]:
        """Get global Y positions of enemies 1-5 (with y=0 at bottom, increasing upward).

        Returns:
            List of 5 enemy global Y positions (index 0 = enemy 5, index 4 = enemy 1)
            Returns ENEMY_NOT_PRESENT for invisible or dead enemies
        """
        global_positions = []
        visibility_states = self.enemy_visibility_states

        for i, (y_pos_addr, y_page_addr) in enumerate(zip(ENEMY_Y_POS, ENEMY_Y_PAGE)):
            if visibility_states[i] in [ENEMY_INVISIBLE, ENEMY_DEAD]:
                global_positions.append(ENEMY_NOT_PRESENT)
            else:
                y_page = self._read_ram_safe(y_page_addr, default=0)
                y_pos_raw = self._get_y_position(y_pos_addr)
                inverted_y = self._transform_y_coordinate(y_page, y_pos_raw)
                global_positions.append(inverted_y)
        return global_positions

    @property
    def enemy_x_positions_relative(self) -> list[int]:
        """Get X positions of enemies 1-5 relative to player using global coordinates.

        Returns:
            List of 5 enemy X positions relative to player (index 0 = enemy 5, index 4 = enemy 1)
            Returns ENEMY_NOT_PRESENT for invisible or dead enemies
        """
        relative_positions = []
        visibility_states = self.enemy_visibility_states
        player_x_global = self.x_position_global
        enemy_x_global = self.enemy_x_positions_global

        for i in range(len(visibility_states)):
            if visibility_states[i] in [ENEMY_INVISIBLE, ENEMY_DEAD]:
                relative_positions.append(ENEMY_NOT_PRESENT)
            else:
                if enemy_x_global[i] != ENEMY_NOT_PRESENT:
                    relative_positions.append(player_x_global - enemy_x_global[i])
                else:
                    relative_positions.append(ENEMY_NOT_PRESENT)
        return relative_positions

    @property
    def enemy_y_positions_relative(self) -> list[int]:
        """Get Y positions of enemies 1-5 relative to player using global coordinates.

        With inverted Y coordinates (y=0 at bottom, increasing upward):
        - Positive values = enemy is above player (enemy has higher Y)
        - Negative values = enemy is below player (enemy has lower Y)

        Returns:
            List of 5 enemy Y positions relative to player (index 0 = enemy 5, index 4 = enemy 1)
            Returns ENEMY_NOT_PRESENT for invisible or dead enemies
        """
        relative_positions = []
        visibility_states = self.enemy_visibility_states
        player_y_global = self.y_position_global
        enemy_y_global = self.enemy_y_positions_global

        for i in range(len(visibility_states)):
            if visibility_states[i] in [ENEMY_INVISIBLE, ENEMY_DEAD]:
                relative_positions.append(ENEMY_NOT_PRESENT)
            else:
                if enemy_y_global[i] != ENEMY_NOT_PRESENT:
                    relative_positions.append(player_y_global - enemy_y_global[i])
                else:
                    relative_positions.append(ENEMY_NOT_PRESENT)
        return relative_positions

    @property
    def enemy_hp(self) -> list[int]:
        """Get HP values of enemies 1-5.

        Returns:
            List of 5 enemy HP values (index 0 = enemy 5, index 4 = enemy 1)
            Returns ENEMY_NOT_PRESENT for invisible or dead enemies
        """
        hp_values = []
        visibility_states = self.enemy_visibility_states
        for i, addr in enumerate(ENEMY_HEALTH):
            if visibility_states[i] in [ENEMY_INVISIBLE, ENEMY_DEAD]:
                hp_values.append(ENEMY_NOT_PRESENT)
            else:
                hp_values.append(self._read_ram_safe(addr, default=0))
        return hp_values