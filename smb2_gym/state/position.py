"""Position, world, and coordinate system properties for SMB2 environment."""

from ..constants import (
    GAME_STATE,
    LEVEL_NAMES,
    PAGE_SIZE,
    PLAYER,
    SCREEN_HEIGHT,
    GlobalCoordinate,
)
from ._base import GameStateMixin


class PositionMixin(GameStateMixin):
    """Mixin class providing position, world, and coordinate properties for SMB2 environment."""

    @property
    def world(self) -> int:
        """Get current world number (1-based for display)."""
        return self._read_ram_safe(GAME_STATE.WORLD_NUMBER) + 1

    @property
    def level(self) -> str:
        """Get current level string (e.g., '1-1', '7-2')."""
        level_id = self._read_ram_safe(GAME_STATE.CURRENT_LEVEL)
        return LEVEL_NAMES.get(level_id, f"L-{level_id:02X}")

    # ---- X ---------------------------------------------------------

    @property
    def x_position(self) -> int:
        """Get player local X position (on current page)."""
        return self._read_ram_safe(PLAYER.X_POSITION)

    @property
    def x_page(self) -> int:
        """Get the X page of the player position."""
        return self._read_ram_safe(PLAYER.X_PAGE)

    def _x_position_global_raw(self) -> int:
        """Get player global X position (raw, without transition handling)."""
        x_page = self._read_ram_safe(PLAYER.X_PAGE)
        x_pos = self._read_ram_safe(PLAYER.X_POSITION)
        return (x_page * PAGE_SIZE) + x_pos

    @property
    def x_position_global(self) -> int:
        """Get player global X position from global coordinate system."""
        return self.global_coordinate_system.global_x

    # ---- Y ---------------------------------------------------------

    @property
    def y_position_raw(self) -> int:
        """Get player local Y position (with y=0 at the TOP."""
        return self._read_ram_safe(PLAYER.Y_POSITION)

    @property
    def y_position(self) -> int:
        """Get player local Y position (with y=0 at BOTTOM, increasing upward)."""
        y_pos = self._read_ram_safe(PLAYER.Y_POSITION)
        return SCREEN_HEIGHT - 1 - y_pos

    @property
    def y_page(self) -> int:
        """Get the Y page of the player position."""
        y_page = self._read_ram_safe(PLAYER.Y_PAGE)
        if y_page == 255:  # Screen wrap around
            return 0
        return y_page

    def _transform_y_coordinate(self, y_page: int, y_pos_raw: int) -> int:
        """Transform raw Y coordinates to inverted system (y=0 at BOTTOM, increasing upward).

        Args:
            y_page: Y page value from RAM
            y_pos_raw: Y position value from RAM

        Returns:
            Inverted Y coordinate
        """
        # Handle wraparound: when goes above top, y_page becomes 255
        if y_page == 255:
            y_page = 0

        y_pos_global: int = y_page * SCREEN_HEIGHT + y_pos_raw

        if self.is_vertical_area:
            max_y_in_level = self.total_pages_in_sub_area * SCREEN_HEIGHT
        else:
            max_y_in_level = SCREEN_HEIGHT

        return max_y_in_level - y_pos_global - 1

    def _y_position_global_raw(self) -> int:
        """Get player global Y position (raw, without transition handling)."""
        y_page = self._read_ram_safe(PLAYER.Y_PAGE)
        y_pos_raw = self._read_ram_safe(PLAYER.Y_POSITION)
        return self._transform_y_coordinate(y_page, y_pos_raw)

    @property
    def y_position_global(self) -> int:
        """Get player global Y position from global coordinate system."""
        return self.global_coordinate_system.global_y

    # ---- Other -----------------------------------------------------

    @property
    def area(self) -> int:
        """Get current area."""
        return self._read_ram_safe(GAME_STATE.AREA)

    @property
    def sub_area(self) -> int:
        """Get current sub-area."""
        return self._read_ram_safe(GAME_STATE.SUB_AREA)

    @property
    def spawn_page(self) -> int:
        """Get current spawn page/entry point."""
        return self._read_ram_safe(GAME_STATE.PAGE)

    @property
    def current_page_position(self) -> int:
        """Get current page position in sub-area."""
        return self._read_ram_safe(GAME_STATE.CURRENT_PAGE_POSITION)

    @property
    def total_pages_in_sub_area(self) -> int:
        """Get total number of pages in the current sub-area."""
        total_pages = self._read_ram_safe(GAME_STATE.TOTAL_PAGES_IN_SUB_AREA)
        return total_pages + 1  # zero indexed

    @property
    def is_vertical_area(self) -> bool:
        """Check if current area has vertical scrolling."""
        direction = self._read_ram_safe(GAME_STATE.SCROLL_DIRECTION)
        return not bool(direction)

    @property
    def global_coordinate_system(self) -> GlobalCoordinate:
        """
        Get global coordinate system combining level structure with player
        position.

        Returns a 4-tuple coordinate system: (Area, Sub-area, Global_X, Global_Y)

        This provides a unified positioning system that combines:
        - Level structure: Area, Sub-area (from memory addresses $04E7-$04E8)
        - Player position: Global X and Y coordinates in the game world

        Note: During door transitions, SMB2 updates sub_area before updating
        player coordinates. This method waits AREA_TRANSITION_FRAMES after detectin25
        transition before accepting new coordinates to ensure they've fully updated.

        Returns:
            GlobalCoordinate: NamedTuple with area, sub_area, global_x, global_y
        """
        current_sub_area = self.sub_area
        current_x = self._x_position_global_raw()
        current_y = self._y_position_global_raw()

        # Check if we're in a transition state where sub_area changed but coordinates haven't
        if (self._previous_sub_area is not None and \
            self._previous_x_global is not None and
            self._previous_y_global is not None):

            # Detect new transition
            if (self._transition_frame_count == 0 and \
                current_sub_area != self._previous_sub_area and \
                current_x == self._previous_x_global and
                current_y == self._previous_y_global):
                self._transition_frame_count = 1
                current_sub_area = self._previous_sub_area

            # Detect transition period
            elif self._transition_frame_count > 0:
                self._transition_frame_count += 1
                if self._transition_frame_count <= self.AREA_TRANSITION_FRAMES:
                    current_sub_area = self._previous_sub_area
                    current_x = self._previous_x_global
                    current_y = self._previous_y_global
                elif self._transition_frame_count == self.AREA_TRANSITION_FRAMES + 1:
                    self._transition_frame_count = 0  # Reset counter

        return GlobalCoordinate(
            area=self.area,
            sub_area=current_sub_area,
            global_x=current_x,
            global_y=current_y,
        )

