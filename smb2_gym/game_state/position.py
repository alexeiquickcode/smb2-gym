"""Position and coordinate system properties for SMB2 environment."""

from .base import GameStateMixin
from ..constants import (
    AREA,
    CURRENT_PAGE_POSITION,
    PAGE,
    PAGE_SIZE,
    PLAYER_X_PAGE,
    PLAYER_X_POSITION,
    PLAYER_Y_PAGE,
    PLAYER_Y_POSITION,
    SCREEN_HEIGHT,
    SCROLL_DIRECTION,
    SUB_AREA,
    TOTAL_PAGES_IN_SUB_AREA,
    WORLD_NUMBER,
    GlobalCoordinate,
)


class PositionMixin(GameStateMixin):
    """Mixin class providing position and coordinate properties for SMB2 environment."""

    @property
    def x_position_global(self) -> int:
        """Get player global X position."""
        x_page = self._read_ram_safe(PLAYER_X_PAGE, default=0)
        x_pos = self._read_ram_safe(PLAYER_X_POSITION, default=0)
        return (x_page * PAGE_SIZE) + x_pos

    @property
    def x_position(self) -> int:
        """Get player local X position (on current page)."""
        x_pos = self._read_ram_safe(PLAYER_X_POSITION, default=0)
        return x_pos

    @property
    def x_page(self) -> int:
        """Get the X page of the player position."""
        x_page = self._read_ram_safe(PLAYER_X_PAGE)
        return x_page

    def _transform_y_coordinate(self, y_page: int, y_pos_raw: int) -> int:
        """Transform raw Y coordinates to inverted system (y=0 at bottom, increasing upward).

        Args:
            y_page: Y page value from RAM
            y_pos: Y position value from RAM

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

    @property
    def y_position_global(self) -> int:
        """Get player global Y position (with y=0 at bottom, increasing upward)."""
        y_page = self._read_ram_safe(PLAYER_Y_PAGE, default=0)
        y_pos_raw = self._get_y_position(PLAYER_Y_POSITION)
        return self._transform_y_coordinate(y_page, y_pos_raw)

    @property
    def y_position(self) -> int:
        """Get player local Y position (with y=0 at bottom, increasing upward)."""
        y_pos = self._get_y_position(PLAYER_Y_POSITION)
        # Invert the y-coordinate within the screen space
        return SCREEN_HEIGHT - 1 - y_pos

    @property
    def y_page(self) -> int:
        """Get the Y page of the player position."""
        y_page = self._read_ram_safe(PLAYER_Y_PAGE)
        if y_page == 255:  # Screen wrap around
            return 0
        return y_page

    @property
    def world(self) -> int:
        """Get current world number. RAM is 0-based, display is 1-based."""
        return self._read_ram_safe(WORLD_NUMBER, default=0) + 1

    @property
    def area(self) -> int:
        """Get current area."""
        area = self._read_ram_safe(AREA, default=0)
        return area

    @property
    def sub_area(self) -> int:
        """Get current sub-area."""
        sub_area = self._read_ram_safe(SUB_AREA, default=0)
        return sub_area

    @property
    def spawn_page(self) -> int:
        """Get current spawn page/entry point."""
        page = self._read_ram_safe(PAGE, default=0)
        return page

    @property
    def current_page_position(self) -> int:
        """Get current page position in sub-area."""
        page_pos = self._read_ram_safe(CURRENT_PAGE_POSITION, default=0)
        return page_pos

    @property
    def total_pages_in_sub_area(self) -> int:
        """Get total number of pages in the current sub-area."""
        total_pages = self._read_ram_safe(TOTAL_PAGES_IN_SUB_AREA, default=0)
        return total_pages + 1  # zero indexed

    @property
    def is_vertical_area(self) -> bool:
        """Check if current area has vertical scrolling."""
        direction = self._read_ram_safe(SCROLL_DIRECTION, default=0)
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
        current_x = self.x_position_global
        current_y = self.y_position_global

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