"""Rendering module for SMB2 environments."""

import logging
from types import TracebackType
from typing import (
    Any,
    Optional,
)

import numpy as np
import pygame
from numpy.typing import NDArray

from smb2_gym.app.info_display import (
    create_info_panel,
    get_required_info_height,
)


class GameRenderer:
    """Centralized game renderer for SMB2 environments."""

    def __init__(self, scale: int = 3, caption: str = "Super Mario Bros 2"):
        """Initialize the game renderer.

        Args:
            scale: Display scale factor
            caption: Window caption
            info_style: Info panel style ('comprehensive', 'minimal', etc.)
        """
        self.scale: int = scale
        self.caption: str = caption
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Display
        self.width: int = 256 * scale
        self.height: int = 240 * scale
        self.info_height: int = get_required_info_height(scale)

        # Pygame objects
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None

        # State
        self.initialized: bool = False

    def initialize(self) -> None:
        """Initialize pygame and create display."""
        if self.initialized:
            return
        try:
            pygame.init()

            # Create display
            self.screen = pygame.display.set_mode((self.width, self.height + self.info_height))
            pygame.display.set_caption(self.caption)

            # Create clock and font
            self.clock = pygame.time.Clock()
            font_size = 18 * self.scale // 2
            self.font = pygame.font.Font(None, font_size)

            self.initialized = True
            self.logger.info(
                f"Pygame renderer initialized: {self.width}x{self.height + self.info_height}"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize pygame renderer: {e}")
            raise

    def render_frame(
        self,
        rgb_frame: Optional[NDArray[np.uint8]],
        info: Optional[dict[str, Any]] = None,
        handle_events: bool = True
    ) -> bool:
        """Render a single frame.

        Args:
            rgb_frame: RGB frame data (240, 256, 3)
            info: Game info dict for info panel
            handle_events: Whether to handle pygame events

        Returns:
            True if should continue, False if quit was requested
        """
        if not self.initialized:
            self.initialize()
        try:
            # Handle pygame events
            if handle_events:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False

            # Clear screen
            if self.screen:
                self.screen.fill((0, 0, 0))

            # Render game frame
            if rgb_frame is not None:

                # Create pygame surface from RGB array
                frame_surface = pygame.Surface((256, 240), depth=24)

                # Convert RGB array to pygame's expected format
                frame_data = np.transpose(rgb_frame, (1, 0, 2))  # (256, 240, 3)
                pygame.surfarray.blit_array(frame_surface, frame_data)

                # Scale and blit to screen
                scaled_frame = pygame.transform.scale(frame_surface, (self.width, self.height))
                if self.screen:
                    self.screen.blit(scaled_frame, (0, 0))

            # Draw info panel and update display
            if info and self.font and self.screen:
                create_info_panel(self.screen, info, self.font, self.height, self.width)
            pygame.display.flip()

            return True

        except Exception as e:
            self.logger.warning(f"Rendering error: {e}")
            return True

    def tick(self, fps: int = 60) -> None:
        """Tick the clock to limit FPS."""
        if self.clock:
            self.clock.tick(fps)

    def close(self) -> None:
        """Close the renderer and cleanup pygame."""
        if self.initialized:
            try:
                pygame.quit()
                self.initialized = False
                self.logger.info("Pygame renderer closed")
            except Exception as e:
                self.logger.warning(f"Error closing pygame renderer: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(
        self, exc_type: Optional[type[BaseException]], exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()


# ------------------------------------------------------------------------------
# ---- Subclasses --------------------------------------------------------------
# ------------------------------------------------------------------------------


class TrainingRenderer(GameRenderer):
    """Specialized renderer for training environments."""

    def __init__(self, scale: int = 3):
        super().__init__(scale=scale, caption="SMB2 Training")

    def render_training_step(
        self,
        rgb_frame: Optional[NDArray[np.uint8]],
        info: dict[str, Any],
        episode: int = 0,
        step: int = 0,
        reward: float = 0.0
    ) -> bool:
        """Render a training step with additional training info.

        Args:
            rgb_frame: RGB frame data
            info: Game info dict
            episode: Current episode number
            step: Current step number
            reward: Current reward

        Returns:
            True if should continue, False if quit requested
        """
        # Add training-specific info
        training_info = info.copy() if info else {}
        training_info.update(
            {
                'episode': episode,
                'step': step,
                'reward': reward  # Pass raw float value for consistent formatting
            }
        )

        return self.render_frame(rgb_frame, training_info, handle_events=True)

    def render_without_fps_limit(
        self,
        rgb_frame: Optional[NDArray[np.uint8]],
        info: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Render without FPS limiting for fast training."""
        return self.render_frame(rgb_frame, info, handle_events=True)


class HumanPlayRenderer(GameRenderer):
    """Specialized renderer for human play."""

    def __init__(self, scale: int = 2):
        super().__init__(scale=scale, caption="Super Mario Bros 2")

    def render_human_step(
        self, rgb_frame: Optional[NDArray[np.uint8]], info: dict[str, Any], fps: int = 60
    ) -> bool:
        """Render a human play step with FPS limiting.

        Args:
            rgb_frame: RGB frame data
            info: Game info dict  
            fps: Target FPS

        Returns:
            True if should continue, False if quit requested
        """
        result = self.render_frame(rgb_frame, info, handle_events=True)
        self.tick(fps)
        return result


def render_frame(screen: pygame.Surface, obs: np.ndarray, width: int, height: int) -> None:
    """Render a game frame to a pygame surface.
    
    Args:
        screen: Pygame surface to render to
        obs: Observation array of shape (240, 256, 3) 
        width: Target width for scaling
        height: Target height for scaling
    """
    screen.fill((0, 0, 0))  # Clear screen
    frame = pygame.Surface((256, 240), depth=24)  # Draw frame
    frame_data = np.transpose(obs, (1, 0, 2))  # (256, 240, 3)
    pygame.surfarray.blit_array(frame, frame_data)
    frame = pygame.transform.scale(frame, (width, height))
    screen.blit(frame, (0, 0))
