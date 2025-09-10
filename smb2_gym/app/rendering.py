"""Rendering module for SMB2 environments."""

import numpy as np
import pygame


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