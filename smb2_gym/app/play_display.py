"""Display and rendering functions for human play interface."""

import numpy as np
import pygame

from smb2_gym.app.info_display import create_info_panel
from smb2_gym.app.rendering import render_frame
from smb2_gym.constants import TILE_COLORS, FineTileType
from smb2_gym.smb2_env import SuperMarioBros2Env


def draw_semantic_map(
    surface: pygame.Surface,
    semantic_map: np.ndarray,
    x_offset: int,
    y_offset: int,
    tile_size: int,
) -> None:
    """Draw the semantic tile map on the surface.

    Args:
        semantic_map: Structured array with 'fine_type', 'color_r', 'color_g', 'color_b' fields
    """
    height, width = semantic_map.shape

    for y in range(height):
        for x in range(width):

            # Get colours
            color = (
                int(semantic_map[y, x]['color_r']),
                int(semantic_map[y, x]['color_g']),
                int(semantic_map[y, x]['color_b']),
            )

            # Calculate position
            screen_y = y * tile_size + y_offset
            screen_x = x * tile_size + x_offset

            rect = pygame.Rect(screen_x, screen_y, tile_size, tile_size)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, (0, 0, 0), rect, 1)  # Black border


def draw_player_position(
    surface: pygame.Surface,
    env: SuperMarioBros2Env,
    x_offset: int,
    y_offset: int,
    tile_size: int,
) -> None:
    """Draw player position on the collision map."""
    # Get player collision tiles from the environment
    player_tiles = env.get_player_collision_tiles()

    for tile_x, tile_y in player_tiles:
        screen_x = tile_x * tile_size + x_offset + tile_size // 2
        screen_y = tile_y * tile_size + y_offset + tile_size // 2
        pygame.draw.circle(surface, (255, 255, 255), (screen_x, screen_y), tile_size // 3)
        pygame.draw.circle(surface, (255, 0, 0), (screen_x, screen_y), tile_size // 3, 2)


def draw_legend(
    surface: pygame.Surface,
    font: pygame.font.Font,
    x_offset: int,
    y_offset: int,
) -> None:
    """Draw a legend for semantic tile types."""
    y_pos = y_offset

    for tile_type in FineTileType:
        if tile_type == FineTileType.EMPTY:
            continue

        color = TILE_COLORS.get(tile_type, (128, 128, 128))

        # Draw colour box
        rect = pygame.Rect(x_offset, y_pos, 16, 16)
        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, (0, 0, 0), rect, 1)

        # Draw text using enum name
        text = font.render(tile_type.name, True, (255, 255, 255))
        surface.blit(text, (x_offset + 20, y_pos))

        y_pos += 20


def render_all(
    screen: pygame.Surface,
    obs: np.ndarray,
    env: SuperMarioBros2Env,
    info: dict,
    game_width: int,
    game_height: int,
    total_width: int,
    total_height: int,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    paused: bool,
) -> None:
    """Render all game elements to screen."""
    # Clear screen
    screen.fill((40, 40, 40))

    # Render game on the left side
    game_surface = pygame.Surface((game_width, game_height))
    render_frame(game_surface, obs, game_width, game_height)
    screen.blit(game_surface, (10, 10))

    # Get and render semantic map on the right side
    semantic_map = env.semantic_map
    map_x_offset = game_width + 30
    map_y_offset = 10
    tile_size = 20

    draw_semantic_map(screen, semantic_map, map_x_offset, map_y_offset, tile_size)
    draw_player_position(screen, env, map_x_offset, map_y_offset, tile_size)

    # Draw semantic map title
    title_text = font.render("Semantic Map", True, (255, 255, 255))
    screen.blit(title_text, (map_x_offset, map_y_offset - 30))

    # Draw legend
    legend_x = map_x_offset + (16 * tile_size) + 10
    legend_y = map_y_offset
    legend_title = small_font.render("Legend:", True, (255, 255, 255))
    screen.blit(legend_title, (legend_x, legend_y))
    draw_legend(screen, small_font, legend_x, legend_y + 20)

    # Draw game info panel at bottom
    create_info_panel(screen, info, font, total_height, total_width)

    # Draw pause indicator
    if paused:
        pause_text = font.render("PAUSED", True, (255, 255, 0))
        text_rect = pause_text.get_rect(center=(total_width // 2, total_height // 2))
        screen.blit(pause_text, text_rect)
