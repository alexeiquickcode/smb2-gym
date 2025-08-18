"""Comprehensive info display module for Super Mario Bros 2."""

from typing import Any

import pygame

from ..constants import CHARACTER_NAMES


def get_required_info_height(scale: int = 1) -> int:
    """Get the minimum height needed for the info display."""
    return 240 * scale // 2  # Height for 4 columns with padding, scaled appropriately


def draw_info(screen: pygame.Surface, info: dict[str, Any], font: pygame.font.Font, start_y: int = 10, screen_width: int = 512) -> None:
    """Draw comprehensive game information on screen in 4 columns.

    Args:
        screen: Pygame screen surface
        info: Game info dictionary from environment
        font: Pygame font object
        start_y: Y position to start drawing
        screen_width: Width of screen for column spacing
    """
    # Calculate column positions
    col_width = screen_width // 4
    col_positions = [10, col_width + 10, 2 * col_width + 10, 3 * col_width + 10]

    # Column 1 - Character & Basic Status
    col1_texts = [
        f"Character: {CHARACTER_NAMES.get(info['character'], 'Unknown')}",
        f"Lives: {info['life']}",
        f"Hearts: {info['hearts']}/4",
        f"Continues: {info['continues']}",
        "",
        f"Position (Local):",
        f"  X: {info['x_pos_local']}",
        f"  Y: {info['y_pos_local']}",
        f"Position (Global):",
        f"  X: {info['x_pos_global']}",
        f"  Y: {info['y_pos_global']}",
        f"Speed: {info['player_speed']}",
        f"On Vine: {'Yes' if info['on_vine'] else 'No'}",
    ]

    # Column 2 - Level & World Info
    col2_texts = [
        f"World: {info['world']}",
        f"Level: {info['level']}",
        "",
        f"Collectibles:",
        f"  Cherries: {info['cherries']}",
        f"  Coins: {info['coins']}",
        "",
        f"Items:",
        f"  Holding: {'Yes' if info['holding_item'] else 'No'}",
        f"  Pulled: {info['item_pulled']}",
    ]

    # Column 3 - Timers & Power-ups
    col3_texts = [
        f"Power-ups & Timers:",
        f"  Starman: {info['starman_timer']}{'⭐' if info['starman_timer'] > 0 else ''}",
        f"  Subspace: {info['subspace_timer']}",
        f"  Stopwatch: {info['stopwatch_timer']}{'⏱️' if info['stopwatch_timer'] > 0 else ''}",
        f"  Float: {info['float_timer']}{'🎈' if info['float_timer'] > 0 else ''}",
        "",
        f"Status Indicators:",
        f"  Starman Active: {'YES' if info['starman_timer'] > 0 else 'No'}",
        f"  Stopwatch Active: {'YES' if info['stopwatch_timer'] > 0 else 'No'}",
        f"  Peach Float: {'YES' if info['float_timer'] > 0 else 'No'}",
    ]

    # Column 4 - Progress & Stats
    levels = info['levels_finished']
    col4_texts = [
        f"Levels Completed:",
        f"  Mario: {levels['mario']}",
        f"  Peach: {levels['peach']}",
        f"  Toad: {levels['toad']}",
        f"  Luigi: {levels['luigi']}",
        "",
        f"Character Abilities:",
        f"  Mario: Balanced",
        f"  Peach: Float",
        f"  Toad: Strong+Fast",
        f"  Luigi: High Jump",
    ]

    # Helper function to draw column
    def draw_column(texts, x_pos, max_lines=12):
        for i, text in enumerate(texts[:max_lines]):
            if text:
                # Color coding for special states
                color = (255, 255, 255)  # Default white
                if "YES" in text:
                    color = (0, 255, 0)  # Green for active states
                elif "⭐" in text or "⏱️" in text or "🎈" in text:
                    color = (255, 255, 0)  # Yellow for active power-ups
                elif text.startswith("  ") and any(
                    word in text for word in ["Starman", "Stopwatch", "Float"]
                ):
                    color = (200, 200, 255)  # Light blue for power-up info
                elif text.startswith("Character:"
                                    ) or text.startswith("Lives:") or text.startswith("Hearts:"):
                    color = (255, 200, 100)  # Orange for important stats

                text_surface = font.render(text, True, color)
                screen.blit(text_surface, (x_pos, start_y + i * 20))

    draw_column(col1_texts, col_positions[0])
    draw_column(col2_texts, col_positions[1])
    draw_column(col3_texts, col_positions[2])
    draw_column(col4_texts, col_positions[3])


# ------------------------------------------------------------------------------
# ---- Supporting Functions ----------------------------------------------------
# ------------------------------------------------------------------------------


def create_info_panel(screen: pygame.Surface, info: dict[str, Any], font: pygame.font.Font, game_height: int, screen_width: int) -> int:
    """Create and draw the info panel below the game screen.

    Args:
        screen: Pygame screen surface
        info: Game info dictionary from environment
        font: Pygame font object
        game_height: Height of game area (where info panel starts)
        screen_width: Width of screen
        
    Returns:
        Height of the info panel
    """
    info_height = get_required_info_height()
    pygame.draw.rect(screen, (40, 40, 40), (0, game_height, screen_width, info_height))
    draw_info(screen, info, font, game_height + 10, screen_width)

    return info_height
