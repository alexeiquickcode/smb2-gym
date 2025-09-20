"""Info display module for Super Mario Bros 2 with clean table format."""

from typing import Any

import pygame

from ..constants import CHARACTER_NAMES


def get_required_info_height(scale: int = 1) -> int:
    """Get the minimum height needed for the info display."""
    return 300 * scale // 2  # Height for table display


def create_info_panel(
    screen: pygame.Surface,
    info: dict[str, Any],
    font: pygame.font.Font,
    game_height: int,
    screen_width: int,
) -> int:
    """Create and draw a clean table info panel below the game screen.

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

    # Draw background
    pygame.draw.rect(screen, (30, 30, 30), (0, game_height, screen_width, info_height))

    # Table configuration
    padding = 10
    line_height = 18  # Increased from 16 to add 2 pixels spacing between rows
    col_width = screen_width // 4

    # Text colors
    label_color = (150, 150, 150)
    value_color = (255, 255, 255)
    header_color = (100, 180, 255)

    # Starting position
    x_start = padding
    y_start = game_height + padding

    # Create all data rows
    data = [
        (
            "Character", CHARACTER_NAMES.get(info['character'],
                                             'Unknown'), "Lives", str(info['life'])
        ),
        ("Hearts", f"{info['hearts']}/4", "World", str(info['world'])),
        ("Level", info['level'], "Area", f"{info['area']}-{info['sub_area']}"),
        ("X Position", str(info['x_pos_local']), "Y Position", str(info['y_pos_local'])),
        ("X Page", str(info['x_page']), "Y Page", str(info['y_page'])),
        ("Global X", str(info['x_pos_global']), "Global Y", str(info['y_pos_global'])),
        ("Player Speed", str(info['player_speed']), "On Vine", "Yes" if info['on_vine'] else "No"),
        (
            "Holding Item", "Yes" if info['holding_item'] else "No", "Item Pulled",
            str(info['item_pulled'])
        ),
        (
            "Current Page", str(info['current_page_position']), "Total Pages",
            str(info['total_pages_in_sub_area'])
        ),
        (
            "Vertical Area", "Yes" if info['is_vertical_area'] else "No", "Spawn Page",
            str(info['spawn_page'])
        ),
        ("Cherries", str(info['cherries']), "Coins", str(info['coins'])),
        (
            "Starman Timer", str(info['starman_timer']), "Subspace Timer",
            str(info['subspace_timer'])
        ),
        (
            "Stopwatch Timer", str(info['stopwatch_timer']), "Float Timer",
            f"{info['float_timer']}/60"
        ),
        ("Invuln Timer", str(info['invulnerability_timer']), "Enemies Defeated", str(info['enemies_defeated'])),
        (
            "Vegetables Pulled", str(info['vegetables_pulled']), "Door Timer",
            str(info['door_transition_timer'])
        ),
        (
            "Subspace Status", str(info['subspace_status']), "Level Completed",
            "Yes" if info['level_completed'] else "No"
        ),

        # Level completion per character
        (
            "Mario Levels", str(info['levels_finished']['mario']), "Luigi Levels",
            str(info['levels_finished']['luigi'])
        ),
        (
            "Peach Levels", str(info['levels_finished']['peach']), "Toad Levels",
            str(info['levels_finished']['toad'])
        ),

        # Enemy positions (relative to player) and HP
        (
            "Enemy 1 Rel X/Y", f"{info['enemy_x_positions_relative'][4]}/{info['enemy_y_positions_relative'][4]}",
            "Enemy 1 HP", str(info['enemy_hp'][4])
        ),
        (
            "Enemy 2 Rel X/Y", f"{info['enemy_x_positions_relative'][3]}/{info['enemy_y_positions_relative'][3]}",
            "Enemy 2 HP", str(info['enemy_hp'][3])
        ),
        (
            "Enemy 3 Rel X/Y", f"{info['enemy_x_positions_relative'][2]}/{info['enemy_y_positions_relative'][2]}",
            "Enemy 3 HP", str(info['enemy_hp'][2])
        ),
        (
            "Enemy 4 Rel X/Y", f"{info['enemy_x_positions_relative'][1]}/{info['enemy_y_positions_relative'][1]}",
            "Enemy 4 HP", str(info['enemy_hp'][1])
        ),
        (
            "Enemy 5 Rel X/Y", f"{info['enemy_x_positions_relative'][0]}/{info['enemy_y_positions_relative'][0]}",
            "Enemy 5 HP", str(info['enemy_hp'][0])
        ),
    ]

    # Draw the table
    current_y = y_start

    for i, row in enumerate(data):
        # Draw two label-value pairs per row
        # First pair
        label1_surface = font.render(row[0] + ":", True, label_color)
        value1_surface = font.render(row[1], True, value_color)
        screen.blit(label1_surface, (x_start, current_y))
        screen.blit(value1_surface, (x_start + col_width, current_y))

        # Second pair
        label2_surface = font.render(row[2] + ":", True, label_color)
        value2_surface = font.render(row[3], True, value_color)
        screen.blit(label2_surface, (x_start + col_width * 2, current_y))
        screen.blit(value2_surface, (x_start + col_width * 3, current_y))

        current_y += line_height

        # Draw thin line after first 3 rows
        if i in [2, 5, 15, 17]:
            pygame.draw.line(
                screen, (60, 60, 60), (x_start, current_y + 2),
                (screen_width - padding, current_y + 2), 1
            )
            current_y += 6

    return info_height


def draw_info(
    screen: pygame.Surface,
    info: dict[str, Any],
    font: pygame.font.Font,
    start_y: int = 10,
    screen_width: int = 512
) -> None:
    """Legacy function for compatibility."""
    create_info_panel(screen, info, font, start_y, screen_width)

