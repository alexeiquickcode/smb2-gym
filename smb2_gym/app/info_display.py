"""Info display module for Super Mario Bros 2."""

from typing import Any

import pygame

from ..constants import CHARACTER_NAMES
from ..constants.object_ids import (
    CollisionFlags,
    EnemyId,
    EnemyState,
    PlayerState,
    SpriteFlags,
)


def get_required_info_height(scale: int = 1) -> int:
    """Get the minimum height needed for the info display."""
    # Base: 18 info rows + 9 enemy slots = 27 rows total
    # 27 rows * 18 pixels + 4 separators * 6 pixels = 486 + 24 = 510 pixels
    return 510 * scale // 2


def format_enemy_name(enemy_id: int | None) -> str:
    """Format enemy name from ID.

    Args:
        enemy_id: Enemy ID from RAM or None if not present

    Returns:
        Formatted string like "TWEETER" or "" if no enemy
    """
    if enemy_id is None:
        return ""
    try:
        enemy_name = EnemyId(enemy_id).name
        return enemy_name
    except ValueError:
        return f"UNKNOWN_{enemy_id:02X}"


def format_collision_flags(collision: int | None) -> str:
    """Format collision flags as names.

    Args:
        collision: Collision flags bitfield or None

    Returns:
        Comma-separated flag names or "" if no collision
    """
    if collision is None or collision == 0:
        return ""

    flags = []
    for flag in CollisionFlags:
        if flag != CollisionFlags.NONE and collision & flag:
            flags.append(flag.name)

    return ",".join(flags) if flags else ""


def format_sprite_flags(sprite_flags: int | None) -> str:
    """Format sprite flags as names.

    Args:
        sprite_flags: Sprite flags bitfield or None

    Returns:
        Comma-separated flag names or "" if no flags
    """
    if sprite_flags is None or sprite_flags == 0:
        return ""

    flags = []
    for flag in SpriteFlags:
        if flag != SpriteFlags.NONE and sprite_flags & flag:
            flags.append(flag.name)

    return ",".join(flags) if flags else ""


def format_enemy_state(state: int, has_enemy: bool) -> str:
    """Format enemy state as name.

    Args:
        state: Enemy state value
        has_enemy: Whether there's an enemy in this slot

    Returns:
        State name like "VISIBLE", "DEAD", or empty string if no enemy
    """
    if not has_enemy:
        return ""
    try:
        return EnemyState(state).name
    except ValueError:
        return str(state)


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

    # Text colours
    label_color = (150, 150, 150)
    value_color = (255, 255, 255)
    header_color = (100, 180, 255)

    # Starting position
    x_start = padding
    y_start = game_height + padding

    # Extract accessor objects
    pc = info['pc']
    pos = info['pos']
    game = info['game']
    enemies = info['enemies']

    # Create all data rows organized into sections
    data = [
        # POSITION SECTION
        ("POSITION", "", "", ""),
        ("Level", game.level, "World", str(game.world)),
        ("Area", f"{pos.area}-{pos.sub_area}", "Subspace Status", str(pc.subspace_status)),
        (
            "Local (X, Y)", f"({pos.x_local}, {pos.y_local})", "Global (X, Y)",
            f"({pos.x_global}, {pos.y_global})"
        ),
        (
            "Page (X, Y)", f"({pos.x_page}, {pos.y_page})", "Current/Total",
            f"{pos.current_page}/{pos.total_pages}"
        ),
        ("Vertical Area", "Yes" if pos.is_vertical else "No", "Spawn Page", str(pos.spawn_page)),

        # PLAYER SECTION
        ("PLAYER", "", "", ""),
        ("Character", CHARACTER_NAMES.get(pc.character, 'Unknown'), "Lives", str(pc.lives)),
        ("Hearts", f"{pc.hearts}/4", "Player Speed", str(pc.speed)),
        ("Cherries", str(pc.cherries), "Coins", str(pc.coins)),
        ("Holding Item", "Yes" if pc.holding_item else "No", "Item Pulled", str(pc.item_pulled)),
        (
            "Big Veggies Pulled", str(pc.big_vegetables_pulled), "On Vine",
            "Yes" if pc.on_vine else "No"
        ),
        ("Starman Timer", str(pc.starman_timer), "Subspace Timer", str(pc.subspace_timer)),
        ("Stopwatch Timer", str(pc.stopwatch_timer), "Float Timer", f"{pc.float_timer}/60"),
        (
            "Invuln Timer", str(pc.invulnerability_timer), "Door Timer",
            str(pc.door_transition_timer)
        ),
        (
            "Level Completed", "Yes" if pc.level_completed else "No", "Player State",
            PlayerState(pc.state).name
            if pc.state in [ps.value for ps in PlayerState] else str(pc.state)
        ),
        (
            "Mario Levels", str(pc.levels_finished['mario']), "Luigi Levels",
            str(pc.levels_finished['luigi'])
        ),
        (
            "Peach Levels", str(pc.levels_finished['peach']), "Toad Levels",
            str(pc.levels_finished['toad'])
        ),

        # ENEMIES SECTION
        ("ENEMIES", "", "", ""),
        (
            "Slot", "Name", "HP", "(X, Y)", "Rel (X, Y)", "Vel (X, Y)", "State", "Timer", "Flags",
            "Collision"
        ),

        # Enemy table - all 9 slots, one row per enemy
        *[
            (
                f"{e.slot_number}", format_enemy_name(e.object_type),
                str(e.health) if e.health is not None else "", f"({e.x_position}, {e.y_position})"
                if e.x_position is not None and e.y_position is not None else "",
                f"({e.relative_x(pos.x_global)}, {e.relative_y(pos.y_global)})"
                if e.relative_x(pos.x_global) is not None and e.relative_y(pos.y_global) is not None
                else "", f"({e.x_velocity}, {e.y_velocity})"
                if e.x_velocity is not None and e.y_velocity is not None else "",
                format_enemy_state(e.state, e.object_type is not None),
                str(e.object_timer) if e.object_timer is not None else "",
                format_sprite_flags(e.sprite_flags), format_collision_flags(e.collision)
            ) for e in enemies
        ],
    ]

    # Draw the table
    current_y = y_start

    for i, row in enumerate(data):
        # Check if this is a section header (4-column row with empty strings in cols 2-4)
        is_section_header = (len(row) == 4 and row[1] == "" and row[2] == "" and row[3] == "")

        # Draw line before section header
        if is_section_header:
            pygame.draw.line(
                screen, (60, 60, 60), (x_start, current_y), (screen_width - padding, current_y), 1
            )
            current_y += 6

        # Check if this is a multi-column enemy table row (10 columns)
        if len(row) == 10:
            # Enemy table: custom column widths
            # Slot(small), Name(big), HP, (X,Y), (RelX,RelY), Vel(X,Y), State, Timer, Flags(bigger), Collision(bigger)
            total_width = screen_width - 2 * padding
            base_width = total_width // 10
            col_widths = [
                int(base_width * 0.5),  # Slot
                int(base_width * 1.5),  # Name
                int(base_width * 0.3),  # HP
                int(base_width * 0.9),  # (X, Y)
                int(base_width * 0.9),  # (RelX, RelY)
                int(base_width * 0.9),  # Vel (X, Y)
                int(base_width * 0.8),  # State - VISIBLE, INVISIBLE, DEAD
                int(base_width * 0.5),  # Timer
                int(base_width * 1.4),  # Flags
                int(base_width * 1.4)  # Collision
            ]

            x_offset = x_start
            for j, cell in enumerate(row):
                # Use header colour for enemy table header row (index 14)
                # POSITION (0-6 = 7 rows), PLAYER (7-13 = 7 rows), ENEMIES header (14), enemy header (15), enemies (16-24)
                color = header_color if i == 15 else value_color
                cell_surface = font.render(str(cell), True, color)
                screen.blit(cell_surface, (x_offset, current_y))
                x_offset += col_widths[j]
        elif is_section_header:
            # Section header - render in blue, centered, no colon
            header_surface = font.render(row[0], True, header_color)
            screen.blit(header_surface, (x_start, current_y))
        else:
            # Regular 4-column layout (label-value pairs)
            label1_surface = font.render(row[0] + ":", True, label_color)
            value1_surface = font.render(row[1], True, value_color)
            screen.blit(label1_surface, (x_start, current_y))
            screen.blit(value1_surface, (x_start + col_width, current_y))

            # Don't add colon to column 3 if it's empty (for headers)
            label2_text = row[2] + ":" if row[2] else ""
            label2_surface = font.render(label2_text, True, label_color)
            value2_surface = font.render(row[3], True, value_color)
            screen.blit(label2_surface, (x_start + col_width * 2, current_y))
            screen.blit(value2_surface, (x_start + col_width * 3, current_y))

        current_y += line_height

        # Draw line after section headers and enemy table header (but not between enemy slots)
        if is_section_header or i == 15:  # After section headers or enemy table header only
            pygame.draw.line(
                screen, (60, 60, 60), (x_start, current_y + 2),
                (screen_width - padding, current_y + 2), 1
            )
            current_y += 6

    return info_height
