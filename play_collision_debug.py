import argparse
from typing import (
    Optional,
    Union,
)

import numpy as np
import pygame

from smb2_gym.actions import actions_to_buttons
from smb2_gym.app import InitConfig
from smb2_gym.app.info_display import (
    create_info_panel,
    get_required_info_height,
)
from smb2_gym.app.keyboard import (
    ALT_KEYBOARD_MAPPING,
    KEYBOARD_MAPPING,
)
from smb2_gym.app.rendering import render_frame
from smb2_gym.constants import (
    DEFAULT_SCALE,
    FONT_SIZE_BASE,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from smb2_gym.smb2_env import SuperMarioBros2Env

# Collision visualization colors
COLLISION_COLORS = {
    0: (200, 200, 200),  # EMPTY - light gray
    1: (101, 67, 33),  # SOLID - brown
    2: (139, 69, 19),  # PLATFORM - saddle brown
    3: (34, 139, 34),  # VINE - forest green
    4: (255, 215, 0),  # DOOR - gold
    5: (147, 112, 219),  # JAR - medium slate blue
    6: (220, 20, 60),  # SPIKES - crimson
    7: (244, 164, 96),  # QUICKSAND - sandy brown
    8: (105, 105, 105),  # CONVEYOR_LEFT - dim gray
    9: (105, 105, 105),  # CONVEYOR_RIGHT - dim gray
    10: (160, 82, 45),  # LADDER - saddle brown
    11: (169, 169, 169),  # CHAIN - dark gray
    12: (255, 20, 147),  # CHERRY - deep pink
    13: (50, 205, 50),  # VEGETABLE - lime green
    14: (186, 85, 211),  # POTION - medium orchid
    15: (255, 140, 0),  # MUSHROOM - dark orange
    16: (255, 0, 0),  # HEART - red
    17: (255, 255, 0),  # STAR - yellow
    18: (255, 0, 0),  # ENEMY - red
    19: (139, 0, 0),  # BOSS - dark red
    20: (255, 165, 0),  # PROJECTILE - orange
    21: (30, 144, 255),  # WATER - dodger blue
    22: (176, 224, 230),  # ICE - powder blue
    23: (255, 255, 255),  # CLOUD - white
    24: (221, 160, 221),  # CARPET - plum
    25: (154, 205, 50),  # HAWK_MOUTH - yellow green
    26: (0, 255, 255),  # CRYSTAL - cyan
    27: (75, 0, 130),  # MASK - indigo
    28: (255, 255, 224),  # POW_BLOCK - light yellow
    29: (128, 0, 0),  # BOMB - maroon
    30: (255, 215, 0),  # KEY - gold
}


def draw_collision_map(
    surface: pygame.Surface, collision_map: np.ndarray, x_offset: int, y_offset: int, tile_size: int
) -> None:
    """Draw the collision map on the surface."""
    height, width = collision_map.shape

    for y in range(height):
        for x in range(width):
            tile_type = collision_map[y, x]
            color = COLLISION_COLORS.get(tile_type, (128, 128, 128))  # Default gray

            # Calculate position
            screen_y = y * tile_size + y_offset
            screen_x = x * tile_size + x_offset

            rect = pygame.Rect(screen_x, screen_y, tile_size, tile_size)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, (0, 0, 0), rect, 1)  # Black border


def draw_player_position(
    surface: pygame.Surface, env: SuperMarioBros2Env, x_offset: int, y_offset: int, tile_size: int
) -> None:
    """Draw player position on the collision map."""
    # Get Mario's actual position on screen
    if hasattr(env, 'get_player_screen_position'):
        player_pos = env.get_player_screen_position()

        if player_pos is not None:
            mario_screen_x, mario_screen_y = player_pos

            # Convert to screen coordinates
            screen_x = mario_screen_x * tile_size + x_offset + tile_size // 2
            screen_y = mario_screen_y * tile_size + y_offset + tile_size // 2

            # Draw player as white circle with red outline
            pygame.draw.circle(surface, (255, 255, 255), (screen_x, screen_y), tile_size // 3)
            pygame.draw.circle(surface, (255, 0, 0), (screen_x, screen_y), tile_size // 3, 2)


def draw_legend(
    surface: pygame.Surface, font: pygame.font.Font, x_offset: int, y_offset: int
) -> None:
    """Draw a legend for collision types."""
    legend_items = [
        (0, "Empty"),
        (1, "Solid"),
        (2, "Platform"),
        (3, "Vine"),
        (4, "Door"),
        (18, "Enemy"),
    ]

    y_pos = y_offset
    for tile_type, name in legend_items:
        color = COLLISION_COLORS.get(tile_type, (128, 128, 128))

        # Draw color box
        rect = pygame.Rect(x_offset, y_pos, 16, 16)
        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, (0, 0, 0), rect, 1)

        # Draw text
        text = font.render(name, True, (255, 255, 255))
        surface.blit(text, (x_offset + 20, y_pos))

        y_pos += 20


def play_with_collision_debug(
    level: Optional[str] = None,
    character: Optional[Union[str, int]] = None,
    custom_rom: Optional[str] = None,
    custom_state: Optional[str] = None,
    scale: int = DEFAULT_SCALE,
) -> None:
    """Play SMB2 with side-by-side collision visualization."""
    pygame.init()

    # Create initialization config
    if custom_rom:
        config = InitConfig(rom_path=custom_rom, save_state_path=custom_state)
    else:
        config = InitConfig(level=level or "1-1", character=character or "luigi")

    print(config.describe())

    # Create env
    env = SuperMarioBros2Env(init_config=config)

    # Setup pygame - wider window for side-by-side display
    game_width = SCREEN_WIDTH * scale
    game_height = SCREEN_HEIGHT * scale
    collision_width = 16 * 20  # 16 tiles * 20 pixels per tile
    collision_height = 16 * 20  # 16 tiles * 20 pixels per tile (now 16x16)

    total_width = game_width + collision_width + 200  # Extra space for legend
    total_height = max(game_height, collision_height) + 100  # Extra space for info

    info_height = get_required_info_height(scale)
    screen = pygame.display.set_mode((total_width, total_height + info_height))
    pygame.display.set_caption("SMB2 - Game + Collision Debug")

    font_size = FONT_SIZE_BASE * scale // 2
    font = pygame.font.Font(None, font_size)
    small_font = pygame.font.Font(None, 16)
    clock = pygame.time.Clock()

    # Reset environment
    obs, info = env.reset()

    # Game loop
    running = True
    paused = False
    game_over = False

    print("Controls:")
    print("    Arrow Keys: Move")
    print("    Z: A button (Jump)")
    print("    X: B button (Pick up/Throw)")
    print("    P: Pause")
    print("    R: Reset")
    print("    ESC: Quit")
    print()
    print("Collision Map Legend:")
    print("    Light Gray: Empty space")
    print("    Brown: Solid blocks")
    print("    Saddle Brown: Platforms")
    print("    Forest Green: Vines")
    print("    Gold: Doors/Keys")
    print("    Red: Enemies")
    print("    White circle with red border: Player position")

    while running:
        # Handle events
        keys_pressed = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    game_over = False
                    print("Game reset!")

        if not paused and not game_over:
            # Get keyboard state
            keys = pygame.key.get_pressed()

            # Check both keyboard mappings
            for key, action in {**KEYBOARD_MAPPING, **ALT_KEYBOARD_MAPPING}.items():
                if keys[key]:
                    if action not in keys_pressed:
                        keys_pressed.append(action)

            # Convert to action based on environment type
            buttons = actions_to_buttons(keys_pressed)
            action = 0  # Default to NOOP

            # Map button combinations to simple actions (same as original)
            if buttons[5] and buttons[0]:  # DOWN + A (super jump)
                action = 11
            elif buttons[6] and buttons[0]:  # LEFT + A
                action = 7
            elif buttons[7] and buttons[0]:  # RIGHT + A
                action = 6
            elif buttons[6] and buttons[1]:  # LEFT + B
                action = 9
            elif buttons[7] and buttons[1]:  # RIGHT + B
                action = 8
            elif buttons[5]:  # DOWN (crouch/charge)
                action = 10
            elif buttons[6]:  # LEFT
                action = 2
            elif buttons[7]:  # RIGHT
                action = 1
            elif buttons[4]:  # UP
                action = 3
            elif buttons[0]:  # A
                action = 4
            elif buttons[1]:  # B
                action = 5

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

        # Clear screen
        screen.fill((40, 40, 40))

        # Render game on the left side
        game_surface = pygame.Surface((game_width, game_height))
        render_frame(game_surface, obs, game_width, game_height)
        screen.blit(game_surface, (10, 10))

        # Get and render collision map on the right side
        collision_map = env.collision_map
        collision_x_offset = game_width + 30
        collision_y_offset = 10
        tile_size = 20

        # Draw collision map
        draw_collision_map(screen, collision_map, collision_x_offset, collision_y_offset, tile_size)

        # Draw player position
        draw_player_position(screen, env, collision_x_offset, collision_y_offset, tile_size)

        # Draw collision map title
        title_text = font.render("Collision Map", True, (255, 255, 255))
        screen.blit(title_text, (collision_x_offset, collision_y_offset - 30))

        # Draw legend
        legend_x = collision_x_offset + collision_width + 10
        legend_y = collision_y_offset
        legend_title = small_font.render("Legend:", True, (255, 255, 255))
        screen.blit(legend_title, (legend_x, legend_y))
        draw_legend(screen, small_font, legend_x, legend_y + 20)

        # Draw player info and collision map stats
        if hasattr(env, 'x_position') and hasattr(env, 'y_position'):
            # Count tile types
            unique, counts = np.unique(collision_map, return_counts=True)
            tile_counts = dict(zip(unique, counts))

            player_info = [
                f"Player Pos: ({env.x_position}, {env.y_position})",
                f"Tile Pos: ({env.x_position // 16}, {env.y_position // 16})",
                f"Can Climb: {env.can_climb if hasattr(env, 'can_climb') else 'N/A'}",
                f"On Special Surface: {env.is_on_special_surface if hasattr(env, 'is_on_special_surface') else 'N/A'}",
                "",
                f"Tile Counts:",
                f"  Empty: {tile_counts.get(0, 0)}",
                f"  Solid: {tile_counts.get(1, 0)}",
                f"  Platform: {tile_counts.get(2, 0)}",
                f"  Enemy: {tile_counts.get(18, 0)}",
            ]

            info_y = collision_y_offset + collision_height + 20
            for i, text in enumerate(player_info):
                if text:  # Skip empty strings
                    rendered_text = small_font.render(text, True, (255, 255, 255))
                    screen.blit(rendered_text, (collision_x_offset, info_y + i * 16))

        # Draw game info panel at bottom
        create_info_panel(screen, info, font, total_height, total_width)

        # Draw pause indicator
        if paused:
            pause_text = font.render("PAUSED", True, (255, 255, 0))
            text_rect = pause_text.get_rect(center=(total_width // 2, total_height // 2))
            screen.blit(pause_text, text_rect)

        # Update display
        pygame.display.flip()
        clock.tick(60)  # 60 FPS

    # Cleanup
    env.close()
    pygame.quit()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Play SMB2 with collision visualization")

    # Character/Level mode arguments
    parser.add_argument("--level", type=str, default="1-1", help="Level to play")
    parser.add_argument(
        "--char",
        type=str,
        default="luigi",
        choices=["mario", "luigi", "peach", "toad"],
        help="Character"
    )

    # Custom ROM mode arguments
    parser.add_argument("--custom-rom", type=str, help="Custom ROM file path")
    parser.add_argument("--custom-state", type=str, help="Custom save state file path")

    # Display options
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE, help="Display scale factor")

    args = parser.parse_args()

    try:
        if args.custom_rom:
            play_with_collision_debug(
                custom_rom=args.custom_rom,
                custom_state=args.custom_state,
                scale=args.scale,
            )
        else:
            play_with_collision_debug(
                level=args.level,
                character=args.char,
                scale=args.scale,
            )
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

