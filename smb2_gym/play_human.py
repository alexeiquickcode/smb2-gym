"""Human-playable interface for Super Mario Bros 2."""

import argparse
import os
import sys
import traceback
from typing import (
    Optional,
    Union,
)

import numpy as np
import pygame

from smb2_gym.app import InitConfig
from smb2_gym.app.info_display import get_required_info_height
from smb2_gym.app.keyboard import get_action_from_keyboard
from smb2_gym.app.play_display import render_all
from smb2_gym.constants import (
    DEFAULT_SCALE,
    FONT_SIZE_BASE,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    WINDOW_CAPTION,
)
from smb2_gym.smb2_env import SuperMarioBros2Env


def _handle_events(
    env: SuperMarioBros2Env,
    paused: bool,
    game_over: bool,
) -> tuple[bool, bool, bool]:
    """Handle pygame events.

    Returns:
        Tuple of (running, paused, game_over)
    """
    running = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_p:
                paused = not paused
            elif event.key == pygame.K_r:
                env.reset()
                game_over = False
                print("Game reset!")
            elif event.key == pygame.K_F5:
                try:
                    env.save_state(0)
                    print("State saved to save_state_0.sav")
                except Exception as e:
                    print(f"Failed to save state: {e}")
            elif event.key == pygame.K_F9:
                try:
                    env.load_state(0)
                    print("State loaded from save_state_0.sav")
                except Exception as e:
                    print(f"Failed to load state: {e}")

    return running, paused, game_over


def _setup_pygame(
    scale: int,
) -> tuple[pygame.Surface, pygame.font.Font, pygame.font.Font, pygame.time.Clock, int, int, int,
           int]:
    """Setup pygame display, fonts, and clock.

    Returns:
        Tuple of (screen, font, small_font, clock, game_width, game_height, total_width, total_height)
    """
    pygame.init()

    # Calculate dimensions
    game_width = SCREEN_WIDTH * scale
    game_height = SCREEN_HEIGHT * scale
    semantic_map_width = 16 * 20  # 16 tiles * 20 pixels per tile
    semantic_map_height = 15 * 20  # 15 tiles * 20 pixels per tile

    total_width = game_width + semantic_map_width + 200  # Extra space for legend
    total_height = max(game_height, semantic_map_height) + 100  # Extra space for info

    # Create display
    info_height = get_required_info_height(scale)
    screen = pygame.display.set_mode((total_width, total_height + info_height))
    pygame.display.set_caption(WINDOW_CAPTION)

    # Create fonts
    font_size = FONT_SIZE_BASE * scale // 2
    font = pygame.font.Font(None, font_size)
    small_font = pygame.font.Font(None, 16)

    # Create clock
    clock = pygame.time.Clock()

    return screen, font, small_font, clock, game_width, game_height, total_width, total_height


# ------------------------------------------------------------------------------
# ---- Main Fn -----------------------------------------------------------------
# ------------------------------------------------------------------------------


def play_human(
    level: Optional[str] = None,
    character: Optional[Union[str, int]] = None,
    custom_rom: Optional[str] = None,
    custom_state: Optional[str] = None,
    scale: int = DEFAULT_SCALE,
) -> None:
    """Play Super Mario Bros 2 with keyboard controls.

    Args:
        level: Level to play (e.g., "1-1", "1-2") - used with character
        character: Character to play as ("mario", "luigi", "peach", or "toad") - used with level
        rom: ROM variant to use ("prg0", "prg0_edited") - used with save_state
        save_state: Save state file to load - used with rom
        custom_rom: Custom ROM file path - used with custom_state
        custom_state: Custom save state file path - used with custom_rom
        scale: Display scale factor
    """
    # Create initialisation config
    if custom_rom:
        config = InitConfig(rom_path=custom_rom, save_state_path=custom_state)
    else:
        config = InitConfig(level=level or "1-1", character=character or "luigi")

    # Print initialisation info
    print(config.describe())

    # Create env
    env = SuperMarioBros2Env(init_config=config)

    # Setup pygame
    screen, font, small_font, clock, game_width, game_height, total_width, total_height = _setup_pygame(
        scale
    )

    # Reset environment
    obs, info = env.reset()

    # Game loop
    running = True
    paused = False

    print("Controls:")
    print("    Arrow Keys: Move")
    print("    Z: A button (Jump)")
    print("    X: B button (Pick up/Throw)")
    print("    Enter: Start")
    print("    Right Shift: Select")
    print("    P: Pause")
    print("    R: Reset")
    print("    ESC: Quit")
    print("\nSave State:")
    print("    F5: Save state (creates save_state_0.sav)")
    print("    F9: Load state (loads save_state_0.sav)")

    game_over = False
    while running:
        running, paused, game_over = _handle_events(env, paused, game_over)

        if not paused and not game_over:
            action = get_action_from_keyboard()
            obs, reward, terminated, truncated, info = env.step(np.int64(action))

            if terminated or truncated:
                if info.get('level_completed'):
                    print("Level Completed! Continuing to next area...")
                else:
                    print("Game Over! Press R (in game window) to reset or ESC to quit.")
                    game_over = True

        render_all(
            screen,
            obs,
            env,
            info,
            game_width,
            game_height,
            total_width,
            total_height,
            font,
            small_font,
            paused,
        )

        # Update display
        pygame.display.flip()
        clock.tick(60)  # 60 FPS for human play

    env.close()
    pygame.quit()


# ------------------------------------------------------------------------------
# ---- Main entrypoint ---------------------------------------------------------
# ------------------------------------------------------------------------------


def main() -> None:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Play Super Mario Bros 2 with keyboard controls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Initialisation modes:
          1. Character/Level mode (default):
             --level 1-1 --char peach

          2. Built-in ROM variant mode:
             --rom prg0_edited --save-state easy_combined_curriculum.sav

          3. Custom ROM mode:
             --custom-rom /path/to/rom.nes --custom-state /path/to/save.sav

        Only one initialisation mode can be used at a time.
        """
    )

    # Character/Level mode arguments
    parser.add_argument(
        "--level",
        type=str,
        help="Level to play (e.g., 1-1, 1-2)",
    )
    parser.add_argument(
        "--char",
        type=str,
        choices=["mario", "luigi", "peach", "toad"],
        help="Character to play as"
    )

    # Built-in ROM mode arguments
    parser.add_argument(
        "--rom",
        type=str,
        choices=["prg0", "prg0_edited"],
        help="ROM variant to use",
    )
    parser.add_argument(
        "--save-state",
        type=str,
        help="Save state file to load",
    )

    # Custom ROM mode arguments
    parser.add_argument(
        "--custom-rom",
        type=str,
        help="Custom ROM file path",
    )
    parser.add_argument(
        "--custom-state",
        type=str,
        help="Custom save state file path",
    )

    # Common arguments
    parser.add_argument(
        "--scale",
        type=int,
        default=DEFAULT_SCALE,
        help="Display scale factor",
    )
    parser.add_argument(
        "--no-save-state",
        action="store_true",
        help="Start from beginning without loading save state"
    )

    args = parser.parse_args()

    try:
        # Create init config (validates arguments)
        if args.custom_rom:
            config = InitConfig(
                rom_path=args.custom_rom,
                save_state_path=args.custom_state if not args.no_save_state else None
            )
        elif args.rom:  # Built-in ROM variant mode
            # Construct paths for built-in ROM variants
            package_dir = os.path.dirname(os.path.abspath(__file__))  # This is smb2_gym/
            rom_path = os.path.join(
                package_dir, '_nes', args.rom, f'super_mario_bros_2_{args.rom}.nes'
            )
            save_path = None
            if args.save_state and not args.no_save_state:
                save_path = os.path.join(package_dir, '_nes', args.rom, 'saves', args.save_state)
            config = InitConfig(rom_path=rom_path, save_state_path=save_path)
        elif args.char is None:
            # No character specified - use select folder save states
            package_dir = os.path.dirname(os.path.abspath(__file__))  # This is smb2_gym/
            rom_path = os.path.join(package_dir, '_nes', 'prg0', 'super_mario_bros_2_prg0.nes')
            level = args.level or "1-1"
            save_path = None
            if not args.no_save_state:
                save_path = os.path.join(
                    package_dir, '_nes', 'prg0', 'saves', 'select', f'{level}.sav'
                )
            config = InitConfig(rom_path=rom_path, save_state_path=save_path)
        else:
            config = InitConfig(level=args.level, character=args.char)

        if args.no_save_state:
            print("Starting from beginning (no save state)")
            if args.custom_rom:
                print("Using custom ROM without save state")
            else:
                print("Auto-navigating to character selection screen...")
                print("Use arrow keys to select character, then press Z (A button) to start!")

        # Call play_human with appropriate parameters based on mode
        if args.custom_rom:
            play_human(
                custom_rom=args.custom_rom,
                custom_state=args.custom_state if not args.no_save_state else None,
                scale=args.scale,
            )
        elif args.rom:  # Built-in ROM variant mode
            play_human(
                custom_rom=config.rom_path,
                custom_state=config.save_state_path,
                scale=args.scale,
            )
        elif args.char is None:
            # Use select folder save states
            play_human(
                custom_rom=config.rom_path,
                custom_state=config.save_state_path,
                scale=args.scale,
            )
        else:
            play_human(
                level=args.level,
                character=args.char,
                scale=args.scale,
            )
    except ValueError as e:
        parser.error(str(e))
    except FileNotFoundError as e:
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
