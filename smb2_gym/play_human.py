"""Human-playable interface for Super Mario Bros 2."""

import argparse
import os
import sys
from typing import (
    Optional,
    Union,
)

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
    WINDOW_CAPTION,
)
from smb2_gym.smb2_env import SuperMarioBros2Env


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
    pygame.init()

    # Create initialization config
    if custom_rom:
        config = InitConfig(rom_path=custom_rom, save_state_path=custom_state)
    else:
        config = InitConfig(level=level or "1-1", character=character or "luigi")

    # Print initialization info
    print(config.describe())

    # Create env
    env = SuperMarioBros2Env(init_config=config)

    # Setup pygame
    width, height = SCREEN_WIDTH * scale, SCREEN_HEIGHT * scale
    info_height = get_required_info_height(scale)
    screen = pygame.display.set_mode((width, height + info_height))
    pygame.display.set_caption(WINDOW_CAPTION)
    font_size = FONT_SIZE_BASE * scale // 2
    font = pygame.font.Font(None, font_size)
    clock = pygame.time.Clock()  # Clock for FPS

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
    print("\nAlternative controls:")
    print("    WASD: Move")
    print("    J: A button")
    print("    K: B button")
    print("\nSave State:")
    print("    F5: Save state (creates save_state_0.sav)")
    print("    F9: Load state (loads save_state_0.sav)")

    game_over = False
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

            # Map button combinations to simple actions
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

            if terminated or truncated:
                if info.get('level_completed'):
                    print("Level Completed! Press R (in game window) to reset or ESC to quit.")
                else:
                    print("Game Over! Press R (in game window) to reset or ESC to quit.")
                game_over = True
                # Don't auto-reset, wait for user/player input

        render_frame(screen, obs, width, height)
        create_info_panel(screen, info, font, height, width)

        # Draw pause indicator
        if paused:
            pause_text = font.render("PAUSED", True, (255, 255, 0))
            text_rect = pause_text.get_rect(center=(width // 2, height // 2))
            screen.blit(pause_text, text_rect)

        # Update display
        pygame.display.flip()
        clock.tick(60)  # 60 FPS for human play

    # Cleanup
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
        Initialization modes:
          1. Character/Level mode (default):
             --level 1-1 --char peach

          2. Built-in ROM variant mode:
             --rom prg0_edited --save-state easy_combined_curriculum.sav

          3. Custom ROM mode:
             --custom-rom /path/to/rom.nes --custom-state /path/to/save.sav

        Only one initialization mode can be used at a time.
        """
    )

    # Character/Level mode arguments
    parser.add_argument("--level", type=str, help="Level to play (e.g., 1-1, 1-2)")
    parser.add_argument(
        "--char",
        type=str,
        choices=["mario", "luigi", "peach", "toad"],
        help="Character to play as"
    )

    # Built-in ROM mode arguments
    parser.add_argument(
        "--rom", type=str, choices=["prg0", "prg0_edited"], help="ROM variant to use"
    )
    parser.add_argument("--save-state", type=str, help="Save state file to load")

    # Custom ROM mode arguments
    parser.add_argument("--custom-rom", type=str, help="Custom ROM file path")
    parser.add_argument("--custom-state", type=str, help="Custom save state file path")

    # Common arguments
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE, help="Display scale factor")
    parser.add_argument(
        "--no-save-state",
        action="store_true",
        help="Start from beginning without loading save state"
    )

    args = parser.parse_args()

    try:
        # Create initialization config (validates arguments)
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
        else:
            config = InitConfig(level=args.level, character=args.char)

        if args.no_save_state:
            print("Starting from beginning (no save state)")
            if args.custom_rom:
                print("Using custom ROM without save state")
            else:
                print("Auto-navigating to character selection screen...")
                print("Use arrow keys to select character, then press Z (A button) to start!")

        # Convert config to play_human parameters for backwards compatibility
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
        else:
            play_human(
                level=args.level,
                character=args.char,
                scale=args.scale,
            )
    except ValueError as e:
        parser.error(str(e))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
