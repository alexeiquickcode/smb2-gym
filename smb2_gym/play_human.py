"""Human-playable interface for Super Mario Bros 2."""

import argparse
import os
import sys
import traceback

import numpy as np
import pygame

from smb2_gym.app import InitConfig
from smb2_gym.app.keyboard import get_action_from_keyboard
from smb2_gym.app.layout import MIN_WINDOW
from smb2_gym.app.play_display import PlayUI
from smb2_gym.constants import (
    DEFAULT_SCALE,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    WINDOW_CAPTION,
)
from smb2_gym.smb2_env import SuperMarioBros2Env


# Space the sidebar and stats panels need alongside a game view of a given scale
SIDEBAR_ALLOWANCE = 300
STATS_ALLOWANCE = 250


def _initial_window_size(scale: int) -> tuple[int, int]:
    """Pick a starting window size that fits the game at `scale` plus the panels."""
    width = SCREEN_WIDTH * scale + SIDEBAR_ALLOWANCE
    height = SCREEN_HEIGHT * scale + STATS_ALLOWANCE
    return max(width, MIN_WINDOW[0]), max(height, MIN_WINDOW[1])


def _handle_events(
    env: SuperMarioBros2Env,
    ui: PlayUI,
    paused: bool,
    game_over: bool,
) -> tuple[bool, bool, bool]:
    """Handle pygame events, including window resizing and panel hotkeys.

    Returns:
        Tuple of (running, paused, game_over)
    """
    running = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.VIDEORESIZE:
            ui.handle_resize(event.w, event.h)

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            ui.handle_click(event.pos)

        elif event.type == pygame.KEYDOWN:
            # The options menu takes keys first while it is open, so its
            # letter shortcuts don't also trigger the panel toggles below.
            if ui.handle_options_key(event.key):
                continue

            # Game controls
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_p:
                paused = not paused
                ui.toast("Paused" if paused else "Resumed")
            elif event.key == pygame.K_r:
                env.reset()
                game_over = False
                ui.toast("Level reset")
            elif event.key == pygame.K_F5:
                try:
                    env.save_state(0)
                    ui.toast("State saved to slot 0")
                except Exception as e:
                    ui.toast(f"Save failed: {e}")
            elif event.key == pygame.K_F9:
                try:
                    env.load_state(0)
                    ui.toast("State loaded from slot 0")
                except Exception as e:
                    ui.toast(f"Load failed: {e}")

            # Interface controls
            elif event.key in (pygame.K_F1, pygame.K_h):
                ui.toggle_panel("help")
            elif event.key == pygame.K_F2:
                ui.toggle_options()
            elif event.key == pygame.K_m:
                ui.toggle_panel("semantic_map")
            elif event.key == pygame.K_l:
                ui.toggle_panel("legend")
            elif event.key == pygame.K_i:
                ui.toggle_panel("stats")
            elif event.key == pygame.K_TAB:
                mods = pygame.key.get_mods()
                ui.next_tab(-1 if mods & pygame.KMOD_SHIFT else 1)
            elif event.key == pygame.K_F11:
                ui.toggle_fullscreen()

    return running, paused, game_over


# ------------------------------------------------------------------------------
# ---- Main Fn -----------------------------------------------------------------
# ------------------------------------------------------------------------------


def play_human(
    level: str | None = None,
    character: str | int | None = None,
    custom_rom: str | None = None,
    custom_state: str | None = None,
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

    # Setup pygame and the resizable UI
    pygame.init()
    width, height = _initial_window_size(scale)
    ui = PlayUI(width, height, caption=WINDOW_CAPTION)
    clock = pygame.time.Clock()

    # Reset environment
    obs, info = env.reset()

    print("\nPress F1 (or H) in the game window for the full controls list.")
    ui.toast("Press F1 for controls")

    running = True
    paused = False
    game_over = False

    while running:
        running, paused, game_over = _handle_events(env, ui, paused, game_over)

        # A full-window overlay holds the game still. Otherwise it keeps running
        # behind the overlay and the overlay's keys double as controller input -
        # pressing S to toggle scaling would also be read as a button.
        if not paused and not game_over and not ui.blocks_input:
            action = get_action_from_keyboard()
            obs, reward, terminated, truncated, info = env.step(np.int64(action))

            if ui.settings.log_rewards:
                print(f"step reward={reward:+.3f}  x={info['pos'].x_global}")

            if terminated or truncated:
                if info.get('level_completed'):
                    ui.toast("Level complete — continuing")
                else:
                    ui.toast("Game over — press R to reset")
                    game_over = True

        ui.render(
            obs,
            env,
            info,
            paused=paused,
            game_over=game_over,
            fps=clock.get_fps(),
            dt=clock.get_time() / 1000.0,
        )

        # 0 means uncapped; tick(0) would busy-wait, so skip the cap entirely.
        if ui.settings.target_fps:
            clock.tick(ui.settings.target_fps)
        else:
            clock.tick()

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
        """,
    )

    # Character/Level mode arguments
    parser.add_argument(
        "--level",
        type=str,
        help="Level to play (e.g., 1-1, 1-2)",
    )
    parser.add_argument(
        "--char", type=str, choices=["mario", "luigi", "peach", "toad"], help="Character to play as"
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
        help="Start from beginning without loading save state",
    )

    args = parser.parse_args()

    try:
        # Create init config (validates arguments)
        if args.custom_rom:
            config = InitConfig(
                rom_path=args.custom_rom,
                save_state_path=args.custom_state if not args.no_save_state else None,
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
    except FileNotFoundError:
        traceback.print_exc()
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
