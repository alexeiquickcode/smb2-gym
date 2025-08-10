"""Human-playable interface for Super Mario Bros 2."""

import argparse
import sys

import pygame

from smb2_gym.actions import actions_to_buttons
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
    CHARACTER_NAMES,
    DEFAULT_SCALE,
    FONT_SIZE_BASE,
    FPS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    WINDOW_CAPTION,
)
from smb2_gym.smb2_env import SuperMarioBros2Env


def play_human(level: str = "1-1", character: str = "mario", scale: int = DEFAULT_SCALE):
    """Play Super Mario Bros 2 with keyboard controls.

    Args:
        level: Level to play (e.g., "1-1", "1-2")
        character: Character to play as ("mario", "luigi", "peach", or "toad")
        scale: Display scale factor
    """
    pygame.init()

    # Create reverse mapping from CHARACTER_NAMES
    char_map = {name.lower(): idx for idx, name in CHARACTER_NAMES.items()}
    char_index = char_map.get(character.lower(), 0)

    env = SuperMarioBros2Env(level=level, character=char_index)

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

        if not paused:
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
                print("Game Over! Press R to reset or ESC to quit.")
                # Don't auto-reset, wait for user input

        render_frame(screen, obs, width, height)

        create_info_panel(screen, info, font, height, width)

        # Draw pause indicator
        if paused:
            pause_text = font.render("PAUSED", True, (255, 255, 0))
            text_rect = pause_text.get_rect(center=(width // 2, height // 2))
            screen.blit(pause_text, text_rect)

        # Update display
        pygame.display.flip()
        clock.tick(FPS)  # Set FPS

    # Cleanup
    env.close()
    pygame.quit()


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="Play Super Mario Bros 2 with keyboard controls")
    parser.add_argument("--level", type=str, default="1-1", help="Level to play (e.g., 1-1, 1-2)")
    parser.add_argument(
        "--char",
        type=str,
        default="mario",
        choices=["mario", "luigi", "peach", "toad"],
        help="Character to play as"
    )
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE, help="Display scale factor")
    args = parser.parse_args()

    print(f"Playing as {args.char} on level {args.level} with scale {args.scale}")

    try:
        play_human(args.level, args.char, args.scale)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
