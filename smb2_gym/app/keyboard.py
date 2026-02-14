"""Keyboard mappings for human play."""

import pygame

from ..actions import actions_to_buttons

KEYBOARD_MAPPING = {
    pygame.K_RIGHT: 'right',
    pygame.K_LEFT: 'left',
    pygame.K_DOWN: 'down',
    pygame.K_UP: 'up',
    pygame.K_z: 'a',
    pygame.K_x: 'b',
    pygame.K_RETURN: 'start',
    pygame.K_RSHIFT: 'select',
}


def get_action_from_keyboard() -> int:
    """Get game action from current keyboard state.

    Returns:
        Action index (0-11)
    """
    keys = pygame.key.get_pressed()
    keys_pressed = []

    # Check keyboard mappings
    for key, action in KEYBOARD_MAPPING.items():
        if keys[key]:
            if action not in keys_pressed:
                keys_pressed.append(action)

    # Convert to button states
    buttons = actions_to_buttons(keys_pressed)

    # Map button combinations to simple actions
    if buttons[5] and buttons[0]:  # DOWN + A (super jump if charged)
        return 11
    elif buttons[6] and buttons[0]:  # LEFT + A
        return 7
    elif buttons[7] and buttons[0]:  # RIGHT + A
        return 6
    elif buttons[6] and buttons[1]:  # LEFT + B
        return 9
    elif buttons[7] and buttons[1]:  # RIGHT + B
        return 8
    elif buttons[5]:  # DOWN (crouch/charge)
        return 10
    elif buttons[6]:  # LEFT
        return 2
    elif buttons[7]:  # RIGHT
        return 1
    elif buttons[4]:  # UP
        return 3
    elif buttons[0]:  # A
        return 4
    elif buttons[1]:  # B
        return 5

    return 0  # NOOP
