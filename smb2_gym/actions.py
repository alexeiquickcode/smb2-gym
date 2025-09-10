"""Action mappings for Super Mario Bros 2."""

from typing import Literal

import numpy as np

from .constants import (
    BUTTON_A,
    BUTTON_B,
    BUTTON_DOWN,
    BUTTON_LEFT,
    BUTTON_RIGHT,
    BUTTON_SELECT,
    BUTTON_START,
    BUTTON_UP,
)

ActionType = Literal["all", "complex", "simple"]

SIMPLE_ACTIONS = [
    ['NOOP'],
    ['right'],
    ['left'],
    ['up'],  # Enter door
    ['A'],  # Jump
    ['B'],  # Pick up/throw
    ['right', 'A'],  # Jump right
    ['left', 'A'],  # Jump left  
    ['right', 'B'],  # Pick up right
    ['left', 'B'],  # Pick up left
    ['down'],  # Crouch/charge super jump (hold 77+ frames then press A)
]

COMPLEX_ACTIONS = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],  # Run right
    ['right', 'B'],  # Pick up right
    ['right', 'A', 'B'],  # Jump throw right
    ['left'],
    ['left', 'A'],  # Run left
    ['left', 'B'],  # Pick up left
    ['left', 'A', 'B'],  # Jump throw left
    ['down'],  # Charge super jump (hold 77+ frames then press A)
    ['down', 'B'],  # Pick up from ground
    ['up'],  # Enter door
    ['A'],  # Jump
    ['B'],  # Pick up/throw
    ['A', 'B'],  # Jump and throw
]

BUTTON_MAP = {
    'a': BUTTON_A,
    'A': BUTTON_A,
    'b': BUTTON_B,
    'B': BUTTON_B,
    'select': BUTTON_SELECT,
    'start': BUTTON_START,
    'up': BUTTON_UP,
    'down': BUTTON_DOWN,
    'left': BUTTON_LEFT,
    'right': BUTTON_RIGHT,
    'noop': None
}


def actions_to_buttons(actions: list[str]) -> np.ndarray:
    """Convert action list to button array.

    Args:
        actions: list of action strings (e.g., ['right', 'A'])

    Returns:
        Boolean array of shape (8,) for button states
    """
    buttons = np.zeros(8, dtype=bool)

    for action in actions:
        action_lower = action.lower()
        if action_lower in BUTTON_MAP:
            button_idx = BUTTON_MAP[action_lower]
            if button_idx is not None:
                buttons[button_idx] = True

    return buttons


def action_to_buttons(action: int) -> np.ndarray:
    """Convert discrete action to button array using all combinations.

    Args:
        action: Discrete action (0-255)

    Returns:
        Boolean array of shape (8,) for button states
    """
    # Each bit represents a button
    buttons = np.zeros(8, dtype=bool)
    for i in range(8):
        buttons[i] = bool(action & (1 << i))
    return buttons


def buttons_to_action(buttons: np.ndarray) -> int:
    """Convert button array to discrete action.

    Args:
        buttons: Boolean array of shape (8,)

    Returns:
        Discrete action (0-255)
    """
    action = 0
    for i in range(8):
        if buttons[i]:
            action |= (1 << i)
    return action


def buttons_to_action_index(buttons: np.ndarray, action_set: list[list[str]]) -> int:
    """Convert button array to action index for a given action set.

    Args:
        buttons: Boolean array of shape (8,)
        action_set: List of action combinations (e.g., SIMPLE_ACTIONS, COMPLEX_ACTIONS)

    Returns:
        Action index in the given action set
    """
    # Convert button state to action list
    pressed = []
    if buttons[BUTTON_RIGHT]:
        pressed.append('right')
    elif buttons[BUTTON_LEFT]:
        pressed.append('left')
    elif buttons[BUTTON_DOWN]:
        pressed.append('down')
    elif buttons[BUTTON_UP]:
        pressed.append('up')

    if buttons[BUTTON_A]:
        pressed.append('A')
    if buttons[BUTTON_B]:
        pressed.append('B')

    # Find matching action in the action set
    for i, action_combo in enumerate(action_set):
        if set(pressed) == set([a for a in action_combo if a.upper() != 'NOOP']):
            return i

    return 0  # NOOP


def get_action_meanings() -> list[list[str]]:
    """Get human-readable meanings for all 256 actions.

    Returns:
        list of action meanings
    """
    meanings = []
    button_names = ['A', 'B', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT']

    for action in range(256):
        buttons = action_to_buttons(action)
        pressed = [button_names[i] for i in range(8) if buttons[i]]
        meanings.append(pressed if pressed else ['NOOP'])

    return meanings
