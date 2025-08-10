"""Keyboard mappings for human play."""

import pygame

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

ALT_KEYBOARD_MAPPING = {
    pygame.K_d: 'right',
    pygame.K_a: 'left',
    pygame.K_s: 'down',
    pygame.K_w: 'up',
    pygame.K_j: 'a',
    pygame.K_k: 'b',
    pygame.K_SPACE: 'start',
    pygame.K_TAB: 'select',
}
