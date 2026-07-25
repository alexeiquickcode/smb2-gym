"""Individual panel renderers for the human-play interface.

Each function draws into a rect handed to it by `layout.compute_layout`, so
none of them assume a fixed window size.
"""

from collections.abc import Sequence

import numpy as np
import pygame

from ..constants import (
    NO_OBJECT,
    TILE_COLORS,
    FineTileType,
)
from .theme import (
    ACCENT,
    BAD,
    BORDER,
    BORDER_STRONG,
    GOOD,
    PANEL,
    PANEL_ALT,
    SECTION_ACCENTS,
    TEXT,
    TEXT_DIM,
    TEXT_FAINT,
    WARN,
    Fonts,
    Metrics,
    blit_clipped,
    draw_card,
    draw_card_header,
    truncate,
)


# ------------------------------------------------------------------------------
# ---- Status bar --------------------------------------------------------------
# ------------------------------------------------------------------------------


def draw_status_bar(
    surface: pygame.Surface,
    rect: pygame.Rect,
    fonts: Fonts,
    metrics: Metrics,
    *,
    title: str,
    level: str,
    character: str,
    lives: int,
    hearts: int,
    coins: int,
    fps: float,
    paused: bool,
    game_over: bool,
) -> None:
    """Draw the top bar: title, run state, and the vitals worth watching."""
    draw_card(surface, rect, metrics, fill=PANEL)

    x = rect.x + metrics.pad
    centre_y = rect.centery

    def put(text: str, color: tuple[int, int, int], font: pygame.font.Font) -> None:
        nonlocal x
        surf = font.render(text, True, color)
        if x + surf.get_width() > rect.right - metrics.pad:
            return
        surface.blit(surf, (x, centre_y - surf.get_height() // 2))
        x += surf.get_width() + metrics.pad

    put(title, TEXT, fonts.title)

    # Run-state pill
    if game_over:
        state_text, state_color = "GAME OVER", BAD
    elif paused:
        state_text, state_color = "PAUSED", WARN
    else:
        state_text, state_color = "PLAYING", GOOD
    _pill(surface, (x, centre_y), state_text, state_color, fonts.small, metrics)
    x += fonts.small.size(state_text)[0] + 3 * metrics.pad

    put(f"{level}", ACCENT, fonts.body)
    put(f"{character}", TEXT_DIM, fonts.body)
    # "HP" rather than a heart glyph: the filled/hollow heart characters render
    # inconsistently across monospace fonts, and some substitute a wide emoji.
    put(f"HP {hearts}/4", BAD if hearts <= 1 else TEXT, fonts.body)
    put(f"LIVES {lives}", TEXT, fonts.body)
    put(f"COINS {coins}", TEXT, fonts.body)

    # FPS and the overlay hints sit right-aligned. These drop off from the left
    # as the window narrows, so the FPS readout survives longest.
    right_bits = [
        (f"{fps:4.0f} FPS", GOOD if fps >= 55 else WARN if fps >= 30 else BAD, fonts.small),
        ("[F1] Help", TEXT_FAINT, fonts.small),
        ("[F2] Options", TEXT_FAINT, fonts.small),
    ]
    rx = rect.right - metrics.pad
    for text, color, font in reversed(right_bits):
        surf = font.render(text, True, color)
        rx -= surf.get_width()
        if rx < x:
            break
        surface.blit(surf, (rx, centre_y - surf.get_height() // 2))
        rx -= metrics.pad


def _pill(
    surface: pygame.Surface,
    centre_left: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    font: pygame.font.Font,
    metrics: Metrics,
) -> None:
    """Draw a small rounded status chip anchored at its middle-left."""
    text_surface = font.render(text, True, color)
    pad = max(4, int(6 * metrics.ui_scale))
    rect = pygame.Rect(
        centre_left[0],
        centre_left[1] - text_surface.get_height() // 2 - pad // 2,
        text_surface.get_width() + 2 * pad,
        text_surface.get_height() + pad,
    )
    tint = tuple(int(c * 0.25) for c in color)
    pygame.draw.rect(surface, tint, rect, border_radius=rect.height // 2)
    pygame.draw.rect(surface, color, rect, width=1, border_radius=rect.height // 2)
    surface.blit(text_surface, (rect.x + pad, rect.y + pad // 2))


# ------------------------------------------------------------------------------
# ---- Game view ---------------------------------------------------------------
# ------------------------------------------------------------------------------


def draw_game_panel(
    surface: pygame.Surface,
    card: pygame.Rect,
    view: pygame.Rect,
    obs: np.ndarray,
    fonts: Fonts,
    metrics: Metrics,
    *,
    paused: bool,
    game_over: bool,
    integer_scaling: bool = False,
) -> None:
    """Draw the emulator frame, scaled to fit while keeping its aspect ratio."""
    draw_card(surface, card, metrics, fill=(8, 9, 12))

    if view.width <= 0 or view.height <= 0:
        return

    frame = pygame.Surface((256, 240), depth=24)
    if obs.ndim == 2:  # grayscale
        frame_data = np.transpose(np.stack([obs, obs, obs], axis=-1), (1, 0, 2))
    else:
        frame_data = np.transpose(obs, (1, 0, 2))
    pygame.surfarray.blit_array(frame, frame_data)

    target = view
    if integer_scaling:
        # Round the scale down to a whole number so every NES pixel becomes the
        # same number of screen pixels. A fractional factor makes some pixels a
        # row wider than their neighbours, which shows up worst on text.
        factor = min(view.width // 256, view.height // 240)
        if factor >= 1:
            target = pygame.Rect(0, 0, 256 * factor, 240 * factor)
            target.center = view.center

    # Nearest-neighbour keeps the pixel art crisp; smoothscale would blur it
    surface.blit(pygame.transform.scale(frame, (target.width, target.height)), target.topleft)
    pygame.draw.rect(surface, BORDER_STRONG, target.inflate(2, 2), width=1)

    if paused or game_over:
        _draw_game_overlay(surface, target, fonts, metrics, game_over=game_over)


def _draw_game_overlay(
    surface: pygame.Surface,
    view: pygame.Rect,
    fonts: Fonts,
    metrics: Metrics,
    *,
    game_over: bool,
) -> None:
    """Dim the game and state why it isn't advancing.

    Only reached when the game is paused or over; `game_over` takes precedence
    since it is the less recoverable state.
    """
    veil = pygame.Surface(view.size, pygame.SRCALPHA)
    veil.fill((6, 7, 10, 170))
    surface.blit(veil, view.topleft)

    if game_over:
        title, subtitle, color = "GAME OVER", "[R] to reset  ·  [Esc] to quit", BAD
    else:
        title, subtitle, color = "PAUSED", "[P] to resume", WARN

    title_surface = fonts.title.render(title, True, color)
    sub_surface = fonts.small.render(subtitle, True, TEXT_DIM)
    total_h = title_surface.get_height() + sub_surface.get_height() + metrics.pad
    top = view.centery - total_h // 2
    surface.blit(title_surface, (view.centerx - title_surface.get_width() // 2, top))
    surface.blit(
        sub_surface,
        (
            view.centerx - sub_surface.get_width() // 2,
            top + title_surface.get_height() + metrics.pad,
        ),
    )


# ------------------------------------------------------------------------------
# ---- Semantic map ------------------------------------------------------------
# ------------------------------------------------------------------------------


def draw_semantic_panel(
    surface: pygame.Surface,
    card: pygame.Rect,
    fonts: Fonts,
    metrics: Metrics,
    semantic_map: np.ndarray,
    player_tiles: Sequence[tuple[int, int]],
    tile_size: int,
    *,
    show_collision: bool = True,
    show_grid: bool = True,
) -> None:
    """Draw the semantic tile map with the player's collision tiles marked.

    Terrain is drawn first and sprite objects inset over the top, so a cell
    holding both still shows the terrain as a ring around the object.
    """
    draw_card(surface, card, metrics)
    content = draw_card_header(
        surface, card, "SEMANTIC MAP", fonts, metrics, accent=ACCENT, hint="[M]"
    )
    if content.height <= 0 or tile_size <= 0:
        return

    rows, cols = semantic_map.shape
    grid_w, grid_h = cols * tile_size, rows * tile_size
    ox = content.x + (content.width - grid_w) // 2
    oy = content.y

    pygame.draw.rect(
        surface, (12, 13, 18), pygame.Rect(ox - 1, oy - 1, grid_w + 2, grid_h + 2), border_radius=3
    )

    # Below ~10px the lines eat most of the cell, so the grid is dropped
    grid_color = (0, 0, 0) if (show_grid and tile_size >= 10) else None
    for y in range(rows):
        for x in range(cols):
            cell = semantic_map[y, x]
            rect = pygame.Rect(ox + x * tile_size, oy + y * tile_size, tile_size, tile_size)

            terrain = FineTileType(int(cell['fine_type']))
            pygame.draw.rect(surface, TILE_COLORS.get(terrain, (200, 200, 200)), rect)

            if int(cell['object_id']) != NO_OBJECT:
                obj = FineTileType(int(cell['object_fine_type']))
                inset = max(1, tile_size // 6)
                pygame.draw.rect(
                    surface,
                    TILE_COLORS.get(obj, (255, 0, 0)),
                    rect.inflate(-inset * 2, -inset * 2),
                )

            if grid_color:
                pygame.draw.rect(surface, grid_color, rect, 1)

    if not show_collision:
        return

    for tile_x, tile_y in player_tiles:
        if not (0 <= tile_x < cols and 0 <= tile_y < rows):
            continue
        cx = ox + tile_x * tile_size + tile_size // 2
        cy = oy + tile_y * tile_size + tile_size // 2
        radius = max(2, tile_size // 3)
        pygame.draw.circle(surface, (255, 255, 255), (cx, cy), radius)
        pygame.draw.circle(surface, (255, 40, 40), (cx, cy), radius, max(1, tile_size // 10))


# ------------------------------------------------------------------------------
# ---- Legend ------------------------------------------------------------------
# ------------------------------------------------------------------------------


def draw_legend_panel(
    surface: pygame.Surface,
    card: pygame.Rect,
    fonts: Fonts,
    metrics: Metrics,
) -> None:
    """Draw the tile-colour legend, wrapped into as many columns as fit."""
    draw_card(surface, card, metrics)
    content = draw_card_header(surface, card, "LEGEND", fonts, metrics, accent=GOOD, hint="[L]")
    if content.height <= 0:
        return

    entries = [t for t in FineTileType if t != FineTileType.EMPTY]
    row_h = max(13, int(16 * metrics.ui_scale))
    swatch = max(7, int(10 * metrics.ui_scale))
    rows_per_col = max(1, content.height // row_h)
    n_cols = max(1, -(-len(entries) // rows_per_col))  # ceil division
    col_w = content.width // n_cols

    for i, tile_type in enumerate(entries):
        col, row = divmod(i, rows_per_col)
        if col >= n_cols:
            break
        x = content.x + col * col_w
        y = content.y + row * row_h
        if y + row_h > content.bottom:
            continue

        box = pygame.Rect(x, y + (row_h - swatch) // 2, swatch, swatch)
        pygame.draw.rect(surface, TILE_COLORS.get(tile_type, (128, 128, 128)), box, border_radius=2)
        pygame.draw.rect(surface, BORDER, box, width=1, border_radius=2)

        label_x = x + swatch + max(4, int(6 * metrics.ui_scale))
        label_w = col_w - (label_x - x) - metrics.pad // 2
        label = truncate(fonts.tiny, tile_type.name, label_w)
        surface.blit(fonts.tiny.render(label, True, TEXT_DIM), (label_x, y))


# ------------------------------------------------------------------------------
# ---- Help overlay ------------------------------------------------------------
# ------------------------------------------------------------------------------

HELP_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "MOVEMENT",
        [
            ("[Arrows]", "Move / crouch"),
            ("[Z]", "A button — jump"),
            ("[X]", "B button — pick up / throw"),
            ("[Enter]", "Start"),
            ("[R Shift]", "Select"),
        ],
    ),
    (
        "GAME",
        [
            ("[P]", "Pause / resume"),
            ("[R]", "Reset level"),
            ("[F5]/[F9]", "Save / load state 0"),
            ("[Esc]", "Quit"),
        ],
    ),
    (
        "INTERFACE",
        [
            ("[F1] or [H]", "Toggle this help"),
            ("[F2]", "Options menu"),
            ("[M]", "Toggle semantic map"),
            ("[L]", "Toggle legend"),
            ("[I]", "Toggle stats panel"),
            ("[Tab]", "Next stats tab"),
            ("[F11]", "Toggle fullscreen"),
        ],
    ),
]


def draw_help_overlay(
    surface: pygame.Surface,
    window: pygame.Rect,
    fonts: Fonts,
    metrics: Metrics,
) -> None:
    """Draw the controls cheatsheet centred over the whole window."""
    veil = pygame.Surface(window.size, pygame.SRCALPHA)
    veil.fill((8, 9, 13, 200))
    surface.blit(veil, (0, 0))

    row_h = metrics.row_height
    n_rows = sum(len(rows) + 2 for _, rows in HELP_SECTIONS)
    card_h = min(window.height - 4 * metrics.gap, n_rows * row_h + 5 * metrics.pad)
    card_w = min(window.width - 4 * metrics.gap, int(430 * metrics.ui_scale))
    card = pygame.Rect(0, 0, card_w, card_h)
    card.center = window.center

    draw_card(surface, card, metrics, fill=PANEL, border=BORDER_STRONG)
    content = draw_card_header(
        surface, card, "CONTROLS", fonts, metrics, accent=ACCENT, hint="[F1] or [H] to close"
    )

    y = content.y
    key_w = int(content.width * 0.38)
    for title, rows in HELP_SECTIONS:
        if y + row_h > content.bottom:
            break
        surface.blit(
            fonts.small.render(title, True, SECTION_ACCENTS.get(title, ACCENT)), (content.x, y)
        )
        y += row_h
        for key, description in rows:
            if y + row_h > content.bottom:
                break
            blit_clipped(surface, fonts.small.render(key, True, TEXT), (content.x, y), key_w)
            blit_clipped(
                surface,
                fonts.small.render(description, True, TEXT_DIM),
                (content.x + key_w, y),
                content.width - key_w,
            )
            y += row_h
        y += row_h // 2


# ------------------------------------------------------------------------------
# ---- Toast -------------------------------------------------------------------
# ------------------------------------------------------------------------------


def draw_toast(
    surface: pygame.Surface,
    window: pygame.Rect,
    fonts: Fonts,
    metrics: Metrics,
    message: str,
    alpha: float,
    color: tuple[int, int, int] = ACCENT,
) -> None:
    """Draw a transient message near the bottom of the window.

    Used for actions with no other visible feedback — saving a state, toggling a
    panel — which previously only printed to stdout.
    """
    if alpha <= 0 or not message:
        return

    text = fonts.small.render(message, True, TEXT)
    pad = metrics.pad
    box = pygame.Rect(0, 0, text.get_width() + 4 * pad, text.get_height() + 2 * pad)
    box.centerx = window.centerx
    box.bottom = window.bottom - 3 * metrics.gap

    layer = pygame.Surface(box.size, pygame.SRCALPHA)
    a = int(max(0.0, min(1.0, alpha)) * 235)
    pygame.draw.rect(layer, (*PANEL_ALT, a), layer.get_rect(), border_radius=metrics.radius)
    pygame.draw.rect(layer, (*color, a), layer.get_rect(), width=1, border_radius=metrics.radius)
    text.set_alpha(a)
    layer.blit(text, (2 * pad, pad))
    surface.blit(layer, box.topleft)


# ------------------------------------------------------------------------------
# ---- Empty-state helper ------------------------------------------------------
# ------------------------------------------------------------------------------


def draw_placeholder(
    surface: pygame.Surface,
    rect: pygame.Rect,
    fonts: Fonts,
    message: str,
) -> None:
    """Centre a dim message in a rect that has nothing else to show."""
    text = fonts.small.render(message, True, TEXT_FAINT)
    surface.blit(
        text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2)
    )
