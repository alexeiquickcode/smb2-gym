"""Responsive layout for the human-play interface.

Given a window size and which panels are visible, produce the rect for every
region. Everything downstream draws into the rects it is handed, so resizing is
purely a matter of recomputing this.
"""

from dataclasses import dataclass

import pygame

from .theme import Metrics


# Native NES frame, used to keep the game view's aspect ratio correct
GAME_ASPECT = 256 / 240

# Sidebar width as a fraction of the window, and its hard bounds in pixels
SIDEBAR_FRACTION = 0.30
SIDEBAR_MIN = 190
SIDEBAR_MAX = 460

# Stats bar height as a fraction of the window, and its bounds
STATS_FRACTION = 0.32
STATS_MIN = 130
STATS_MAX = 460

# Below this window width the sidebar is dropped so the game keeps usable space
NARROW_WIDTH = 620

# Floor for the window we *request* at startup. It is deliberately not enforced
# on resize: tiling compositors size windows themselves, and fighting them by
# re-asserting a minimum causes an endless resize loop. The layout below copes
# with any size it is given.
MIN_WINDOW = (560, 420)


@dataclass
class PanelVisibility:
    """Which optional panels are currently shown."""

    sidebar: bool = True  # the semantic map column as a whole
    semantic_map: bool = True
    legend: bool = True
    stats: bool = True
    help: bool = False


@dataclass
class Layout:
    """Computed rects for one frame."""

    window: pygame.Rect
    status: pygame.Rect
    game_card: pygame.Rect
    game_view: pygame.Rect  # aspect-correct frame inside the game card
    sidebar: pygame.Rect | None
    semantic_card: pygame.Rect | None
    legend_card: pygame.Rect | None
    stats_card: pygame.Rect | None
    tile_size: int


def compute_ui_scale(width: int, height: int) -> float:
    """Pick a UI scale from the window size.

    Tied to the smaller dimension so text doesn't overflow its card when the
    window is wide but short.
    """
    scale = min(width / 1100.0, height / 780.0)
    return max(0.72, min(1.6, scale))


def compute_layout(
    width: int,
    height: int,
    visibility: PanelVisibility,
    metrics: Metrics,
    map_shape: tuple[int, int] = (15, 16),
) -> Layout:
    """Compute all region rects for the current window size.

    Args:
        width: Window width in pixels
        height: Window height in pixels
        visibility: Which optional panels are shown
        metrics: Spacing metrics for the current UI scale
        map_shape: (rows, cols) of the semantic map, used to size its tiles
    """
    gap = metrics.gap
    window = pygame.Rect(0, 0, width, height)

    # A tiling compositor can hand us any size at all, so every region below is
    # clamped rather than assuming a minimum window.
    # Status bar spans the top
    status = pygame.Rect(gap, gap, max(0, width - 2 * gap), metrics.status_height)
    content_top = status.bottom + gap
    content_height = max(0, height - content_top - gap)
    content = pygame.Rect(gap, content_top, max(0, width - 2 * gap), content_height)

    # Stats bar along the bottom
    stats_card: pygame.Rect | None = None
    if visibility.stats:
        stats_height = int(min(STATS_MAX, max(STATS_MIN, content.height * STATS_FRACTION)))
        stats_height = min(stats_height, max(0, content.height - 160))
        if stats_height > 60:
            stats_card = pygame.Rect(
                content.x, content.bottom - stats_height, content.width, stats_height
            )
            content = pygame.Rect(
                content.x, content.y, content.width, content.height - stats_height - gap
            )

    # Sidebar down the right. Dropped entirely on narrow windows.
    sidebar: pygame.Rect | None = None
    semantic_card: pygame.Rect | None = None
    legend_card: pygame.Rect | None = None
    show_sidebar = (
        visibility.sidebar
        and (visibility.semantic_map or visibility.legend)
        and width >= NARROW_WIDTH
    )

    game_area = content
    if show_sidebar:
        sidebar_width = int(min(SIDEBAR_MAX, max(SIDEBAR_MIN, width * SIDEBAR_FRACTION)))

        # The game view is capped by the available height, so any width the game
        # card can't use is given to the sidebar instead of sitting empty.
        game_needs = int((content.height - 2 * metrics.pad) * GAME_ASPECT) + 2 * metrics.pad
        slack = content.width - gap - game_needs - sidebar_width
        if slack > 0:
            sidebar_width = int(min(SIDEBAR_MAX, sidebar_width + slack))

        sidebar_width = min(sidebar_width, max(0, content.width - 300))
        if sidebar_width >= SIDEBAR_MIN:
            sidebar = pygame.Rect(
                content.right - sidebar_width, content.y, sidebar_width, content.height
            )
            game_area = pygame.Rect(
                content.x, content.y, content.width - sidebar_width - gap, content.height
            )
            semantic_card, legend_card = _split_sidebar(sidebar, visibility, metrics, map_shape)

    game_card = _hug_game_card(game_area, metrics)
    game_view = _fit_game_view(game_card, metrics)

    tile_size = 0
    if semantic_card is not None:
        rows, cols = map_shape
        inner_w = semantic_card.width - 2 * metrics.pad
        inner_h = semantic_card.height - metrics.header_height - 2 * metrics.pad
        tile_size = max(4, int(min(inner_w / max(1, cols), inner_h / max(1, rows))))

    return Layout(
        window=window,
        status=status,
        game_card=game_card,
        game_view=game_view,
        sidebar=sidebar,
        semantic_card=semantic_card,
        legend_card=legend_card,
        stats_card=stats_card,
        tile_size=tile_size,
    )


def _split_sidebar(
    sidebar: pygame.Rect,
    visibility: PanelVisibility,
    metrics: Metrics,
    map_shape: tuple[int, int],
) -> tuple[pygame.Rect | None, pygame.Rect | None]:
    """Divide the sidebar between the semantic map and the legend.

    The map gets exactly the height its square tiles need at the sidebar's
    width; the legend takes whatever is left. That keeps the tiles square
    without stretching the card.
    """
    gap = metrics.gap
    if not visibility.semantic_map:
        return None, sidebar if visibility.legend else None

    rows, cols = map_shape
    inner_w = sidebar.width - 2 * metrics.pad
    tile = max(4, inner_w // max(1, cols))
    wanted = tile * rows + metrics.header_height + 2 * metrics.pad

    if not visibility.legend:
        # Only the map is showing - hug the grid rather than stretching the card
        return pygame.Rect(sidebar.x, sidebar.y, sidebar.width, min(sidebar.height, wanted)), None

    # The legend wraps into columns, so it only needs a few rows' worth of
    # height. Give the map everything above that.
    legend_min = metrics.header_height + 2 * metrics.pad + 6 * int(16 * metrics.ui_scale)
    map_height = int(min(wanted, sidebar.height - legend_min - gap))
    if map_height < metrics.header_height + 4 * tile:
        return sidebar, None  # squeezing both would leave neither readable

    semantic = pygame.Rect(sidebar.x, sidebar.y, sidebar.width, map_height)
    legend = pygame.Rect(
        sidebar.x,
        semantic.bottom + gap,
        sidebar.width,
        sidebar.height - map_height - gap,
    )
    return semantic, legend


def _hug_game_card(area: pygame.Rect, metrics: Metrics) -> pygame.Rect:
    """Shrink the game card to the width its frame actually needs, then centre it.

    Without this, a card much wider than its aspect-limited frame leaves a wide
    band of empty panel on either side of the game.
    """
    usable_height = area.height - 2 * metrics.pad
    if usable_height <= 0:
        return area  # too short to hold a frame at all; nothing to hug

    needed = int(usable_height * GAME_ASPECT) + 2 * metrics.pad
    if needed >= area.width:
        return area
    return pygame.Rect(area.x + (area.width - needed) // 2, area.y, needed, area.height)


def _fit_game_view(card: pygame.Rect, metrics: Metrics) -> pygame.Rect:
    """Centre an aspect-correct game frame inside its card.

    Returns an empty rect when the card cannot hold a frame at the correct
    aspect ratio, rather than a clamped one - a 1px-tall "frame" would both
    look wrong and misreport the aspect.
    """
    inner = card.inflate(-2 * metrics.pad, -2 * metrics.pad)
    if inner.width <= 0 or inner.height <= 0:
        return pygame.Rect(card.x, card.y, 0, 0)

    if inner.width / inner.height > GAME_ASPECT:
        height = inner.height
        width = int(height * GAME_ASPECT)
    else:
        width = inner.width
        height = int(width / GAME_ASPECT)

    if width <= 0 or height <= 0:
        return pygame.Rect(card.x, card.y, 0, 0)

    return pygame.Rect(
        inner.x + (inner.width - width) // 2,
        inner.y + (inner.height - height) // 2,
        width,
        height,
    )
