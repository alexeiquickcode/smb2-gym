"""Tests for the resizable play UI layout.

These run headless via SDL's dummy video driver so they work in CI.
"""

import os

import pytest


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

pygame = pytest.importorskip("pygame")

from smb2_gym.app.layout import (  # noqa: E402
    NARROW_WIDTH,
    PanelVisibility,
    compute_layout,
    compute_ui_scale,
)
from smb2_gym.app.theme import Metrics  # noqa: E402


WINDOW_SIZES = [
    (1, 1),
    (60, 40),
    (400, 90),
    (90, 400),
    (560, 420),
    (900, 700),
    (1280, 860),
    (1920, 1080),
    (3840, 2160),
]


@pytest.fixture(scope="module", autouse=True)
def _pygame_display():
    """Initialise a headless pygame display for the module."""
    pygame.init()
    pygame.display.set_mode((640, 480))
    yield
    pygame.quit()


def _metrics_for(width: int, height: int) -> Metrics:
    return Metrics(ui_scale=compute_ui_scale(width, height))


@pytest.mark.parametrize("width,height", WINDOW_SIZES)
def test_layout_rects_are_never_negative(width, height):
    """Every computed rect must have non-negative dimensions at any size."""
    layout = compute_layout(width, height, PanelVisibility(), _metrics_for(width, height))

    rects = [layout.status, layout.game_card, layout.game_view]
    rects += [
        r
        for r in (layout.sidebar, layout.semantic_card, layout.legend_card, layout.stats_card)
        if r is not None
    ]

    for rect in rects:
        assert rect.width >= 0, f"negative width at {width}x{height}: {rect}"
        assert rect.height >= 0, f"negative height at {width}x{height}: {rect}"


@pytest.mark.parametrize("width,height", WINDOW_SIZES)
def test_game_view_keeps_nes_aspect_ratio(width, height):
    """The game view must stay 256:240 so the picture is never stretched."""
    layout = compute_layout(width, height, PanelVisibility(), _metrics_for(width, height))
    view = layout.game_view

    # Below ~50px the frame is a handful of pixels and integer rounding
    # dominates the ratio, so the check is only meaningful above that.
    if view.width < 50 or view.height < 50:
        pytest.skip("window too small for a meaningful game view")

    aspect = view.width / view.height
    assert aspect == pytest.approx(256 / 240, rel=0.02)


@pytest.mark.parametrize("width,height", WINDOW_SIZES)
def test_panels_stay_inside_the_window(width, height):
    """No panel may extend past the window bounds."""
    layout = compute_layout(width, height, PanelVisibility(), _metrics_for(width, height))
    window = pygame.Rect(0, 0, width, height)

    for rect in (layout.sidebar, layout.semantic_card, layout.legend_card, layout.stats_card):
        if rect is None or rect.width == 0 or rect.height == 0:
            continue
        assert window.contains(rect), f"{rect} escapes {window} at {width}x{height}"


def test_sidebar_dropped_on_narrow_windows():
    """Below the narrow threshold the sidebar yields its space to the game."""
    width, height = NARROW_WIDTH - 40, 700
    layout = compute_layout(width, height, PanelVisibility(), _metrics_for(width, height))
    assert layout.sidebar is None
    assert layout.semantic_card is None


def test_hidden_panels_are_not_laid_out():
    """Toggling panels off removes their rects entirely."""
    visibility = PanelVisibility(semantic_map=False, legend=False, stats=False)
    layout = compute_layout(1280, 860, visibility, _metrics_for(1280, 860))

    assert layout.semantic_card is None
    assert layout.legend_card is None
    assert layout.stats_card is None


def test_semantic_card_hugs_grid_when_legend_hidden():
    """With the legend hidden the map card shrinks instead of stretching."""
    visibility = PanelVisibility(legend=False)
    layout = compute_layout(1280, 860, visibility, _metrics_for(1280, 860))

    assert layout.semantic_card is not None
    assert layout.sidebar is not None
    assert layout.semantic_card.height < layout.sidebar.height
