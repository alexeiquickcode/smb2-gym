"""Regression tests for window resize handling.

The bug these guard against: calling `pygame.display.set_mode` while handling a
resize event makes SDL emit another resize event, which re-enters the handler.
Tiling compositors (Hyprland, sway, i3) trigger it instantly because they force
the window size and re-assert it after every mode change, so the application
locks up in an endless resize loop.
"""

import os

import pytest


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

pygame = pytest.importorskip("pygame")

from smb2_gym.app.play_display import PlayUI  # noqa: E402


@pytest.fixture
def ui():
    """A headless PlayUI instance."""
    pygame.init()
    instance = PlayUI(1000, 800)
    yield instance
    pygame.quit()


def test_handle_resize_never_calls_set_mode(ui, monkeypatch):
    """Resize handling must not re-enter set_mode - that is the feedback loop."""
    calls = []
    real_set_mode = pygame.display.set_mode

    def spy(*args, **kwargs):
        calls.append(args)
        return real_set_mode(*args, **kwargs)

    monkeypatch.setattr(pygame.display, "set_mode", spy)

    # A compositor spamming resizes, including sizes far below the requested
    # startup minimum - previously these were forcibly overridden.
    for size in [(300, 200), (301, 200), (300, 200), (120, 90), (2400, 100), (1, 1)]:
        ui.handle_resize(*size)

    assert calls == [], f"handle_resize called set_mode {len(calls)} times"


def test_handle_resize_tracks_the_live_surface(ui):
    """The UI follows the actual surface size, not the event's numbers."""
    ui.screen = pygame.display.set_mode((820, 640), pygame.RESIZABLE)

    # Deliberately pass numbers that disagree with the real surface; the surface
    # must win, since under a tiler the event can lag or lie.
    ui.handle_resize(99999, 99999)

    assert ui.screen.get_size() == (820, 640)


def test_resize_below_startup_minimum_is_not_overridden(ui):
    """A compositor may size the window below MIN_WINDOW; we must accept it."""
    ui.screen = pygame.display.set_mode((240, 180), pygame.RESIZABLE)
    ui.handle_resize(240, 180)

    assert ui.screen.get_size() == (240, 180)
