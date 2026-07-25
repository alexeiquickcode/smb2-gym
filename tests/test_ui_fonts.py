"""Tests for monospace font resolution.

`pygame.font.match_font` fuzzy-matches and returns an unrelated font (commonly
FreeMono) rather than reporting failure, so a fallback list built on it stops
at its first entry and the interface renders in whatever it happened to hit.
These tests pin the fontconfig-based resolution that replaced it.
"""

import os

import pytest


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

pygame = pytest.importorskip("pygame")

from smb2_gym.app.theme import (  # noqa: E402
    MONO_FAMILIES,
    Fonts,
    _fc_match,
    _mono,
    _normalise,
    _resolve_mono,
)


@pytest.fixture(scope="module", autouse=True)
def _pygame_font():
    pygame.init()
    yield
    pygame.quit()


def test_mono_returns_a_usable_font():
    """A font must always come back, even with no system fonts installed."""
    font = _mono(14)
    assert font.size("0")[0] > 0
    assert font.get_height() > 0


def test_font_is_monospaced():
    """Columns of numbers only align if the digits share one advance width.

    Checked over several sizes because hinting can round an individual glyph a
    pixel wide at one specific size without the face being proportional.
    """
    for size in (12, 13, 15, 18, 24):
        font = _mono(size)
        widths = {font.size(ch)[0] for ch in "0123456789"}
        assert len(widths) == 1, f"digits not uniform at size {size}: {widths}"


def test_bold_is_heavier_than_regular():
    """Headings must be visually distinct from body text."""
    regular = _mono(20)
    bold = _mono(20, bold=True)
    assert bold.size("HEADING")[0] >= regular.size("HEADING")[0]


def test_unavailable_family_resolves_to_none():
    """fc-match must not fuzzy-substitute a font we did not ask for."""
    result = _fc_match("Definitely Not An Installed Font 12345", bold=False)
    assert result is None


def test_resolved_font_file_exists():
    """Whatever we resolve to must be a real, loadable file."""
    path = _resolve_mono(False)
    if path is None:
        pytest.skip("no system monospace font available")
    assert os.path.exists(path)
    pygame.font.Font(path, 14)  # must not raise


def test_glyphs_used_by_the_ui_are_present():
    """The interface uses a few non-ASCII characters; they must render."""
    font = _mono(14)
    for char in "…·—":
        assert font.size(char)[0] > 0, f"missing glyph: {char!r}"


def test_normalise_ignores_case_and_separators():
    assert _normalise("JetBrains Mono") == "jetbrainsmono"
    assert _normalise("DejaVu Sans Mono") == _normalise("dejavusansmono")


def test_preference_list_ends_with_generic_monospace():
    """The final entry must be the generic alias so fontconfig can fall back."""
    assert MONO_FAMILIES[-1] == "monospace"


def test_fonts_rebuild_changes_sizes():
    """Rebuilding at a new UI scale must actually resize the fonts."""
    fonts = Fonts(1.0)
    small_height = fonts.body.get_height()
    fonts.rebuild(1.6)
    assert fonts.body.get_height() > small_height
