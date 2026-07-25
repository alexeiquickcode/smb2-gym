"""Visual theme for the human-play interface.

A single place for colours, spacing and font sizing so the panels stay
consistent. Everything that scales with the window goes through `Metrics`,
which is recomputed on resize.
"""

import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache

import pygame


# ------------------------------------------------------------------------------
# ---- Palette -----------------------------------------------------------------
# ------------------------------------------------------------------------------

BG = (16, 18, 24)  # window background
PANEL = (26, 29, 38)  # card surface
PANEL_ALT = (32, 36, 47)  # zebra striping / nested surface
BORDER = (48, 54, 70)  # card outline
BORDER_STRONG = (72, 80, 102)  # emphasised outline

TEXT = (228, 232, 241)  # primary text
TEXT_DIM = (138, 147, 168)  # labels, secondary text
TEXT_FAINT = (92, 99, 118)  # disabled / empty cells

ACCENT = (94, 168, 255)  # headings, active tab
ACCENT_DIM = (58, 104, 158)
GOOD = (86, 204, 137)
WARN = (247, 191, 88)
BAD = (240, 104, 104)

SHADOW = (10, 11, 15)

# Section header accents, keyed by the section title used in the stats panel
SECTION_ACCENTS = {
    "POSITION": (94, 168, 255),
    "PLAYER": (146, 205, 120),
    "ENEMIES": (240, 140, 104),
    "TIMERS": (198, 148, 245),
    "PROGRESS": (247, 191, 88),
    "JUMP": (120, 200, 235),
    "PHYSICS": (198, 148, 245),
    "PICK UP": (247, 191, 88),
}


# ------------------------------------------------------------------------------
# ---- Metrics -----------------------------------------------------------------
# ------------------------------------------------------------------------------


@dataclass
class Metrics:
    """Pixel measurements derived from the current window size.

    A single `ui_scale` factor drives padding and font sizes so the interface
    stays legible when the window is small and doesn't look sparse when it is
    large.
    """

    ui_scale: float = 1.0

    @property
    def gap(self) -> int:
        """Space between cards."""
        return max(6, int(10 * self.ui_scale))

    @property
    def pad(self) -> int:
        """Inner padding of a card."""
        return max(6, int(12 * self.ui_scale))

    @property
    def radius(self) -> int:
        return max(4, int(8 * self.ui_scale))

    @property
    def row_height(self) -> int:
        return max(14, int(20 * self.ui_scale))

    @property
    def header_height(self) -> int:
        return max(18, int(26 * self.ui_scale))

    @property
    def status_height(self) -> int:
        return max(22, int(30 * self.ui_scale))

    @property
    def tab_height(self) -> int:
        return max(20, int(28 * self.ui_scale))


class Fonts:
    """Font set for the interface, rebuilt whenever the UI scale changes."""

    def __init__(self, ui_scale: float = 1.0) -> None:
        self.rebuild(ui_scale)

    def rebuild(self, ui_scale: float) -> None:
        """Recreate the fonts for a new UI scale."""
        self.ui_scale = ui_scale
        self.title = _mono(int(18 * ui_scale), bold=True)
        self.heading = _mono(int(15 * ui_scale), bold=True)
        self.body = _mono(int(15 * ui_scale))
        self.small = _mono(int(13 * ui_scale))
        self.tiny = _mono(int(12 * ui_scale))


# Monospace families in order of preference. Anything here renders well at the
# small sizes this interface uses; the generic "monospace" at the end lets
# fontconfig pick the system default rather than dropping to pygame's built-in.
MONO_FAMILIES = [
    "JetBrains Mono",
    "Fira Code",
    "Cascadia Code",
    "CaskaydiaCove Nerd Font Mono",
    "Source Code Pro",
    "IBM Plex Mono",
    "Inconsolata",
    "Hack",
    "Ubuntu Mono",
    "Menlo",
    "Consolas",
    "Adwaita Mono",
    "DejaVu Sans Mono",
    "Liberation Mono",
    "Noto Sans Mono",
    "monospace",
]

# Families that are technically monospace but look poor on screen at UI sizes.
# fontconfig happily returns these as a "match" for anything, so they are only
# acceptable when nothing better exists.
_POOR_MONO = ("freemono", "nimbusmono", "courier", "courier10", "fixed")


def _normalise(name: str) -> str:
    """Lowercase and strip separators so family names compare loosely."""
    return "".join(c for c in name.lower() if c.isalnum())


def _fc_match(family: str, bold: bool) -> str | None:
    """Resolve a font family to a file via fontconfig.

    `pygame.font.match_font` is unusable for this: it fuzzy-matches and returns
    an unrelated font (often FreeMono) instead of reporting failure, so a
    fallback list built on it silently stops at the first entry. fc-match
    reports the family it actually resolved, which lets us verify the result.
    """
    weight = "bold" if bold else "regular"
    try:
        result = subprocess.run(
            ["fc-match", f"{family}:weight={weight}", "--format=%{family}\t%{file}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None  # no fontconfig (Windows, minimal container)

    if result.returncode != 0 or "\t" not in result.stdout:
        return None

    resolved, _, path = result.stdout.partition("\t")
    path = path.strip()
    if not path or not os.path.exists(path):
        return None

    # fontconfig always returns *something*; only accept it when it really is
    # the family we asked for. "monospace" is a generic alias, so let it pass.
    if family != "monospace":
        wanted = _normalise(family)
        got = {_normalise(part) for part in resolved.split(",")}
        if not any(wanted in name or name in wanted for name in got):
            return None

    return path


@lru_cache(maxsize=8)
def _resolve_mono(bold: bool) -> str | None:
    """Find the best available monospace font file, or None for the default.

    Cached because it shells out to fc-match, and fonts are rebuilt on every
    window resize.
    """
    fallback: str | None = None

    for family in MONO_FAMILIES:
        path = _fc_match(family, bold)
        if not path:
            continue
        if any(poor in _normalise(os.path.basename(path)) for poor in _POOR_MONO):
            fallback = fallback or path  # remember, but keep looking for better
            continue
        return path

    # Last resort before pygame's built-in: pygame's own matcher. It is
    # unreliable about *which* font it returns, but a real mono font beats the
    # default sans for column alignment.
    if fallback is None:
        try:
            fallback = pygame.font.match_font("monospace", bold=bold)
        except Exception:  # pragma: no cover - platform dependent
            fallback = None

    return fallback


def _mono(size: int, bold: bool = False) -> pygame.font.Font:
    """Load a monospace font at `size`, falling back to the pygame default.

    Monospace keeps the numeric readouts from jittering as values change width,
    which matters a lot when watching stats update at 60fps.
    """
    size = max(9, size)

    path = _resolve_mono(bold)
    if path:
        try:
            font = pygame.font.Font(path, size)
            # A file matched for "bold" may itself be the regular face, so ask
            # for synthetic bold too; pygame ignores this if already bold.
            if bold:
                font.set_bold(True)
            return font
        except (OSError, pygame.error):  # pragma: no cover - corrupt font file
            pass

    # pygame's built-in font is sized differently to a real TTF at the same
    # nominal point size, hence the correction factor.
    font = pygame.font.Font(None, int(size * 1.25))
    font.set_bold(bold)
    return font


# ------------------------------------------------------------------------------
# ---- Drawing helpers ---------------------------------------------------------
# ------------------------------------------------------------------------------


def draw_card(
    surface: pygame.Surface,
    rect: pygame.Rect,
    metrics: Metrics,
    fill: tuple[int, int, int] = PANEL,
    border: tuple[int, int, int] = BORDER,
) -> None:
    """Draw a rounded panel with a subtle drop shadow."""
    if rect.width <= 0 or rect.height <= 0:
        return
    shadow_rect = rect.move(0, max(1, int(2 * metrics.ui_scale)))
    pygame.draw.rect(surface, SHADOW, shadow_rect, border_radius=metrics.radius)
    pygame.draw.rect(surface, fill, rect, border_radius=metrics.radius)
    pygame.draw.rect(surface, border, rect, width=1, border_radius=metrics.radius)


def draw_card_header(
    surface: pygame.Surface,
    rect: pygame.Rect,
    title: str,
    fonts: Fonts,
    metrics: Metrics,
    accent: tuple[int, int, int] = ACCENT,
    hint: str = "",
) -> pygame.Rect:
    """Draw a card title bar and return the remaining content rect."""
    if rect.height <= metrics.header_height:
        return pygame.Rect(rect.x, rect.bottom, rect.width, 0)

    text_y = rect.y + metrics.pad // 2
    bar_x = rect.x + metrics.pad
    bar_h = fonts.heading.get_height()

    # Accent tick to the left of the title
    pygame.draw.rect(
        surface,
        accent,
        pygame.Rect(bar_x, text_y + 2, max(2, int(3 * metrics.ui_scale)), bar_h - 4),
        border_radius=2,
    )

    label_x = bar_x + max(8, int(9 * metrics.ui_scale))
    surface.blit(fonts.heading.render(title, True, TEXT), (label_x, text_y))

    if hint:
        hint_surface = fonts.tiny.render(hint, True, TEXT_FAINT)
        hint_x = rect.right - metrics.pad - hint_surface.get_width()
        if hint_x > label_x + fonts.heading.size(title)[0] + metrics.pad:
            surface.blit(hint_surface, (hint_x, text_y + (bar_h - hint_surface.get_height()) // 2))

    divider_y = text_y + bar_h + metrics.pad // 2
    pygame.draw.line(
        surface, BORDER, (rect.x + metrics.pad, divider_y), (rect.right - metrics.pad, divider_y), 1
    )

    content_top = divider_y + metrics.pad // 2
    return pygame.Rect(
        rect.x + metrics.pad,
        content_top,
        max(0, rect.width - 2 * metrics.pad),
        max(0, rect.bottom - metrics.pad - content_top),
    )


def blit_clipped(
    surface: pygame.Surface,
    text: pygame.Surface,
    pos: tuple[int, int],
    max_width: int,
) -> None:
    """Blit text, cropped to `max_width` so it never bleeds out of its column."""
    if max_width <= 0:
        return
    if text.get_width() <= max_width:
        surface.blit(text, pos)
    else:
        surface.blit(text, pos, pygame.Rect(0, 0, max_width, text.get_height()))


def truncate(font: pygame.font.Font, text: str, max_width: int) -> str:
    """Shorten `text` with an ellipsis so it fits inside `max_width` pixels."""
    if max_width <= 0:
        return ""
    if font.size(text)[0] <= max_width:
        return text
    ellipsis = "…"
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if font.size(text[:mid] + ellipsis)[0] <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + ellipsis if lo else ""
