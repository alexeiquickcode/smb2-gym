"""Options menu for the human-play interface.

Settings that are useful to change mid-session without restarting: the frame
rate cap, whether the game view snaps to whole pixels, and which overlays are
drawn. Each option is a small record with its own getter/setter so the menu
stays declarative -- adding one means adding a row, not a new branch.
"""

from collections.abc import Callable
from dataclasses import (
    dataclass,
    field,
)

import pygame

from .theme import (
    ACCENT,
    BORDER_STRONG,
    GOOD,
    PANEL,
    PANEL_ALT,
    TEXT,
    TEXT_DIM,
    TEXT_FAINT,
    Fonts,
    Metrics,
    blit_clipped,
    draw_card,
    draw_card_header,
)


# The speeds worth cycling between: half speed for inspecting a tricky jump,
# 60 for normal play, and higher caps for getting through a level quickly.
# 0 means "no cap" - run as fast as the emulator and display allow.
FPS_CHOICES = [15, 30, 60, 90, 120, 240, 0]
DEFAULT_FPS = 60


def format_fps(value: int) -> str:
    return "Unlimited" if value == 0 else f"{value} FPS"


@dataclass
class Settings:
    """User-adjustable settings for a play session."""

    target_fps: int = DEFAULT_FPS

    # Snap the game view to a whole-number scale. Nearest-neighbour upscaling by
    # a fractional factor makes some NES pixels wider than others, which shows
    # up as uneven text; the trade-off is a smaller picture.
    integer_scaling: bool = False

    # Draw the player's collision tiles over the semantic map.
    show_collision: bool = True

    # Grid lines between semantic map cells.
    show_grid: bool = True

    # Print each frame's reward to stdout; useful when shaping a reward fn.
    log_rewards: bool = False

    def cycle_fps(self, step: int = 1) -> int:
        """Move to the next/previous FPS cap and return the new value."""
        try:
            index = FPS_CHOICES.index(self.target_fps)
        except ValueError:
            index = FPS_CHOICES.index(DEFAULT_FPS)
        self.target_fps = FPS_CHOICES[(index + step) % len(FPS_CHOICES)]
        return self.target_fps


@dataclass
class Option:
    """One row in the options menu."""

    key: str  # the hotkey shown alongside the row
    label: str
    describe: Callable[[Settings], str]
    activate: Callable[[Settings], None]
    hint: str = ""
    is_on: Callable[[Settings], bool] | None = None


def _toggle(name: str) -> Callable[[Settings], None]:
    def apply(settings: Settings) -> None:
        setattr(settings, name, not getattr(settings, name))

    return apply


def _on_off(name: str) -> Callable[[Settings], str]:
    def describe(settings: Settings) -> str:
        return "On" if getattr(settings, name) else "Off"

    return describe


OPTIONS: list[Option] = [
    Option(
        key="[-] [+]",
        label="Frame rate",
        describe=lambda s: format_fps(s.target_fps),
        activate=lambda s: s.cycle_fps(1),
        hint="Speed up or slow the game down",
    ),
    Option(
        key="[S]",
        label="Integer scaling",
        describe=_on_off("integer_scaling"),
        activate=_toggle("integer_scaling"),
        hint="Even pixels, slightly smaller picture",
        is_on=lambda s: s.integer_scaling,
    ),
    Option(
        key="[C]",
        label="Collision overlay",
        describe=_on_off("show_collision"),
        activate=_toggle("show_collision"),
        hint="Player's collision tiles on the map",
        is_on=lambda s: s.show_collision,
    ),
    Option(
        key="[G]",
        label="Map grid",
        describe=_on_off("show_grid"),
        activate=_toggle("show_grid"),
        hint="Grid lines on the semantic map",
        is_on=lambda s: s.show_grid,
    ),
    Option(
        key="[O]",
        label="Log rewards",
        describe=_on_off("log_rewards"),
        activate=_toggle("log_rewards"),
        hint="Print step rewards to the terminal",
        is_on=lambda s: s.log_rewards,
    ),
]


# Keys that activate an option directly, so the menu is usable without arrows.
OPTION_HOTKEYS: dict[int, int] = {
    pygame.K_s: 1,
    pygame.K_c: 2,
    pygame.K_g: 3,
    pygame.K_o: 4,
}


@dataclass
class OptionsMenu:
    """Selection state for the options overlay."""

    selected: int = 0
    _rects: list[pygame.Rect] = field(default_factory=list)

    def move(self, step: int) -> None:
        self.selected = (self.selected + step) % len(OPTIONS)

    def activate(self, settings: Settings, step: int = 1) -> str:
        """Apply the selected option and return a message describing the result."""
        option = OPTIONS[self.selected]
        if option.label == "Frame rate":
            settings.cycle_fps(step)
        else:
            option.activate(settings)
        return f"{option.label}: {option.describe(settings)}"

    def activate_index(self, index: int, settings: Settings) -> str:
        self.selected = index
        return self.activate(settings)

    def handle_click(self, pos: tuple[int, int], settings: Settings) -> str | None:
        """Activate whichever row was clicked, if any."""
        for index, rect in enumerate(self._rects):
            if rect.collidepoint(pos):
                return self.activate_index(index, settings)
        return None


def draw_options_overlay(
    surface: pygame.Surface,
    window: pygame.Rect,
    fonts: Fonts,
    metrics: Metrics,
    settings: Settings,
    menu: OptionsMenu,
) -> None:
    """Draw the options menu centred over the window."""
    veil = pygame.Surface(window.size, pygame.SRCALPHA)
    veil.fill((8, 9, 13, 200))
    surface.blit(veil, (0, 0))

    # Each row stacks a label over a hint, so it needs both lines plus breathing
    # room - metrics.row_height alone is sized for single-line table rows.
    stacked = fonts.small.get_height() + fonts.tiny.get_height() + max(6, int(8 * metrics.ui_scale))
    row_h = max(metrics.row_height, stacked)
    card_h = min(window.height - 4 * metrics.gap, len(OPTIONS) * row_h + 6 * metrics.pad)
    card_w = min(window.width - 4 * metrics.gap, int(560 * metrics.ui_scale))
    card = pygame.Rect(0, 0, card_w, card_h)
    card.center = window.center

    draw_card(surface, card, metrics, fill=PANEL, border=BORDER_STRONG)
    content = draw_card_header(
        surface, card, "OPTIONS", fonts, metrics, accent=ACCENT, hint="[F2] to close"
    )

    menu._rects = []
    key_w = int(content.width * 0.20)
    value_w = int(content.width * 0.26)

    y = content.y
    for index, option in enumerate(OPTIONS):
        if y + row_h > content.bottom:
            break

        row = pygame.Rect(content.x - metrics.pad // 2, y - 2, content.width + metrics.pad, row_h)
        menu._rects.append(row)

        if index == menu.selected:
            pygame.draw.rect(surface, PANEL_ALT, row, border_radius=4)
            pygame.draw.rect(surface, ACCENT, row, width=1, border_radius=4)

        blit_clipped(
            surface,
            fonts.small.render(option.key, True, TEXT_DIM),
            (content.x, y + 2),
            key_w,
        )
        blit_clipped(
            surface,
            fonts.small.render(option.label, True, TEXT),
            (content.x + key_w, y + 2),
            max(0, content.width - key_w - value_w - metrics.pad),
        )

        # An enabled toggle reads green; the frame rate is just a value.
        on = option.is_on(settings) if option.is_on else None
        value_color = TEXT if on is None else (GOOD if on else TEXT_FAINT)
        blit_clipped(
            surface,
            fonts.small.render(option.describe(settings), True, value_color),
            (content.right - value_w, y + 2),
            value_w,
        )

        # The hint sits under the label, so it must stop short of the value
        # column too - otherwise a long hint runs underneath "On"/"Off".
        if option.hint and row_h >= int(30 * metrics.ui_scale):
            blit_clipped(
                surface,
                fonts.tiny.render(option.hint, True, TEXT_FAINT),
                (content.x + key_w, y + 2 + fonts.small.get_height()),
                max(0, content.width - key_w - value_w - metrics.pad),
            )

        y += row_h

    # Spelled out rather than drawn with arrow glyphs: the font chain falls back
    # across many families and arrows are not guaranteed in all of them.
    footer = "[Up]/[Down] select  ·  [Left]/[Right] or [Enter] change  ·  click a row"
    blit_clipped(
        surface,
        fonts.tiny.render(footer, True, TEXT_FAINT),
        (content.x, card.bottom - metrics.pad - fonts.tiny.get_height()),
        content.width,
    )
