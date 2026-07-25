"""Display orchestration for the human play interface.

`PlayUI` owns the window and all view state (which panels are open, the active
stats tab, the toast queue) and redraws the whole interface each frame from the
current window size, so resizing needs no special handling beyond recomputing
the layout.
"""

from typing import (
    Any,
)

import numpy as np
import pygame

from smb2_gym.app.layout import (
    MIN_WINDOW,
    Layout,
    PanelVisibility,
    compute_layout,
    compute_ui_scale,
)
from smb2_gym.app.options import (
    OPTION_HOTKEYS,
    OptionsMenu,
    Settings,
    draw_options_overlay,
    format_fps,
)
from smb2_gym.app.panels import (
    draw_game_panel,
    draw_help_overlay,
    draw_legend_panel,
    draw_semantic_panel,
    draw_status_bar,
    draw_toast,
)
from smb2_gym.app.stats_panel import (
    TABS,
    draw_stats_panel,
)
from smb2_gym.app.theme import (
    BG,
    Fonts,
    Metrics,
)
from smb2_gym.constants import CHARACTER_NAMES
from smb2_gym.smb2_env import SuperMarioBros2Env


TOAST_SECONDS = 2.0


class PlayUI:
    """Stateful renderer for the human-play window."""

    def __init__(self, width: int, height: int, caption: str = "Super Mario Bros 2") -> None:
        self.screen = pygame.display.set_mode(
            (max(width, MIN_WINDOW[0]), max(height, MIN_WINDOW[1])), pygame.RESIZABLE
        )
        pygame.display.set_caption(caption)

        self.caption = caption
        self.visibility = PanelVisibility()
        self.active_tab = 0
        self.fullscreen = False
        self._windowed_size = self.screen.get_size()

        self.settings = Settings()
        self.options = OptionsMenu()
        self.show_options = False

        self._toast_message = ""
        self._toast_remaining = 0.0

        self._ui_scale = 0.0
        self.metrics = Metrics()
        self.fonts: Fonts | None = None
        self._tab_rects: list[pygame.Rect] = []
        self._sync_scale(force=True)

    # ---- Window state ------------------------------------------------

    def handle_resize(self, width: int, height: int) -> None:
        """Adopt a new window size reported by the windowing system.

        This must NOT call `set_mode`. Under a RESIZABLE window SDL has already
        resized the surface by the time the event arrives, and calling
        `set_mode` here makes SDL emit a fresh VIDEORESIZE, which lands back in
        this handler - an endless resize loop. Tiling compositors (Hyprland,
        sway, i3) hit it immediately because they force the window size and
        re-assert it against every mode change we make.
        """
        del width, height  # the surface is the source of truth, not the event
        if not self.fullscreen:
            self._windowed_size = self.screen.get_size()
        self._sync_scale()

    def toggle_fullscreen(self) -> None:
        """Switch between fullscreen and the last windowed size.

        Uses borderless desktop fullscreen rather than a real video-mode
        change: it is instant, it never alters the user's screen resolution,
        and it is what tiling compositors expect. Under a tiler the request may
        simply be ignored, which is fine - the layout follows whatever size we
        end up with.
        """
        self.fullscreen = not self.fullscreen
        try:
            if self.fullscreen:
                self._windowed_size = self.screen.get_size()
                self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.SCALED)
            else:
                self.screen = pygame.display.set_mode(self._windowed_size, pygame.RESIZABLE)
        except pygame.error:
            # Some compositors refuse the mode change; stay where we are
            self.fullscreen = not self.fullscreen
            self.toast("Fullscreen unavailable")
            return
        self._sync_scale()
        self.toast("Fullscreen on" if self.fullscreen else "Fullscreen off")

    def _sync_scale(self, force: bool = False) -> None:
        """Rebuild fonts and metrics when the window size changes the UI scale."""
        width, height = self.screen.get_size()
        scale = compute_ui_scale(width, height)
        if force or abs(scale - self._ui_scale) > 0.01 or self.fonts is None:
            self._ui_scale = scale
            self.metrics = Metrics(ui_scale=scale)
            if self.fonts is None:
                self.fonts = Fonts(scale)
            else:
                self.fonts.rebuild(scale)

    # ---- View state --------------------------------------------------

    def toggle_panel(self, name: str) -> None:
        """Toggle a named panel and confirm it with a toast."""
        current = getattr(self.visibility, name)
        setattr(self.visibility, name, not current)
        labels = {
            "semantic_map": "Semantic map",
            "legend": "Legend",
            "stats": "Stats panel",
            "help": "Help",
        }
        if name != "help":
            self.toast(f"{labels.get(name, name)} {'shown' if not current else 'hidden'}")

    def next_tab(self, step: int = 1) -> None:
        """Move to the next (or previous) stats tab."""
        self.active_tab = (self.active_tab + step) % len(TABS)
        if not self.visibility.stats:
            self.visibility.stats = True

    @property
    def blocks_input(self) -> bool:
        """Whether an overlay is capturing input, so the game must hold still.

        Both overlays cover the window and share letter keys with the game, so
        stepping the emulator behind them would feed the menu's keys to Mario.
        """
        return self.show_options or self.visibility.help

    def toggle_options(self) -> None:
        """Open or close the options menu."""
        self.show_options = not self.show_options
        if self.show_options:
            # The two full-window overlays would otherwise stack on each other.
            self.visibility.help = False

    def handle_options_key(self, key: int) -> bool:
        """Handle a key while the options menu is open.

        Returns:
            True if the key was consumed, so the caller does not also treat it
            as a global hotkey - otherwise `S`, `C` and `G` would fight with the
            panel toggles underneath.
        """
        if not self.show_options:
            return False

        # Arrow keys navigate; letters activate their option directly. `S` is
        # the integer-scaling hotkey, so it must not double as "move down".
        if key == pygame.K_UP:
            self.options.move(-1)
        elif key == pygame.K_DOWN:
            self.options.move(1)
        elif key in (pygame.K_RIGHT, pygame.K_RETURN, pygame.K_SPACE):
            self.toast(self.options.activate(self.settings, 1))
        elif key == pygame.K_LEFT:
            self.toast(self.options.activate(self.settings, -1))
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.settings.cycle_fps(1)
            self.toast(f"Frame rate: {format_fps(self.settings.target_fps)}")
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.settings.cycle_fps(-1)
            self.toast(f"Frame rate: {format_fps(self.settings.target_fps)}")
        elif key in OPTION_HOTKEYS:
            self.toast(self.options.activate_index(OPTION_HOTKEYS[key], self.settings))
        elif key == pygame.K_ESCAPE:
            self.show_options = False
        else:
            return False
        return True

    def handle_click(self, pos: tuple[int, int]) -> None:
        """Route a mouse click to the options menu or the stats tabs."""
        if self.show_options:
            message = self.options.handle_click(pos, self.settings)
            if message:
                self.toast(message)
            return

        for i, rect in enumerate(self._tab_rects):
            if rect.collidepoint(pos):
                self.active_tab = i
                return

    def toast(self, message: str) -> None:
        """Show a transient message at the bottom of the window."""
        self._toast_message = message
        self._toast_remaining = TOAST_SECONDS

    # ---- Frame -------------------------------------------------------

    def render(
        self,
        obs: np.ndarray,
        env: SuperMarioBros2Env,
        info: dict[str, Any],
        *,
        paused: bool,
        game_over: bool,
        fps: float,
        dt: float,
    ) -> None:
        """Draw one full frame of the interface."""
        # Re-derive the scale from the live surface every frame. Resize events
        # differ between SDL versions and window managers (VIDEORESIZE,
        # WINDOWRESIZED, or under some compositors none at all), so the surface
        # itself is the only reliable source of the current size.
        self._sync_scale()
        assert self.fonts is not None
        width, height = self.screen.get_size()
        self.screen.fill(BG)

        semantic_map = env.semantic_map
        layout = compute_layout(
            width,
            height,
            self.visibility,
            self.metrics,
            map_shape=(semantic_map.shape[0], semantic_map.shape[1]),
        )

        self._draw_status(layout, info, paused=paused, game_over=game_over, fps=fps)

        draw_game_panel(
            self.screen,
            layout.game_card,
            layout.game_view,
            obs,
            self.fonts,
            self.metrics,
            paused=paused,
            game_over=game_over,
            integer_scaling=self.settings.integer_scaling,
        )

        if layout.semantic_card is not None:
            draw_semantic_panel(
                self.screen,
                layout.semantic_card,
                self.fonts,
                self.metrics,
                semantic_map,
                env.get_player_collision_tiles(),
                layout.tile_size,
                show_collision=self.settings.show_collision,
                show_grid=self.settings.show_grid,
            )

        if layout.legend_card is not None:
            draw_legend_panel(self.screen, layout.legend_card, self.fonts, self.metrics)

        if layout.stats_card is not None:
            self._tab_rects = draw_stats_panel(
                self.screen,
                layout.stats_card,
                self.fonts,
                self.metrics,
                info,
                self.active_tab,
            )
        else:
            self._tab_rects = []

        self._draw_toast(layout, dt)

        if self.visibility.help:
            draw_help_overlay(self.screen, layout.window, self.fonts, self.metrics)

        if self.show_options:
            draw_options_overlay(
                self.screen,
                layout.window,
                self.fonts,
                self.metrics,
                self.settings,
                self.options,
            )

        pygame.display.flip()

    def _draw_status(
        self,
        layout: Layout,
        info: dict[str, Any],
        *,
        paused: bool,
        game_over: bool,
        fps: float,
    ) -> None:
        """Draw the top status bar from the current game info."""
        assert self.fonts is not None
        pc, game = info['pc'], info['game']
        draw_status_bar(
            self.screen,
            layout.status,
            self.fonts,
            self.metrics,
            title=self.caption,
            level=f"World {game.world} · {game.level}",
            character=CHARACTER_NAMES.get(pc.character, "Unknown"),
            lives=pc.lives,
            hearts=pc.hearts,
            coins=pc.coins,
            fps=fps,
            paused=paused,
            game_over=game_over,
        )

    def _draw_toast(self, layout: Layout, dt: float) -> None:
        """Draw and age out the current toast message."""
        assert self.fonts is not None
        if self._toast_remaining <= 0:
            return
        self._toast_remaining = max(0.0, self._toast_remaining - dt)
        # Hold at full opacity, then fade over the last half second
        alpha = min(1.0, self._toast_remaining / 0.5)
        draw_toast(self.screen, layout.window, self.fonts, self.metrics, self._toast_message, alpha)
