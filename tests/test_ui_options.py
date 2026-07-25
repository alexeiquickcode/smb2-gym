"""Tests for the options menu.

The menu's job is to change settings and have those settings reach the
renderer. Both halves are covered: the state machine on its own, and a real
render proving each visual option actually changes pixels.
"""

import os

import numpy as np
import pygame
import pytest

from smb2_gym.app.options import (
    DEFAULT_FPS,
    FPS_CHOICES,
    OPTION_HOTKEYS,
    OPTIONS,
    OptionsMenu,
    Settings,
    format_fps,
)


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# ------------------------------------------------------------------------------
# ---- Settings ----------------------------------------------------------------
# ------------------------------------------------------------------------------


def test_fps_cycles_through_every_choice_and_wraps():
    settings = Settings()
    seen = [settings.cycle_fps(1) for _ in range(len(FPS_CHOICES))]
    assert set(seen) == set(FPS_CHOICES)
    assert settings.target_fps == DEFAULT_FPS  # a full cycle returns home


def test_fps_cycles_backwards():
    settings = Settings()
    assert settings.cycle_fps(-1) != DEFAULT_FPS
    assert settings.cycle_fps(1) == DEFAULT_FPS


def test_fps_recovers_from_an_unknown_value():
    """A value outside the list must not raise; it falls back to the default."""
    settings = Settings(target_fps=37)
    assert settings.cycle_fps(1) in FPS_CHOICES


def test_unlimited_fps_is_labelled_not_zero():
    assert format_fps(0) == "Unlimited"
    assert format_fps(60) == "60 FPS"


def test_zero_is_a_choice_so_the_cap_can_be_removed():
    assert 0 in FPS_CHOICES


# ------------------------------------------------------------------------------
# ---- Menu state --------------------------------------------------------------
# ------------------------------------------------------------------------------


def test_selection_wraps_in_both_directions():
    menu = OptionsMenu()
    menu.move(-1)
    assert menu.selected == len(OPTIONS) - 1
    menu.move(1)
    assert menu.selected == 0


def test_activating_a_toggle_flips_it_and_reports_the_new_value():
    settings = Settings()
    index = next(i for i, o in enumerate(OPTIONS) if o.label == "Map grid")
    before = settings.show_grid
    message = OptionsMenu().activate_index(index, settings)
    assert settings.show_grid is not before
    assert "Map grid" in message


def test_every_hotkey_points_at_a_real_option():
    for index in OPTION_HOTKEYS.values():
        assert 0 <= index < len(OPTIONS)


def test_hotkeys_are_unique():
    indices = list(OPTION_HOTKEYS.values())
    assert len(indices) == len(set(indices))


def test_frame_rate_row_is_not_bound_to_a_toggle_hotkey():
    """The frame rate is stepped with -/+, so no letter key may target it."""
    fps_index = next(i for i, o in enumerate(OPTIONS) if o.label == "Frame rate")
    assert fps_index not in OPTION_HOTKEYS.values()


def test_every_option_describes_itself_for_both_states():
    """No option may crash or return an empty label in either state."""
    for option in OPTIONS:
        for value in (
            Settings(),
            Settings(
                **{
                    "integer_scaling": True,
                    "show_collision": False,
                    "show_grid": False,
                    "log_rewards": True,
                }
            ),
        ):
            assert option.describe(value)


# ------------------------------------------------------------------------------
# ---- Settings reach the renderer ---------------------------------------------
# ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ui_env():
    """A rendered UI over a live environment, for pixel comparisons."""
    from smb2_gym.app import InitConfig
    from smb2_gym.app.play_display import PlayUI
    from smb2_gym.smb2_env import SuperMarioBros2Env

    pygame.init()
    env = SuperMarioBros2Env(init_config=InitConfig(level="1-1", character="mario"))
    ui = PlayUI(900, 700)
    obs, info = env.reset()
    for _ in range(60):
        obs, _, terminated, truncated, info = env.step(np.int64(1))
        if terminated or truncated:
            break
    yield ui, env, obs, info
    env.close()
    pygame.quit()


def _snapshot(ui, env, obs, info):
    ui.render(obs, env, info, paused=False, game_over=False, fps=60.0, dt=0.016)
    return pygame.surfarray.array3d(ui.screen).copy()


@pytest.mark.parametrize("setting", ["integer_scaling", "show_collision", "show_grid"])
def test_visual_setting_changes_the_frame(ui_env, setting):
    """Each visual option must actually alter what is drawn, and be reversible.

    Without this a setting can toggle happily in the menu while never being
    threaded through to the panel that should honour it.
    """
    ui, env, obs, info = ui_env
    ui.show_options = False

    before = _snapshot(ui, env, obs, info)
    setattr(ui.settings, setting, not getattr(ui.settings, setting))
    after = _snapshot(ui, env, obs, info)
    assert (before != after).any(), f"{setting} had no visible effect"

    setattr(ui.settings, setting, not getattr(ui.settings, setting))
    assert (_snapshot(ui, env, obs, info) == before).all(), f"{setting} did not restore"


def test_opening_options_closes_the_help_overlay(ui_env):
    """Two full-window overlays must not stack."""
    ui, _, _, _ = ui_env
    ui.show_options = False
    ui.visibility.help = True
    ui.toggle_options()
    assert ui.show_options and not ui.visibility.help
    ui.toggle_options()


def test_keys_are_only_consumed_while_the_menu_is_open(ui_env):
    """Closed, the menu must let `S`/`C`/`G` fall through to the panel toggles."""
    ui, _, _, _ = ui_env
    ui.show_options = False
    assert ui.handle_options_key(pygame.K_s) is False

    ui.show_options = True
    assert ui.handle_options_key(pygame.K_s) is True
    ui.show_options = False


def test_escape_closes_the_menu(ui_env):
    ui, _, _, _ = ui_env
    ui.show_options = True
    assert ui.handle_options_key(pygame.K_ESCAPE) is True
    assert not ui.show_options


def test_unrelated_keys_are_not_swallowed(ui_env):
    """A key the menu has no use for must stay available to the game."""
    ui, _, _, _ = ui_env
    ui.show_options = True
    assert ui.handle_options_key(pygame.K_F5) is False
    ui.show_options = False


def test_menu_renders_at_a_range_of_window_sizes(ui_env):
    """The overlay must survive small windows without raising."""
    ui, env, obs, info = ui_env
    ui.show_options = True
    try:
        for size in [(560, 440), (900, 700), (1600, 1000)]:
            ui.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
            ui._sync_scale()
            ui.render(obs, env, info, paused=False, game_over=False, fps=60.0, dt=0.016)
    finally:
        ui.show_options = False


# ------------------------------------------------------------------------------
# ---- Character tab -----------------------------------------------------------
# ------------------------------------------------------------------------------


def test_every_tab_renders(ui_env):
    """Each tab must draw without raising, including the static CHARACTER one."""
    from smb2_gym.app.stats_panel import TABS

    ui, env, obs, info = ui_env
    ui.show_options = False
    for index in range(len(TABS)):
        ui.active_tab = index
        ui.render(obs, env, info, paused=False, game_over=False, fps=60.0, dt=0.016)
    ui.active_tab = 0


# ------------------------------------------------------------------------------
# ---- Overlays hold the game still --------------------------------------------
# ------------------------------------------------------------------------------


def test_no_overlay_lets_the_game_run(ui_env):
    ui, _, _, _ = ui_env
    ui.show_options = False
    ui.visibility.help = False
    assert ui.blocks_input is False


def test_options_menu_blocks_game_input(ui_env):
    """The game must not step behind the menu.

    Its rows are bound to letter keys that are also controller input, so a
    running game would read `S`/`C`/`G` as button presses.
    """
    ui, _, _, _ = ui_env
    ui.visibility.help = False
    ui.show_options = True
    assert ui.blocks_input is True
    ui.show_options = False


def test_help_overlay_blocks_game_input(ui_env):
    ui, _, _, _ = ui_env
    ui.show_options = False
    ui.visibility.help = True
    assert ui.blocks_input is True
    ui.visibility.help = False
