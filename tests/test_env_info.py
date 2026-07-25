"""Tests for the info dict contract.

Training code reads `info` every step, so its shape matters as much as its
values. Two things are guarded here: the key set never changes between reset
and step, and the reason an episode ended is always recoverable.
"""

import numpy as np
import pytest

from smb2_gym.app import InitConfig
from smb2_gym.smb2_env import SuperMarioBros2Env


EPISODE_KEYS = ["life_lost", "end_reason", "level_completed", "game_over"]


def test_episode_keys_exist_at_reset(env_no_render):
    """`info['life_lost']` used to raise KeyError until a life was actually lost."""
    env = env_no_render
    obs, info = env.reset()
    for key in EPISODE_KEYS:
        assert key in info, f"{key} missing from reset info"


def test_episode_keys_are_falsey_at_reset(env_no_render):
    env = env_no_render
    obs, info = env.reset()
    assert info['life_lost'] is False
    assert info['end_reason'] is None
    assert info['level_completed'] is False
    assert info['game_over'] is False


@pytest.mark.slow
def test_info_schema_is_stable_across_steps(env_no_render):
    """Vectorised wrappers assume a fixed key set; a conditional key breaks them."""
    env = env_no_render
    obs, info = env.reset()
    expected = set(info)

    for step in range(200):
        obs, _, terminated, truncated, info = env.step(np.int64(1))
        assert set(info) == expected, f"info keys changed at step {step}: {set(info) ^ expected}"
        if terminated or truncated:
            break


@pytest.mark.slow
def test_life_lost_is_always_a_bool(env_no_render):
    env = env_no_render
    obs, info = env.reset()
    for _ in range(150):
        obs, _, terminated, truncated, info = env.step(np.int64(1))
        assert isinstance(info['life_lost'], bool)
        if terminated or truncated:
            break


def test_truncation_reports_max_steps():
    """A step-limited episode must say why it ended."""
    env = SuperMarioBros2Env(
        init_config=InitConfig(level="1-1", character="luigi"),
        max_episode_steps=20,
    )
    try:
        obs, info = env.reset()
        for _ in range(40):
            obs, _, terminated, truncated, info = env.step(np.int64(0))
            if terminated or truncated:
                break
        assert truncated
        assert info['end_reason'] == 'max_steps'
    finally:
        env.close()


def test_end_reason_is_none_while_the_episode_runs(env_no_render):
    """A live episode has no end reason, so a reward fn can test it directly."""
    env = env_no_render
    obs, info = env.reset()
    obs, _, terminated, truncated, info = env.step(np.int64(1))
    if not (terminated or truncated):
        assert info['end_reason'] is None


@pytest.mark.slow
def test_terminated_always_carries_a_reason(env_no_render):
    """`terminated` conflates winning and dying, so the reason must be set."""
    env = env_no_render
    obs, info = env.reset()
    for _ in range(600):
        obs, _, terminated, truncated, info = env.step(np.int64(1))
        if terminated or truncated:
            assert info['end_reason'] in {
                'level_completed',
                'game_over',
                'life_lost',
                'max_steps',
            }, f"unset end_reason on a finished episode: {info['end_reason']!r}"
            break
