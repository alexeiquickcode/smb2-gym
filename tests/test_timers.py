"""Tests for the player timer RAM addresses.

These guard against silent wrong-address bugs: a timer that reads a plausible
byte from an unrelated variable still returns a number, so nothing fails
loudly - the value is simply wrong forever.
"""

from itertools import pairwise

import numpy as np
import pytest

from smb2_gym.constants import ENEMY_SLOTS, TIMERS


def test_carpet_timer_address():
    """Pidgit's carpet timer lives at $00B9.

    It was previously read from $008E, which is a different variable entirely:
    Data Crystal documents $008E as the Bomb 3 fuse, and Xkeeper's disassembly
    labels it FreeSubconsCorkCounter.
    """
    assert TIMERS.PIDGIT_CARPET == 0x00B9


def test_carpet_timer_is_not_an_enemy_slot_variable():
    """$008E sits in the enemy object-timer block; the carpet timer must not.

    This is what made the old address wrong in a way that was easy to miss -
    it aliased a per-slot enemy timer, so it held changing values that looked
    like a working countdown.
    """
    enemy_timers = {slot.object_timer for slot in ENEMY_SLOTS}
    assert TIMERS.PIDGIT_CARPET not in enemy_timers


def test_timer_addresses_are_distinct():
    """No two timers may share an address."""
    addresses = {
        name: value
        for name, value in vars(TIMERS).items()
        if not name.startswith("_") and isinstance(value, int)
    }
    duplicates = [
        (a, b)
        for i, (a, av) in enumerate(addresses.items())
        for b, bv in list(addresses.items())[i + 1 :]
        if av == bv
    ]
    assert not duplicates, f"timers share an address: {duplicates}"


@pytest.mark.slow
def test_carpet_timer_reads_zero_without_a_carpet(env_no_render):
    """With no carpet in play the timer must read 0, not stale garbage.

    A wrong address typically shows up here as a non-zero value that drifts,
    because it is really tracking something else.
    """
    env = env_no_render
    obs, info = env.reset()

    for _ in range(200):
        obs, _, terminated, truncated, info = env.step(np.int64(1))
        if terminated or truncated:
            break
        assert info['pc'].pidgit_carpet_timer == 0


def test_float_addresses_are_not_the_same():
    """The live timer and the static budget are two different bytes.

    $0553 (`JumpFloatLength`) is the per-character budget and never moves;
    $04C9 is the countdown. Reading the budget as the timer shows a meter
    pinned at 60/60 that never drains, which is what this guards against.
    """
    assert TIMERS.FLOAT == 0x04C9
    assert TIMERS.FLOAT_LENGTH == 0x0553
    assert TIMERS.FLOAT != TIMERS.FLOAT_LENGTH


@pytest.mark.slow
def test_float_timer_counts_down_while_peach_floats():
    """Holding A through a descent must drain the float timer from its budget."""
    from smb2_gym.app import InitConfig
    from smb2_gym.smb2_env import SuperMarioBros2Env

    A, NOOP, RIGHT = 4, 0, 1
    env = SuperMarioBros2Env(init_config=InitConfig(level="1-1", character="peach"))
    try:
        obs, info = env.reset()
        assert info['pc'].float_length == 60

        # Walk clear of the starting vine, then settle on solid ground.
        for _ in range(400):
            obs, _, _, _, info = env.step(np.int64(RIGHT))
        previous, stable = None, 0
        for _ in range(180):
            obs, _, _, _, info = env.step(np.int64(NOOP))
            y = info['pos'].y_global
            stable = stable + 1 if y == previous else 0
            previous = y
            if stable > 25:
                break
        assert info['pc'].float_timer == 0, "timer should be idle while grounded"

        # Hold A through the jump and the descent that follows.
        seen = []
        for _ in range(60):
            obs, _, terminated, truncated, info = env.step(np.int64(A))
            seen.append(info['pc'].float_timer)
            if terminated or truncated:
                break

        assert max(seen) == 60, f"float never armed: {sorted(set(seen))}"
        assert min(seen) < 60, f"float never drained: {sorted(set(seen))}"
        # It must tick down rather than jump about
        drops = sum(1 for a, b in pairwise(seen) if b == a - 1)
        assert drops >= 10, f"expected a steady countdown, got {seen}"
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.parametrize("character", ["mario", "toad", "luigi"])
def test_float_timer_stays_zero_for_everyone_else(character):
    """Only Peach can float, so the timer never arms for the others."""
    from smb2_gym.app import InitConfig
    from smb2_gym.smb2_env import SuperMarioBros2Env

    A, NOOP, RIGHT = 4, 0, 1
    env = SuperMarioBros2Env(init_config=InitConfig(level="1-1", character=character))
    try:
        obs, info = env.reset()
        assert info['pc'].float_length == 0

        for _ in range(400):
            obs, _, _, _, info = env.step(np.int64(RIGHT))
        for _ in range(60):
            obs, _, _, _, info = env.step(np.int64(NOOP))

        seen = set()
        for _ in range(60):
            obs, _, terminated, truncated, info = env.step(np.int64(A))
            seen.add(info['pc'].float_timer)
            if terminated or truncated:
                break
        assert seen == {0}, f"{character} armed a float timer: {sorted(seen)}"
    finally:
        env.close()


def test_deprecated_spelling_still_works(env_no_render):
    """`pidget_carpet_timer` was public API, so the misspelling must keep working."""
    env = env_no_render
    obs, info = env.reset()
    pc = info['pc']
    assert pc.pidget_carpet_timer == pc.pidgit_carpet_timer
