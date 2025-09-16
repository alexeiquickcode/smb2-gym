"""Test random actions in the SMB2 environment."""

import pytest


@pytest.mark.slow
def test_random_actions_with_rendering(env_with_render):
    """Test the environment with random actions and rendering as shown in README."""

    # Set frame speed to 10x for faster testing while still rendering
    env_with_render.set_frame_speed(15.0)

    # Reset environment
    obs, info = env_with_render.reset()

    # Verify initial state
    assert obs is not None, "Initial observation should not be None"
    assert info is not None, "Initial info should not be None"
    assert 'life' in info, "Info should contain 'life' key"
    assert 'hearts' in info, "Info should contain 'hearts' key"
    assert 'x_pos_global' in info, "Info should contain 'x_pos_global' key"
    assert 'y_pos_global' in info, "Info should contain 'y_pos_global' key"

    steps_completed = 0

    # Run for 5000 frames at 10x speed with rendering
    for step in range(5000):
        action = env_with_render.action_space.sample()
        obs, reward, terminated, truncated, info = env_with_render.step(action)

        # Verify step results
        assert obs is not None, f"Observation should not be None at step {step}"
        assert info is not None, f"Info should not be None at step {step}"
        assert isinstance(reward, (int, float)), f"Reward should be numeric at step {step}"
        assert isinstance(terminated, bool), f"Terminated should be boolean at step {step}"
        assert isinstance(truncated, bool), f"Truncated should be boolean at step {step}"

        steps_completed = step

        if terminated or truncated:
            obs, info = env_with_render.reset()
            assert obs is not None, "Observation after reset should not be None"
            assert info is not None, "Info after reset should not be None"

    # Verify we completed some steps
    assert steps_completed > 0, "Should have completed at least one step"


def test_random_actions_no_rendering(env_no_render):
    """Test the environment with random actions without rendering (faster)."""

    # Set frame speed to 10x for faster testing
    env_no_render.set_frame_speed(10.0)

    # Reset environment
    obs, info = env_no_render.reset()

    # Verify initial state
    assert obs is not None, "Initial observation should not be None"
    assert info is not None, "Initial info should not be None"

    episodes_completed = 0
    steps_total = 0

    # Run for 5000 frames at 10x speed
    for step in range(5000):
        action = env_no_render.action_space.sample()
        obs, reward, terminated, truncated, info = env_no_render.step(action)

        # Basic assertions
        assert obs is not None, f"Observation should not be None at step {step}"
        assert isinstance(reward, (int, float)), f"Reward should be numeric"

        steps_total += 1

        if terminated or truncated:
            episodes_completed += 1
            obs, info = env_no_render.reset()

            # Don't reset too many times in test
            if episodes_completed >= 3:
                break

    # Verify we completed some meaningful testing
    assert steps_total > 10, f"Should have completed more steps, only did {steps_total}"
