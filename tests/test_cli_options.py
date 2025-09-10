"""Tests for CLI options and package installation."""

import subprocess


def test_cli_command_available(cli_command):
    """Test that the smb2-play CLI command is available."""
    result = subprocess.run([cli_command, "--help"], capture_output=True, text=True, timeout=5)

    assert result.returncode == 0, f"CLI command failed with error: {result.stderr}"


def test_cli_help_shows_all_options(cli_command):
    """Test that CLI help shows all expected options from README."""
    result = subprocess.run([cli_command, "--help"], capture_output=True, text=True, timeout=5)

    assert result.returncode == 0, "Help command should succeed"
    help_text = result.stdout

    # Check that help mentions the main option categories from README
    expected_options = ["--level", "--char", "--rom", "--custom-rom", "--scale", "--no-save-state"]

    for option in expected_options:
        assert option in help_text, f"Option {option} should be mentioned in help output"


def test_package_can_be_imported():
    """Test that the smb2_gym package can be imported."""
    import smb2_gym
    assert smb2_gym is not None


def test_main_components_can_be_imported():
    """Test that main package components can be imported."""
    from smb2_gym import SuperMarioBros2Env
    from smb2_gym.app import InitConfig

    assert SuperMarioBros2Env is not None
    assert InitConfig is not None


def test_environment_can_be_created(env_no_render):
    """Test that a basic environment can be created successfully."""
    assert env_no_render is not None
    assert hasattr(env_no_render, 'reset')
    assert hasattr(env_no_render, 'step')
    assert hasattr(env_no_render, 'close')
    assert hasattr(env_no_render, 'action_space')
    assert hasattr(env_no_render, 'observation_space')

    # Test that we can reset the environment
    obs, info = env_no_render.reset()
    assert obs is not None
    assert info is not None
