"""Pytest configuration and fixtures for smb2-gym tests."""

import pytest
from smb2_gym import SuperMarioBros2Env
from smb2_gym.app import InitConfig


@pytest.fixture
def basic_env_config():
    """Basic environment configuration for testing."""
    return InitConfig(level="1-1", character="luigi")


@pytest.fixture
def env_no_render(basic_env_config):
    """Create an environment without rendering for faster testing."""
    env = SuperMarioBros2Env(
        init_config=basic_env_config,
        render_mode=None,
        action_type="simple"
    )
    yield env
    env.close()


@pytest.fixture
def env_with_render(basic_env_config):
    """Create an environment with rendering (marked as slow)."""
    env = SuperMarioBros2Env(
        init_config=basic_env_config,
        render_mode="human",
        action_type="simple"
    )
    yield env
    env.close()


@pytest.fixture(scope="session")
def cli_command():
    """CLI command name for testing."""
    return "smb2-play"


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )