"""Super Mario Bros 2 Gymnasium Environment."""

from importlib.metadata import (
    PackageNotFoundError,
    version,
)

from .actions import (
    COMPLEX_ACTIONS,
    SIMPLE_ACTIONS,
    ActionType,
)
from .app import InitConfig
from .constants import *
from .smb2_env import SuperMarioBros2Env


try:
    # Single source of truth: the version in pyproject.toml. Hardcoding it here
    # as well means the two drift apart on every release that forgets one.
    __version__ = version("smb2-gym")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0.dev0"
__all__ = [
    "COMPLEX_ACTIONS",
    "SIMPLE_ACTIONS",
    "ActionType",
    "InitConfig",
    "SuperMarioBros2Env",
]
