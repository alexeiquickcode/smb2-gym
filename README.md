# smb2-gym

[![Python](https://img.shields.io/pypi/pyversions/smb2-gym)](https://pypi.org/project/smb2-gym/)
[![PyPI](https://img.shields.io/pypi/v/smb2-gym)](https://pypi.org/project/smb2-gym/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Gymnasium environment for Super Mario Bros 2 (Europe/Doki Doki Panic version) using customer TetaNES emulator Python bindings. Perfect for reinforcement learning experiments. Currently I have only compiled the bindings on Arch Linux but I will add more soon and also create a fork of the TetaNES repo. 

A full list of the available RAM map properties is available at RAM [Data Crystal](https://datacrystal.tcrf.net/wiki/Super_Mario_Bros._2_(NES)/RAM_map) but I have added some extras that aare not listed there. Currently I have only added save states for the 1-1 but I will continue to add some more.

## Installation

```bash
pip install smb2-gym
```

## Quick Start

### Basic Usage

```python
import gymnasium as gym
from smb2_gym import SuperMarioBros2Env

# Create environment
env = SuperMarioBros2Env(
    render_mode="human",  # "human" or "rgb_array"
    level="1-1",          # Starting level
    character=3,          # 0=Mario, 1=Princess, 2=Toad, 3=Luigi
    action_type="simple"  # "simple", "complex", or "all"
)

# Reset environment
obs, info = env.reset()

# Run game loop
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Custom Reward Function

```python
from smb2_gym import SuperMarioBros2Env

class CustomSMB2Env(SuperMarioBros2Env):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Custom reward based on x-position progress
        reward = info['x_pos'] / 100.0

        # Bonus for collecting cherries
        ram = self.get_ram()
        cherries = ram[0x06FE]
        reward += cherries * 10

        return obs, reward, terminated, truncated, info
```

## Play as Human

```bash
smb2-play --level 1-1 --char mario --scale 3
```

**Controls:**
- Arrow Keys: Move
- Z: A button (Jump)
- X: B button (Pick up/Throw)
- Enter: Start
- Right Shift: Select
- P: Pause
- R: Reset
- ESC: Quit

**Alternative Controls:**
- WASD: Move
- J: A button
- K: B button

**Save States:**
- F5: Save state
- F9: Load state

**Options:**
- `--level`: Level to play (1-1 through 7-2, default: 1-1)
- `--char`: Character (mario, luigi, peach, toad, default: mario)
- `--scale`: Display scale factor (default: 3)

## Disclaimer

This project is for educational and research purposes only. Users must provide their own legally obtained ROM files.
