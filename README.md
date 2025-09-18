# smb2-gym

[![Python](https://img.shields.io/pypi/pyversions/smb2-gym)](https://pypi.org/project/smb2-gym/)
[![PyPI](https://img.shields.io/pypi/v/smb2-gym)](https://pypi.org/project/smb2-gym/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Gymnasium environment for Super Mario Bros 2 (Europe/Doki Doki Panic version) using TetaNES emulator Python bindings. Perfect for reinforcement learning experiments and research.

**Features:**
- Curated action sets for faster training (`simple`, `complex`)
- Comprehensive game state via info dict (35+ properties)
- Multiple initialization modes (character/level, custom ROMs, save states)
- Human-playable interface with keyboard controls
- Up to 350+ and 750+ FPS rendered and non-rendered respectively

A somewhat comprehensive list of the available RAM map properties is available at [Data Crystal](https://datacrystal.tcrf.net/wiki/Super_Mario_Bros._2_(NES)/RAM_map), but this library includes many extras that are not documented anywhere.

## Installation

```bash
pip install smb2-gym
```

## Quick Start

### Basic Usage

```python
from smb2_gym import SuperMarioBros2Env
from smb2_gym.app import InitConfig

# Create environment with character/level mode
config = InitConfig(level="1-1", character="luigi")
env = SuperMarioBros2Env(
    init_config=config,
    render_mode="human",     # "human" or None
    action_type="simple"     # "simple" (11), "complex" (16), or "all" (256)
)

# Reset environment
obs, info = env.reset()

# Run game loop
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # Access game state from info dict
    print(f"Lives: {info['life']}, Hearts: {info['hearts']}")
    print(f"Position: ({info['x_pos_global']}, {info['y_pos_global']})")

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Initialization Modes

```python
from smb2_gym.app import InitConfig

# 1. Character/Level mode (default)
config = InitConfig(level="1-1", character="peach")

# 2. Built-in ROM variant mode
config = InitConfig(rom="prg0", save_state="level_1_1.sav")

# 3. Custom ROM mode
config = InitConfig(
    rom_path="/path/to/your/smb2.nes",
    save_state_path="/path/to/save.sav"  # Optional
)
```

### Custom Reward Function

```python
from smb2_gym import SuperMarioBros2Env
from smb2_gym.app import InitConfig

class CustomSMB2Env(SuperMarioBros2Env):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Custom reward based on x-position progress
        reward = info['x_pos_global'] / 100.0

        # Bonus for collecting cherries
        reward += info['cherries'] * 10

        # Bonus for hearts
        reward += info['hearts'] * 5

        # Penalty for losing a life
        if info.get('life_lost'):
            reward -= 100

        return obs, reward, terminated, truncated, info

# Usage
config = InitConfig(level="1-1", character="luigi")
env = CustomSMB2Env(init_config=config, action_type="simple")
```

## Play as Human

The package includes a human-playable interface with multiple initialization modes:

### Character/Level Mode (Default)
```bash
# Play as Luigi on level 1-1
smb2-play --level 1-1 --char luigi --scale 3

# TODO: Yet to implement all levels states
# Play as Peach on level 2-3
smb2-play --level 2-3 --char peach 
```

### Built-in ROM Variant Mode
```bash
# Use specific ROM variant with save state
smb2-play --rom prg0_edited --save-state easy_combined_curriculum.sav
```

### Custom ROM Mode  
```bash
# Use your own ROM file
smb2-play --custom-rom /path/to/smb2.nes

# Use custom ROM with save state
smb2-play --custom-rom /path/to/smb2.nes --custom-state /path/to/save.sav

# Start from beginning without save state
smb2-play --custom-rom /path/to/smb2.nes --no-save-state
```

### Controls

**Primary Controls:**
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

### CLI Options

**Character/Level Mode:**
- `--level`: Level to play (1-1 through 7-2, default: 1-1)
- `--char`: Character (mario, luigi, peach, toad, default: luigi)

**Built-in ROM Mode:**
- `--rom`: ROM variant (prg0, prg0_edited)
- `--save-state`: Save state filename

**Custom ROM Mode:**
- `--custom-rom`: Path to custom ROM file
- `--custom-state`: Path to custom save state (optional)
- `--no-save-state`: Start from beginning without loading save state

**Display:**
- `--scale`: Display scale factor (1-4, default: 3)

## Disclaimer

This project is for educational and research purposes only. Users must provide their own legally obtained ROM files.
