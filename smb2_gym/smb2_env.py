"""Super Mario Bros 2 (Europe) Gymnasium Environment."""

import os
from typing import (
    Any,
    Optional,
)

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from tetanes_py import NesEnv

from .actions import (
    COMPLEX_ACTIONS,
    SIMPLE_ACTIONS,
    ActionType,
    action_to_buttons,
    actions_to_buttons,
    get_action_meanings,
)
from .constants import (
    CHARACTER,
    CHARACTER_NAMES,
    CHERRIES,
    CONTINUES,
    CURRENT_LEVEL,
    DOOR_TRANSITION_TIMER,
    ENEMIES_DEFEATED,
    ENEMY_HEALTH,
    ENEMY_ID,
    ENEMY_X_POS,
    ENEMY_Y_POS,
    FLOAT_TIMER,
    GAME_INIT_FRAMES,
    ITEM_HOLDING,
    ITEM_PULLED,
    LEVEL_NAMES,
    LEVEL_TILESET,
    LEVEL_TRANSITION,
    LEVELS_FINISHED_LUIGI,
    LEVELS_FINISHED_MARIO,
    LEVELS_FINISHED_PEACH,
    LEVELS_FINISHED_TOAD,
    LIFE_METER,
    LIVES,
    MAX_CHERRIES,
    MAX_COINS,
    MAX_CONTINUES,
    MAX_HEARTS,
    MAX_LIVES,
    MAX_SAVE_SLOTS,
    ON_VINE,
    PAGE_SIZE,
    PLAYER_SPEED,
    PLAYER_STATE,
    PLAYER_X_PAGE,
    PLAYER_X_POSITION,
    PLAYER_Y_PAGE,
    PLAYER_Y_POSITION,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    STARMAN_TIMER,
    STOPWATCH_TIMER,
    SUBSPACE_COINS,
    SUBSPACE_STATUS,
    SUBSPACE_TIMER,
    VEGETABLES_PULLED,
    WORLD_NUMBER,
    Y_POSITION_WRAPAROUND_THRESHOLD,
)


class SuperMarioBros2Env(gym.Env):
    """
    Gymnasium environment for Super Mario Bros 2 (Europe).

    This environment provides a minimal interface to the NES emulator,
    returning pixel observations and allowing all 256 button combinations as 
    actions. 

    Rewards are always 0 - users should implement their own reward functions 
    based on the RAM values available in the info dict.
    """

    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: Optional[int] = None,
        level: str = "1-1",
        character: int = 3,
        action_type: ActionType = "simple",
        reset_on_life_loss: bool = False,
    ):
        """Initialize the SMB2 environment.

        Args:
            render_mode: 'human' or None
            max_episode_steps: Maximum steps per episode (for truncation)
            level: Level to start at (e.g., "1-1", "7-2")
            character: Character to play as (0=Mario, 1=Princess, 2=Toad, 3=Luigi)
            action_type: Type of action space
            reset_on_life_loss: If True, episode terminates when Mario loses a life
        """
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.reset_on_life_loss = reset_on_life_loss

        self._init_game_parameters(level, character, action_type)
        self._init_emulator()
        self._init_spaces()
        self._init_state_tracking()

    def _init_game_parameters(self, level: str, character: int, action_type: str) -> None:
        """Initialize and validate game parameters.

        Args:
            level: Starting level
            character: Character selection
            action_type: Type of action space
        """
        # Validate and store level
        self.starting_level = level
        self.starting_level_id = self._validate_level(level)

        # Validate and store character
        if character not in [0, 1, 2, 3]:
            raise ValueError(
                f"Invalid character {character}. Must be 0-3 (0=Mario, 1=Princess, 2=Toad, 3=Luigi)"
            )
        self.starting_character = character

        # Validate and store action type
        if action_type not in ["all", "complex", "simple"]:
            raise ValueError(
                f"Invalid action_type '{action_type}'. Must be 'all', 'complex', or 'simple'"
            )
        self.action_type = action_type

    def _init_emulator(self) -> None:
        """Initialize the NES emulator and load ROM."""
        # Always use the bundled ROM
        rom_path = os.path.join(
            os.path.dirname(__file__), '_nes', 'roms', 'super_mario_bros2_europe.nes'
        )
        rom_path = os.path.abspath(rom_path)

        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"Bundled ROM file not found: {rom_path}")

        # Initialize TetaNES. We want video frames for display
        self._nes = NesEnv(headless=False)

        # Load ROM
        with open(rom_path, 'rb') as f:
            rom_data = f.read()
        rom_name = os.path.basename(rom_path)
        self._nes.load_rom(rom_name, rom_data)

    def _init_spaces(self) -> None:
        """Initialize observation and action spaces."""
        # Define observation space
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8
        )

        # Define action space based on action_type
        if self.action_type == "all":
            self.action_space = spaces.Discrete(256)
            self._action_meanings = get_action_meanings()
        elif self.action_type == "complex":
            self.action_space = spaces.Discrete(len(COMPLEX_ACTIONS))
            self._action_meanings = COMPLEX_ACTIONS
        elif self.action_type == "simple":
            self.action_space = spaces.Discrete(len(SIMPLE_ACTIONS))
            self._action_meanings = SIMPLE_ACTIONS

    def _init_state_tracking(self) -> None:
        """Initialize state tracking variables."""
        self._done = False
        self._episode_steps = 0
        self._previous_lives = None  # Track lives to detect life loss

    # ---- Primary Gym methods ---------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment by loading a save state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            observation: Initial frame
            info: Initial info dict
        """
        super().reset(seed=seed)

        # Reset NES first
        self._nes.reset()
        self._done = False
        self._episode_steps = 0

        # Construct path to save state file
        character_name = CHARACTER_NAMES[self.starting_character].lower()
        level_filename = self.starting_level + '.sav'
        save_path = os.path.join(
            os.path.dirname(__file__), '_nes', 'levels', character_name, level_filename
        )

        # Load the save state - this is required
        if not os.path.exists(save_path):
            raise FileNotFoundError(
                f"Save state file not found: {save_path}. "
                f"Required save states must exist for character {character_name} level {self.starting_level}"
            )

        self.load_state_from_path(save_path)

        # Get one frame after loading save state
        obs, _, _, _, _ = self._nes.step([False] * 8, render=True)

        info = self.info

        # Initialize life tracking for detecting life loss
        self._previous_lives = self.lives

        return np.array(obs), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step the environment.

        Args:
            action: Discrete action (0-255)

        Returns:
            observation: Current frame
            reward: Always 0.0
            terminated: True if game over
            truncated: True if max steps reached
            info: dict with game state
        """
        if self._done:
            raise RuntimeError("Cannot step after episode is done. Call reset().")

        # Convert and validate action to buttons
        buttons = self._validate_and_convert_action(action)

        # 1. Step emulator
        obs, _, _, _, nes_info = self._nes.step(buttons.tolist(), render=True)
        self._episode_steps += 1

        # 2. Get game state
        info = self.info
        info.update(nes_info)  # Include NES emulator info

        # 3. Check for life loss and update tracking
        life_lost = self._detect_life_loss()
        if life_lost:
            info['life_lost'] = True

        # Update life tracking for next step
        self._previous_lives = self.lives

        # 4. Check termination
        terminated = self.is_game_over or life_lost
        truncated = (
            self.max_episode_steps is not None and self._episode_steps >= self.max_episode_steps
        )

        self._done = terminated or truncated
        reward = 0.0  # Always return 0 reward

        return np.array(obs), reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render the environment.

        Returns:
            RGB array for display, None if no render mode
        """
        if self.render_mode == 'human':
            return self._nes.get_observation()
        return None

    def _read_ram_safe(self, address: int, default: int = 0) -> int:
        """Safely read from RAM with fallback.

        Args:
            address: RAM address to read
            default: Default value if RAM reading is not available

        Returns:
            Value at RAM address or default
        """
        if hasattr(self._nes, 'read_ram'):
            return self._nes.read_ram(address)
        return default

    # ---- Properties ------------------------------------------------

    @property
    def info(self) -> dict[str, Any]:
        """Get current game info from RAM.

        Returns:
            dict with game state information
        """
        return {
            'life': self.lives,
            'x_pos_global': self.x_position_global,
            'y_pos_global': self.y_position_global,
            'x_pos_local': self.x_position,
            'y_pos_local': self.y_position,
            'x_page': self.x_page,
            'y_page': self.y_page,
            'world': self.world,
            'level': self.level,
            'character': self.character,
            'hearts': self.hearts,
            'cherries': self.cherries,
            'coins': self.coins,
            'starman_timer': self.starman_timer,
            'subspace_timer': self.subspace_timer,
            'stopwatch_timer': self.stopwatch_timer,
            'holding_item': self.holding_item,
            'item_pulled': self.item_pulled,
            'continues': self.continues,
            'player_speed': self.player_speed,
            'on_vine': self.on_vine,
            'float_timer': self.float_timer,
            'levels_finished': self.levels_finished,
            'door_transition_timer': self.door_transition_timer,
            'vegetables_pulled': self.vegetables_pulled,
            'subspace_status': self.subspace_status,
            'enemies_defeated': self.enemies_defeated,
        }

    @property
    def is_game_over(self) -> bool:
        """Check if game is over (lives = 0)."""
        if self._episode_steps < GAME_INIT_FRAMES:  # Give the game 5 seconds to fully initialize
            return False

        lives = self.lives
        return lives == 0

    def _detect_life_loss(self) -> bool:
        """Detect if Mario lost a life this step.

        Returns:
            True if a life was lost, False otherwise
        """
        if not self.reset_on_life_loss:
            return False

        if self._previous_lives is None:
            return False

        # Don't detect life loss during initialization
        if self._episode_steps < GAME_INIT_FRAMES:
            return False

        current_lives = self.lives
        return current_lives < self._previous_lives

    @property
    def lives(self) -> int:
        """Get current lives."""
        lives = self._read_ram_safe(LIVES, default=2)
        # Validate the value - SMB2 has 2-5 lives typically
        if 0 <= lives <= MAX_LIVES:
            return lives
        return 2  # Default if invalid

    @property
    def x_position_global(self) -> int:
        """Get player global X position."""
        x_page = self._read_ram_safe(PLAYER_X_PAGE, default=0)
        x_pos = self._read_ram_safe(PLAYER_X_POSITION, default=0)
        return (x_page * PAGE_SIZE) + x_pos

    @property
    def x_position(self) -> int:
        """Get player global X position."""
        x_pos = self._read_ram_safe(PLAYER_X_POSITION, default=0)
        return x_pos

    @property
    def x_page(self) -> int:
        """Get the X page of the player position."""
        x_page = self._read_ram_safe(PLAYER_X_PAGE)
        return x_page

    @property
    def y_position_global(self) -> int:
        """Get player global Y position."""
        y_page = self._read_ram_safe(PLAYER_Y_PAGE, default=0)
        y_pos = self._read_ram_safe(PLAYER_Y_POSITION, default=0)
        global_position = (y_page * PAGE_SIZE) + y_pos

        # NOTE: Handle out-of-bounds positions (wraparound when Mario goes above
        # screen). Valid Y positions in SMB2 are typically 0-1000, anything > 10000
        # is wraparound
        if global_position > Y_POSITION_WRAPAROUND_THRESHOLD:
            return 0

        return global_position

    @property
    def y_position(self) -> int:
        """Get player global Y position."""
        y_pos = self._read_ram_safe(PLAYER_Y_POSITION, default=0)
        return y_pos

    @property
    def y_page(self) -> int:
        """Get the Y page of the player position."""
        y_page = self._read_ram_safe(PLAYER_Y_PAGE)
        return y_page

    @property
    def world(self) -> int:
        """Get current world number. RAM is 0-based, display is 1-based."""
        return self._read_ram_safe(WORLD_NUMBER, default=0) + 1

    @property
    def level(self) -> str:
        """Get current level string."""
        level_id = self._read_ram_safe(CURRENT_LEVEL, default=0)
        return LEVEL_NAMES.get(level_id, f"L-{level_id:02X}")

    @property
    def character(self) -> int:
        """Get selected character (0=Mario, 1=Princess, 2=Toad, 3=Luigi)."""
        char = self._read_ram_safe(CHARACTER, default=0)
        if 0 <= char <= 3:
            return char
        return 0

    @property
    def hearts(self) -> int:
        """Get current hearts (1-4)."""
        life_meter = self._read_ram_safe(LIFE_METER, default=0x1F)  # Default 2 hearts
        # Convert: 0x0F=1, 0x1F=2, 0x2F=3, 0x3F=4
        # The upper nibble indicates hearts - 1
        if life_meter == 0x0F:
            return 1
        elif life_meter == 0x1F:
            return 2
        elif life_meter == 0x2F:
            return 3
        elif life_meter == 0x3F:
            return 4
        else:
            # If value doesn't match expected pattern, use lower nibble + 1
            hearts = (life_meter & 0x0F) + 1
            if 1 <= hearts <= MAX_HEARTS:
                return hearts
            return 2  # Default

    @property
    def cherries(self) -> int:
        """Get cherries collected."""
        cherries = self._read_ram_safe(CHERRIES, default=0)
        # Cherries should be 0-19 (max per level)
        if 0 <= cherries <= MAX_CHERRIES:
            return cherries
        return 0

    @property
    def coins(self) -> int:
        """Get coins collected in Subspace."""
        coins = self._read_ram_safe(SUBSPACE_COINS, default=0)
        # Coins should be reasonable
        if 0 <= coins <= MAX_COINS:
            return coins
        return 0

    @property
    def starman_timer(self) -> int:
        """Get starman timer."""
        return self._read_ram_safe(STARMAN_TIMER, default=0)

    @property
    def subspace_timer(self) -> int:
        """Get subspace timer."""
        return self._read_ram_safe(SUBSPACE_TIMER, default=0)

    @property
    def stopwatch_timer(self) -> int:
        """Get stopwatch timer."""
        return self._read_ram_safe(STOPWATCH_TIMER, default=0)

    @property
    def enemies_defeated(self) -> int:
        """Get count of enemies defeated (for heart spawning)."""
        return self._read_ram_safe(ENEMIES_DEFEATED, default=0)

    @property
    def holding_item(self) -> bool:
        """Check if character is holding an item."""
        return self._read_ram_safe(ITEM_HOLDING, default=0) == 1

    @property
    def item_pulled(self) -> int:
        """Get item pulled from ground."""
        return self._read_ram_safe(ITEM_PULLED, default=0)

    @property
    def continues(self) -> int:
        """Get number of continues."""
        continues = self._read_ram_safe(CONTINUES, default=0)
        if 0 <= continues <= MAX_CONTINUES:
            return continues
        return 0

    @property
    def player_speed(self) -> int:
        """Get player horizontal speed."""
        return self._read_ram_safe(PLAYER_SPEED, default=0)

    @property
    def on_vine(self) -> bool:
        """Check if character is on a vine."""
        return self._read_ram_safe(ON_VINE, default=0) == 1

    @property
    def float_timer(self) -> int:
        """Get Princess float timer."""
        return self._read_ram_safe(FLOAT_TIMER, default=0)

    @property
    def levels_finished(self) -> dict:
        """Get levels finished per character."""
        return {
            'mario': self._read_ram_safe(LEVELS_FINISHED_MARIO, default=0),
            'peach': self._read_ram_safe(LEVELS_FINISHED_PEACH, default=0),
            'toad': self._read_ram_safe(LEVELS_FINISHED_TOAD, default=0),
            'luigi': self._read_ram_safe(LEVELS_FINISHED_LUIGI, default=0),
        }

    @property
    def door_transition_timer(self) -> int:
        """Get door transition timer."""
        return self._read_ram_safe(DOOR_TRANSITION_TIMER, default=0)

    @property
    def vegetables_pulled(self) -> int:
        """Get total vegetables pulled."""
        return self._read_ram_safe(VEGETABLES_PULLED, default=0)

    @property
    def subspace_status(self) -> int:
        """Get subspace status (0=not in subspace, 2=in subspace)."""
        return self._read_ram_safe(SUBSPACE_STATUS, default=0)

    # ---- Validators ------------------------------------------------

    def _validate_and_convert_action(self, action: int) -> np.ndarray:
        """Validate and convert action to button array based on action type.

        Args:
            action: Discrete action index

        Returns:
            Button array for NES controller

        Raises:
            ValueError: If action is invalid for the current action type
        """
        if self.action_type == "all":
            if not 0 <= action <= 255:
                raise ValueError(f"Invalid action {action}. Must be 0-255 for 'all' action type")
            return action_to_buttons(action)
        elif self.action_type == "complex":
            if action >= len(COMPLEX_ACTIONS):
                raise ValueError(f"Invalid action {action}. Must be 0-{len(COMPLEX_ACTIONS)-1}")
            return actions_to_buttons(COMPLEX_ACTIONS[action])
        elif self.action_type == "simple":
            if action >= len(SIMPLE_ACTIONS):
                raise ValueError(f"Invalid action {action}. Must be 0-{len(SIMPLE_ACTIONS)-1}")
            return actions_to_buttons(SIMPLE_ACTIONS[action])
        else:
            raise ValueError('Action type not supported.')

    def _validate_level(self, level: str) -> int:
        """Validate and convert level string to level ID.

        Args:
            level: Level string like "1-1" or "7-2"

        Returns:
            Level ID for RAM

        Raises:
            ValueError: If level is invalid
        """
        # Reverse mapping
        level_str_to_id = {v: k for k, v in LEVEL_NAMES.items()}

        if level not in level_str_to_id:
            valid_levels = sorted(LEVEL_NAMES.values())
            raise ValueError(
                f"Invalid level '{level}'. Valid levels are: {', '.join(valid_levels)}"
            )

        return level_str_to_id[level]

    # ---- Other bindings --------------------------------------------

    def get_action_meanings(self) -> list:
        """Get the meanings of actions for this environment.

        Returns:
            List of action meanings based on the action_type
        """
        return self._action_meanings

    def save_state(self, slot: int) -> None:
        """Save current emulator state to a slot.

        Args:
            slot: Save state slot (0-9)
        """
        if not 0 <= slot < MAX_SAVE_SLOTS:
            raise ValueError(f"Slot must be between 0-9, got {slot}")
        self._nes.save_state(slot)

    def load_state(self, slot: int) -> None:
        """Load emulator state from a slot.

        Args:
            slot: Save state slot (0-9)
        """
        if not 0 <= slot < MAX_SAVE_SLOTS:
            raise ValueError(f"Slot must be between 0-9, got {slot}")
        self._nes.load_state(slot)

    def save_state_to_path(self, filepath: str) -> None:
        """Save current emulator state to a file.

        Args:
            filepath: Path where to save the state file
        """
        self._nes.save_state_to_path(filepath)

    def load_state_from_path(self, filepath: str) -> None:
        """Load emulator state from a file.

        Args:
            filepath: Path to the state file to load
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Save state file not found: {filepath}")
        self._nes.load_state_from_path(filepath)
