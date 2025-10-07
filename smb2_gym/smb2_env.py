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
from .app import InitConfig
from .app.info_display import create_info_panel
from .app.rendering import render_frame
from .constants import (
    GAME_INIT_FRAMES,
    LIVES,
    MAX_SAVE_SLOTS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from .game_state.character_stats import get_character_stats
from .game_state.collision_map import CollisionMapMixin
from .game_state.enemies import EnemiesMixin
from .game_state.player_state import PlayerStateMixin
from .game_state.position import PositionMixin
from .game_state.timers import TimersMixin
class SuperMarioBros2Env(
    gym.Env,
    PositionMixin,
    PlayerStateMixin,
    TimersMixin,
    EnemiesMixin,
    CollisionMapMixin,
):
    """
    Gymnasium environment for Super Mario Bros 2 (Europe).

    This environment provides a minimal interface to the NES emulator,
    returning pixel observations and allowing all 256 button combinations as
    actions.

    Rewards are always 0 - users should implement their own reward functions
    based on the RAM values available in the info dict.
    """

    # Number of frames to wait during area transitions before accepting new coordinates
    AREA_TRANSITION_FRAMES = 98

    def __init__(
        self,
        init_config: InitConfig,
        render_mode: Optional[str] = None,
        max_episode_steps: Optional[int] = None,
        action_type: ActionType = "simple",
        reset_on_life_loss: bool = False,
        render_fps: Optional[int] = None,
        frame_method: str = "rgb",
        env_name: Optional[str] = None,
    ):
        """Initialize the SMB2 environment.

        Args:
            init_config: InitConfig object specifying initialization mode
            render_mode: 'human' or None
            max_episode_steps: Maximum steps per episode (for truncation)
            action_type: Type of action space
            reset_on_life_loss: If True, episode terminates when Mario loses a life
            render_fps: FPS for human rendering (None = no limit, good for training)
            frame_method: Frame rendering method ('rgb', 'grayscale')
                - 'rgb': RGB rendering
                - 'grayscale': Grayscale rendering (faster, 67% less memory)
        """
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.reset_on_life_loss = reset_on_life_loss
        self.init_config = init_config
        self.render_fps = render_fps
        self.env_name = env_name
        if self.env_name:
            print(f'Creating {self.env_name} environment...')

        # Validate and store frame method
        valid_frame_methods = ["rgb", "grayscale"]
        if frame_method not in valid_frame_methods:
            raise ValueError(
                f"Invalid frame_method '{frame_method}'. Must be one of {valid_frame_methods}"
            )
        self.frame_method = frame_method

        # Store relevant attributes (only meaningful for built-in ROM mode)
        if not self.init_config.rom_path:  # Built-in ROM mode
            self.starting_level = self.init_config.level
            self.starting_level_id = self.init_config.level_id
            self.starting_character = self.init_config.character_id
        else:  # Custom ROM mode
            self.starting_level = None
            self.starting_level_id = None
            self.starting_character = None

        # Validate and store action type
        if action_type not in ["all", "complex", "simple"]:
            raise ValueError(
                f"Invalid action_type '{action_type}'. Must be 'all', 'complex', or 'simple'"
            )
        self.action_type = action_type

        self._init_emulator()
        self._init_spaces()
        self._init_state_tracking()

        # Initialize rendering attributes but defer pygame init until needed
        self._screen = None
        self._clock = None
        self._pygame_initialized = False

    def _init_emulator(self) -> None:
        """Initialize the NES emulator and load ROM."""
        rom_path = self.init_config.get_rom_path()
        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"ROM file not found: {rom_path}")

        # Initialize TetaNES with frame rendering method
        self._nes = NesEnv(headless=False, frame_method=self.frame_method)

        # Load ROM
        with open(rom_path, 'rb') as f:
            rom_data = f.read()
        rom_name = os.path.basename(rom_path)
        self._nes.load_rom(rom_name, rom_data)

    def _init_spaces(self) -> None:
        """Initialize observation and action spaces."""
        # Define observation space based on frame method
        if self.frame_method == "grayscale":
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(SCREEN_HEIGHT, SCREEN_WIDTH),
                dtype=np.uint8,
            )
        else:  # rgb
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3),
                dtype=np.uint8,
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
        self._previous_sub_area = None  # Track sub-area for transition detection
        self._previous_x_global = None  # Track x position for transition detection
        self._previous_y_global = None  # Track y position for transition detection
        self._transition_frame_count = 0  # Count frames since transition detected

    def _init_rendering(self) -> None:
        """Initialize pygame rendering when first needed."""
        if self._pygame_initialized or self.render_mode != 'human':
            return

        # Lazy load this, we don't need for non rendered envs
        import pygame
        pygame.init()

        from .app.info_display import get_required_info_height
        from .constants import (
            DEFAULT_SCALE,
            FONT_SIZE_BASE,
            SCREEN_HEIGHT,
            SCREEN_WIDTH,
        )

        self._scale = DEFAULT_SCALE
        self._width = SCREEN_WIDTH * self._scale
        self._height = SCREEN_HEIGHT * self._scale
        self._info_height = get_required_info_height(self._scale)

        self._screen = pygame.display.set_mode((self._width, self._height + self._info_height))
        pygame.display.set_caption("Super Mario Bros 2")
        self._clock = pygame.time.Clock() if self.render_fps is not None else None

        # Setup font for info display
        self._font_size = FONT_SIZE_BASE * self._scale // 2
        self._font = pygame.font.Font(None, self._font_size)
        self._pygame_initialized = True

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
        self._transition_frame_count = 0

        save_path = self.init_config.get_save_state_path()

        if save_path and not os.path.exists(save_path):
            raise FileNotFoundError(f"Save state file not found: {save_path}")

        if save_path:
            self.load_state_from_path(save_path)
        else:
            # When no save state, navigate to character selection screen
            # Wait for title screen to appear
            for _ in range(120):  # 2 seconds
                self._nes.step([False] * 8, render=False)

            # Press START to get past title screen
            start_button = [False, False, False, True, False, False, False, False]  # START button
            for _ in range(10):  # Press START
                self._nes.step(start_button, render=False)
            for _ in range(10):  # Release
                self._nes.step([False] * 8, render=False)

            # Wait for transition to character select screen
            for _ in range(120):  # 2 seconds
                self._nes.step([False] * 8, render=False)

            # Stop here - let the user select their character manually

        # Get one frame after reset/loading save state
        obs, _, _, _, _ = self._nes.step([False] * 8, render=True)

        info = self.info

        # Initialize tracking for detecting life loss and level completion
        self._previous_lives = self.lives
        self._previous_levels_finished = self.levels_finished.copy()

        # Initialize tracking with consistent global coordinates
        global_coords = self.global_coordinate_system
        self._previous_sub_area = global_coords.sub_area
        self._previous_x_global = global_coords.global_x
        self._previous_y_global = global_coords.global_y
        if self.render_mode == 'human':
            self.render(obs)

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

        # Update tracking for next step
        level_completed = self.level_completed
        self._previous_lives = self.lives
        self._previous_levels_finished = self.levels_finished.copy()

        # Track global coords
        global_coords = self.global_coordinate_system
        self._previous_sub_area = global_coords.sub_area
        self._previous_x_global = global_coords.global_x
        self._previous_y_global = global_coords.global_y

        # 4. Check termination
        terminated = self.is_game_over or life_lost  # or level_completed
        truncated = (
            self.max_episode_steps is not None and self._episode_steps >= self.max_episode_steps
        )

        self._done = terminated or truncated
        reward = 0.0  # Always return 0 reward

        # Render if in human mode
        if self.render_mode == 'human':
            self.render(obs)

        return np.array(obs), reward, terminated, truncated, info

    def render(self, obs: np.ndarray) -> Optional[np.ndarray]:
        """Render the environment.

        Args:
            obs: Observation array to render.

        Returns:
            RGB array for display, None if no render mode
        """
        if self.render_mode == 'human':
            # Lazy load
            if not self._pygame_initialized:
                self._init_rendering()

            if self._screen is not None:
                import pygame

                # Handle pygame events to prevent window freezing
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return obs

                # Render
                render_frame(self._screen, obs, self._width, self._height)
                create_info_panel(self._screen, self.info, self._font, self._height, self._width)

                pygame.display.flip()
                # Only limit FPS if render_fps is specified
                if self._clock is not None and self.render_fps is not None:
                    self._clock.tick(self.render_fps)

            return obs
        return None

    def _get_y_position(self, address: int) -> int:
        """Safely read Y position from RAM and clamp to valid range.

        Args:
            address: RAM address to read Y position from

        Returns:
            Y position clamped to 0-239 range
        """
        y_pos = self._read_ram_safe(address, default=0)
        return max(0, min(y_pos, SCREEN_HEIGHT - 1))

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

    def _read_ppu(self, address: int, default: int = 0) -> int:
        """Read from PPU memory.

        Args:
            address: PPU address to read (e.g., 0x2000-0x23FF for nametable)
            default: Default value if PPU reading is not available

        Returns:
            Value at PPU address or default
        """
        if hasattr(self._nes, 'read_ppu'):
            return self._nes.read_ppu(address)
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
            'area': self.area,
            'sub_area': self.sub_area,
            'spawn_page': self.spawn_page,
            'current_page_position': self.current_page_position,
            'total_pages_in_sub_area': self.total_pages_in_sub_area,
            'is_vertical_area': self.is_vertical_area,
            'global_coordinates': self.global_coordinate_system,
            'character': self.character,
            'hearts': self.hearts,
            'cherries': self.cherries,
            'coins': self.coins,
            'starman_timer': self.starman_timer,
            'subspace_timer': self.subspace_timer,
            'stopwatch_timer': self.stopwatch_timer,
            'invulnerability_timer': self.invulnerability_timer,
            'framerule_timer': self.framerule_timer,
            'unknown_timer': self.unknown_timer,
            'bob_omb_5_timer': self.bob_omb_5_timer,
            'bob_omb_4_timer': self.bob_omb_4_timer,
            'bob_omb_3_timer': self.bob_omb_3_timer,
            'bob_omb_2_timer': self.bob_omb_2_timer,
            'bob_omb_1_timer': self.bob_omb_1_timer,
            'bomb_1_timer': self.bomb_1_timer,
            'bomb_2_timer': self.bomb_2_timer,
            'pidget_carpet_timer': self.pidget_carpet_timer,
            'holding_item': self.holding_item,
            'item_pulled': self.item_pulled,
            'continues': self.continues,
            'player_speed': self.player_speed,
            'on_vine': self.on_vine,
            'float_timer': self.float_timer,
            'levels_finished': self.levels_finished,
            'vegetables_pulled': self.vegetables_pulled,
            'subspace_status': self.subspace_status,
            'level_transition': self.level_transition,
            'level_completed': self.level_completed,
            'door_transition_timer': self.door_transition_timer,
            'enemies_defeated': self.enemies_defeated,
            'enemy_x_positions': self.enemy_x_positions,
            'enemy_y_positions': self.enemy_y_positions,
            'enemy_speeds': self.enemy_speeds,
            'enemy_visibility_states': self.enemy_visibility_states,
            'enemy_x_positions_global': self.enemy_x_positions_global,
            'enemy_y_positions_global': self.enemy_y_positions_global,
            'enemy_x_pages': self.enemy_x_pages,
            'enemy_y_pages': self.enemy_y_pages,
            'enemy_x_positions_relative': self.enemy_x_positions_relative,
            'enemy_y_positions_relative': self.enemy_y_positions_relative,
            'enemy_hp': self.enemy_hp,
            'character_stats': self.character_stats,
            'player_state': self.player_state,
            'collision_map': self.collision_map,
            'interaction_matrix': self.interaction_matrix,
            'can_climb': self.can_climb,
            'is_on_special_surface': self.is_on_special_surface,
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
    def character_stats(self):
        """Get current character's statistics and abilities."""
        return get_character_stats(self.character)

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

    # ---- Other bindings --------------------------------------------

    def get_action_meanings(self) -> list[list[str]]:
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

    def set_frame_speed(self, speed: float) -> None:
        """Set the frame speed for faster/slower emulation.

        Args:
            speed: Frame speed multiplier (1.0 = normal speed, 2.0 = 2x speed, etc.)
                   Must be positive.

        Raises:
            ValueError: If speed is not positive
        """
        if speed <= 0.0:
            raise ValueError("Frame speed must be positive")
        self._nes.set_frame_speed(speed)

    def get_frame_speed(self) -> float:
        """Get the current frame speed multiplier.

        Returns:
            Current frame speed (1.0 = normal speed)
        """
        return self._nes.get_frame_speed()

    def close(self) -> None:
        """Close the environment and clean up resources."""
        if hasattr(self, '_pygame_initialized') and self._pygame_initialized:
            import pygame
            pygame.quit()
            self._screen = None
            self._pygame_initialized = False
