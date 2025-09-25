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
    AREA,
    CHARACTER,
    CHARACTER_NAMES,
    CHERRIES,
    CONTINUES,
    CURRENT_LEVEL,
    CURRENT_PAGE_POSITION,
    DOOR_TRANSITION_TIMER,
    ENEMIES_DEFEATED,
    ENEMY_DEAD,
    ENEMY_HEALTH,
    ENEMY_ID,
    ENEMY_INVISIBLE,
    ENEMY_NOT_PRESENT,
    ENEMY_SPEED,
    ENEMY_VISIBILITY,
    ENEMY_VISIBLE,
    ENEMY_X_PAGE,
    ENEMY_X_POS,
    ENEMY_Y_PAGE,
    ENEMY_Y_POS,
    FLOAT_TIMER,
    GAME_INIT_FRAMES,
    INVULNERABILITY_TIMER,
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
    PAGE,
    PAGE_SIZE,
    PLAYER_SPEED,
    PLAYER_STATE,
    PLAYER_X_PAGE,
    PLAYER_X_POSITION,
    PLAYER_Y_PAGE,
    PLAYER_Y_POSITION,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SCROLL_DIRECTION,
    STARMAN_TIMER,
    STOPWATCH_TIMER,
    SUB_AREA,
    SUBSPACE_COINS,
    SUBSPACE_STATUS,
    SUBSPACE_TIMER,
    TOTAL_PAGES_IN_SUB_AREA,
    VEGETABLES_PULLED,
    WORLD_NUMBER,
    GlobalCoordinate,
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
        terminated = self.is_game_over or life_lost or level_completed
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
        """Get player local X position (on current page)."""
        x_pos = self._read_ram_safe(PLAYER_X_POSITION, default=0)
        return x_pos

    @property
    def x_page(self) -> int:
        """Get the X page of the player position."""
        x_page = self._read_ram_safe(PLAYER_X_PAGE)
        return x_page

    def _transform_y_coordinate(self, y_page: int, y_pos_raw: int) -> int:
        """Transform raw Y coordinates to inverted system (y=0 at bottom, increasing upward).

        Args:
            y_page: Y page value from RAM
            y_pos: Y position value from RAM

        Returns:
            Inverted Y coordinate
        """
        # Handle wraparound: when goes above top, y_page becomes 255
        if y_page == 255:
            y_page = 0

        y_pos_global: int = y_page * SCREEN_HEIGHT + y_pos_raw

        if self.is_vertical_area:
            max_y_in_level = self.total_pages_in_sub_area * SCREEN_HEIGHT
        else:
            max_y_in_level = SCREEN_HEIGHT

        return max_y_in_level - y_pos_global - 1

    @property
    def y_position_global(self) -> int:
        """Get player global Y position (with y=0 at bottom, increasing upward)."""
        y_page = self._read_ram_safe(PLAYER_Y_PAGE, default=0)
        y_pos_raw = self._get_y_position(PLAYER_Y_POSITION)
        return self._transform_y_coordinate(y_page, y_pos_raw)

    @property
    def y_position(self) -> int:
        """Get player local Y position (with y=0 at bottom, increasing upward)."""
        y_pos = self._get_y_position(PLAYER_Y_POSITION)
        # Invert the y-coordinate within the screen space
        return SCREEN_HEIGHT - 1 - y_pos

    @property
    def y_page(self) -> int:
        """Get the Y page of the player position."""
        y_page = self._read_ram_safe(PLAYER_Y_PAGE)
        if y_page == 255:  # Screen wrap around
            return 0
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
    def area(self) -> int:
        """Get current area."""
        area = self._read_ram_safe(AREA, default=0)
        return area

    @property
    def sub_area(self) -> int:
        """Get current sub-area."""
        sub_area = self._read_ram_safe(SUB_AREA, default=0)
        return sub_area

    @property
    def spawn_page(self) -> int:
        """Get current spawn page/entry point."""
        page = self._read_ram_safe(PAGE, default=0)
        return page

    @property
    def current_page_position(self) -> int:
        """Get current page position in sub-area."""
        page_pos = self._read_ram_safe(CURRENT_PAGE_POSITION, default=0)
        return page_pos

    @property
    def total_pages_in_sub_area(self) -> int:
        """Get total number of pages in the current sub-area."""
        total_pages = self._read_ram_safe(TOTAL_PAGES_IN_SUB_AREA, default=0)
        return total_pages + 1  # zero indexed

    @property
    def is_vertical_area(self) -> bool:
        """Check if current area has vertical scrolling."""
        direction = self._read_ram_safe(SCROLL_DIRECTION, default=0)
        return not bool(direction)

    @property
    def global_coordinate_system(self) -> GlobalCoordinate:
        """
        Get global coordinate system combining level structure with player
        position.

        Returns a 4-tuple coordinate system: (Area, Sub-area, Global_X, Global_Y)

        This provides a unified positioning system that combines:
        - Level structure: Area, Sub-area (from memory addresses $04E7-$04E8)
        - Player position: Global X and Y coordinates in the game world

        Note: During door transitions, SMB2 updates sub_area before updating
        player coordinates. This method waits AREA_TRANSITION_FRAMES after detectin25
        transition before accepting new coordinates to ensure they've fully updated.

        Returns:
            GlobalCoordinate: NamedTuple with area, sub_area, global_x, global_y
        """
        current_sub_area = self.sub_area
        current_x = self.x_position_global
        current_y = self.y_position_global

        # Check if we're in a transition state where sub_area changed but coordinates haven't
        if (self._previous_sub_area is not None and \
            self._previous_x_global is not None and
            self._previous_y_global is not None):

            # Detect new transition
            if (self._transition_frame_count == 0 and \
                current_sub_area != self._previous_sub_area and \
                current_x == self._previous_x_global and
                current_y == self._previous_y_global):
                self._transition_frame_count = 1
                current_sub_area = self._previous_sub_area

            # Detect transition period
            elif self._transition_frame_count > 0:
                self._transition_frame_count += 1
                if self._transition_frame_count <= self.AREA_TRANSITION_FRAMES:
                    current_sub_area = self._previous_sub_area
                    current_x = self._previous_x_global
                    current_y = self._previous_y_global
                elif self._transition_frame_count == self.AREA_TRANSITION_FRAMES + 1:
                    self._transition_frame_count = 0  # Reset counter

        return GlobalCoordinate(
            area=self.area,
            sub_area=current_sub_area,
            global_x=current_x,
            global_y=current_y,
        )

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
            # If value doesn't match expected pattern, use upper nibble + 1
            hearts = ((life_meter & 0xF0) >> 4) + 1
            if 1 <= hearts <= MAX_HEARTS:
                return hearts
            return 2  # Default

    @property
    def cherries(self) -> int:
        """Get cherries collected."""
        cherries = self._read_ram_safe(CHERRIES, default=0)
        if 0 <= cherries <= MAX_CHERRIES:
            return cherries
        return 0

    @property
    def coins(self) -> int:
        """Get coins collected in Subspace."""
        coins = self._read_ram_safe(SUBSPACE_COINS, default=0)
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
    def invulnerability_timer(self) -> int:
        """Get invulnerability timer (time left until character becomes vulnerable)."""
        return self._read_ram_safe(INVULNERABILITY_TIMER, default=0)

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
        """Get player horizontal speed (signed: positive=right, negative=left)."""
        speed = self._read_ram_safe(PLAYER_SPEED, default=0)
        return speed if speed < 128 else speed - 256

    @property
    def on_vine(self) -> bool:
        """Check if character is on a vine."""
        return self._read_ram_safe(ON_VINE, default=0) == 1

    @property
    def float_timer(self) -> int:
        """Get Princess float timer (available float time, max 60 frames = 1 second)."""
        return self._read_ram_safe(FLOAT_TIMER, default=0)

    @property
    def levels_finished(self) -> dict[str, int]:
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

    @property
    def level_transition(self) -> int:
        """Get level transition state.

        NOTE: This value at 0x04EC appears to change for less than a frame.
        The game sets it to non-zero and immediately clears it back to 0
        within the same frame's CPU execution (as seen in disassembly at
        $E66D: STA $04EC). Therefore, we cannot reliably detect transitions
        by polling this value once per frame.
        For reliable level completion detection, use the increase in
        'levels_finished' counter instead.

        Values (theoretical):
        0 - normal gameplay
        1 - restart same level
        2 - game over
        3 - end level, go to bonus game (level completed)
        4 - warp
        """
        # This will almost always return 0 due to sub-frame clearing? TODO: Delete
        return self._read_ram_safe(LEVEL_TRANSITION, default=0)

    @property
    def level_completed(self) -> bool:
        """Detect if a level was just completed.

        Returns True if any character's levels_finished counter has increased
        since the last step.
        """
        if not hasattr(self, '_previous_levels_finished'):
            return False

        current_levels_finished = self.levels_finished
        for char_name in ['mario', 'peach', 'toad', 'luigi']:
            if current_levels_finished[char_name] > self._previous_levels_finished.get(
                char_name, 0
            ):
                return True
        return False

    @property
    def enemy_visibility_states(self) -> list[int]:
        """Get visibility states of enemies 1-5.

        Returns:
            List of 5 enemy visibility states (index 0 = enemy 5, index 4 = enemy 1)
            0 = Invisible, 1 = Visible, 2 = Dead
        """
        return [self._read_ram_safe(addr, default=0) for addr in ENEMY_VISIBILITY]

    @property
    def enemy_x_positions(self) -> list[int]:
        """Get X positions of enemies 1-5 on the current page.

        Returns:
            List of 5 enemy X positions (index 0 = enemy 5, index 4 = enemy 1)
            Returns ENEMY_NOT_PRESENT for invisible or dead enemies
        """
        positions = []
        visibility_states = self.enemy_visibility_states
        for i, addr in enumerate(ENEMY_X_POS):
            if visibility_states[i] in [ENEMY_INVISIBLE, ENEMY_DEAD]:
                positions.append(ENEMY_NOT_PRESENT)
            else:
                positions.append(self._read_ram_safe(addr, default=0))
        return positions

    @property
    def enemy_y_positions(self) -> list[int]:
        """Get Y positions of enemies 1-5 on the current page (with y=0 at bottom).

        Returns:
            List of 5 enemy Y positions (index 0 = enemy 5, index 4 = enemy 1)
            Returns ENEMY_NOT_PRESENT for invisible or dead enemies
        """
        positions = []
        visibility_states = self.enemy_visibility_states
        for i, addr in enumerate(ENEMY_Y_POS):
            if visibility_states[i] in [ENEMY_INVISIBLE, ENEMY_DEAD]:
                positions.append(ENEMY_NOT_PRESENT)
            else:
                y_pos = self._get_y_position(addr)
                # Invert the y-coordinate within the screen space
                positions.append(SCREEN_HEIGHT - 1 - y_pos)
        return positions

    @property
    def enemy_speeds(self) -> list[int]:
        """Get horizontal speeds of enemies 1-5 (signed: positive=right, negative=left).

        Returns:
            List of 5 enemy speeds (index 0 = enemy 5, index 4 = enemy 1)
            Returns ENEMY_NOT_PRESENT for invisible or dead enemies
        """
        speeds = []
        visibility_states = self.enemy_visibility_states
        for i, addr in enumerate(ENEMY_SPEED):
            if visibility_states[i] in [ENEMY_INVISIBLE, ENEMY_DEAD]:
                speeds.append(ENEMY_NOT_PRESENT)
            else:
                speed = self._read_ram_safe(addr, default=0)
                signed_speed = speed if speed < 128 else speed - 256
                speeds.append(signed_speed)
        return speeds

    @property
    def enemy_x_pages(self) -> list[int]:
        """Get X pages of enemies 1-5.

        Returns:
            List of 5 enemy X pages (index 0 = enemy 5, index 4 = enemy 1)
            Returns ENEMY_NOT_PRESENT for invisible or dead enemies
        """
        pages = []
        visibility_states = self.enemy_visibility_states
        for i, addr in enumerate(ENEMY_X_PAGE):
            if visibility_states[i] in [ENEMY_INVISIBLE, ENEMY_DEAD]:
                pages.append(ENEMY_NOT_PRESENT)
            else:
                pages.append(self._read_ram_safe(addr, default=0))
        return pages

    @property
    def enemy_y_pages(self) -> list[int]:
        """Get Y pages of enemies 1-5.

        Returns:
            List of 5 enemy Y pages (index 0 = enemy 5, index 4 = enemy 1)
            Returns ENEMY_NOT_PRESENT for invisible or dead enemies
        """
        pages = []
        visibility_states = self.enemy_visibility_states
        for i, addr in enumerate(ENEMY_Y_PAGE):
            if visibility_states[i] in [ENEMY_INVISIBLE, ENEMY_DEAD]:
                pages.append(ENEMY_NOT_PRESENT)
            else:
                pages.append(self._read_ram_safe(addr, default=0))
        return pages

    @property
    def enemy_x_positions_global(self) -> list[int]:
        """Get global X positions of enemies 1-5.

        Returns:
            List of 5 enemy global X positions (index 0 = enemy 5, index 4 = enemy 1)
            Returns ENEMY_NOT_PRESENT for invisible or dead enemies
        """
        global_positions = []
        visibility_states = self.enemy_visibility_states

        for i, (x_pos_addr, x_page_addr) in enumerate(zip(ENEMY_X_POS, ENEMY_X_PAGE)):
            if visibility_states[i] in [ENEMY_INVISIBLE, ENEMY_DEAD]:
                global_positions.append(ENEMY_NOT_PRESENT)
            else:
                x_page = self._read_ram_safe(x_page_addr, default=0)
                x_pos = self._read_ram_safe(x_pos_addr, default=0)
                global_x = (x_page * PAGE_SIZE) + x_pos
                global_positions.append(global_x)
        return global_positions

    @property
    def enemy_y_positions_global(self) -> list[int]:
        """Get global Y positions of enemies 1-5 (with y=0 at bottom, increasing upward).

        Returns:
            List of 5 enemy global Y positions (index 0 = enemy 5, index 4 = enemy 1)
            Returns ENEMY_NOT_PRESENT for invisible or dead enemies
        """
        global_positions = []
        visibility_states = self.enemy_visibility_states

        for i, (y_pos_addr, y_page_addr) in enumerate(zip(ENEMY_Y_POS, ENEMY_Y_PAGE)):
            if visibility_states[i] in [ENEMY_INVISIBLE, ENEMY_DEAD]:
                global_positions.append(ENEMY_NOT_PRESENT)
            else:
                y_page = self._read_ram_safe(y_page_addr, default=0)
                y_pos_raw = self._get_y_position(y_pos_addr)
                inverted_y = self._transform_y_coordinate(y_page, y_pos_raw)
                global_positions.append(inverted_y)
        return global_positions

    @property
    def enemy_x_positions_relative(self) -> list[int]:
        """Get X positions of enemies 1-5 relative to player using global coordinates.

        Returns:
            List of 5 enemy X positions relative to player (index 0 = enemy 5, index 4 = enemy 1)
            Returns ENEMY_NOT_PRESENT for invisible or dead enemies
        """
        relative_positions = []
        visibility_states = self.enemy_visibility_states
        player_x_global = self.x_position_global
        enemy_x_global = self.enemy_x_positions_global

        for i in range(len(visibility_states)):
            if visibility_states[i] in [ENEMY_INVISIBLE, ENEMY_DEAD]:
                relative_positions.append(ENEMY_NOT_PRESENT)
            else:
                if enemy_x_global[i] != ENEMY_NOT_PRESENT:
                    relative_positions.append(player_x_global - enemy_x_global[i])
                else:
                    relative_positions.append(ENEMY_NOT_PRESENT)
        return relative_positions

    @property
    def enemy_y_positions_relative(self) -> list[int]:
        """Get Y positions of enemies 1-5 relative to player using global coordinates.

        With inverted Y coordinates (y=0 at bottom, increasing upward):
        - Positive values = enemy is above player (enemy has higher Y)
        - Negative values = enemy is below player (enemy has lower Y)

        Returns:
            List of 5 enemy Y positions relative to player (index 0 = enemy 5, index 4 = enemy 1)
            Returns ENEMY_NOT_PRESENT for invisible or dead enemies
        """
        relative_positions = []
        visibility_states = self.enemy_visibility_states
        player_y_global = self.y_position_global
        enemy_y_global = self.enemy_y_positions_global

        for i in range(len(visibility_states)):
            if visibility_states[i] in [ENEMY_INVISIBLE, ENEMY_DEAD]:
                relative_positions.append(ENEMY_NOT_PRESENT)
            else:
                if enemy_y_global[i] != ENEMY_NOT_PRESENT:
                    relative_positions.append(player_y_global - enemy_y_global[i])
                else:
                    relative_positions.append(ENEMY_NOT_PRESENT)
        return relative_positions

    @property
    def enemy_hp(self) -> list[int]:
        """Get HP values of enemies 1-5.

        Returns:
            List of 5 enemy HP values (index 0 = enemy 5, index 4 = enemy 1)
            Returns ENEMY_NOT_PRESENT for invisible or dead enemies
        """
        hp_values = []
        visibility_states = self.enemy_visibility_states
        for i, addr in enumerate(ENEMY_HEALTH):
            if visibility_states[i] in [ENEMY_INVISIBLE, ENEMY_DEAD]:
                hp_values.append(ENEMY_NOT_PRESENT)
            else:
                hp_values.append(self._read_ram_safe(addr, default=0))
        return hp_values

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
