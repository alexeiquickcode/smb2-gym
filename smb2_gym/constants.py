"""
RAM addresses and constants for Super Mario Bros 2 (Europe).

Detailed RAM map available at: 

https://datacrystal.tcrf.net/wiki/Super_Mario_Bros._2_(NES)/RAM_map
"""

# Display/Rendering constants
SCREEN_WIDTH = 256
SCREEN_HEIGHT = 240
DEFAULT_SCALE = 3
FPS = 60
FONT_SIZE_BASE = 18
WINDOW_CAPTION = "Super Mario Bros 2"

# Button indices for NES controller
BUTTON_A = 0
BUTTON_B = 1
BUTTON_SELECT = 2
BUTTON_START = 3
BUTTON_UP = 4
BUTTON_DOWN = 5
BUTTON_LEFT = 6
BUTTON_RIGHT = 7

# Action types
ACTION_TYPE_SIMPLE = "simple"
ACTION_TYPE_COMPLEX = "complex"
ACTION_TYPE_ALL = "all"

# Game limits
MAX_CHERRIES = 20
MAX_COINS = 99
MAX_CONTINUES = 9
MAX_LIVES = 9
MAX_HEARTS = 4

# Game mechanics
PAGE_SIZE = 256  # Memory page size for position calculations
Y_POSITION_WRAPAROUND_THRESHOLD = 10000  # Y positions above this are wraparound
GAME_INIT_FRAMES = 300  # Frames to wait for game initialization

# Save state slots
MAX_SAVE_SLOTS = 10  # 0-9

# Player state
PLAYER_X_PAGE = 0x0014  # Page of main character's X position
PLAYER_X_POSITION = 0x0028  # Main character's X position on page
PLAYER_Y_PAGE = 0x001E  # Page of main character's Y position
PLAYER_Y_POSITION = 0x0032  # Y position on page
PLAYER_STATE = 0x0050  # Player state/animation

# Game state
LIVES = 0x04ED
LIFE_METER = 0x04C2  # 0F=1 heart, 1F=2 hearts, 2F=3 hearts, 3F=4 hearts
CHARACTER = 0x008F  # 0=Mario, 1=Peach, 2=Toad, 3=Luigi
CURRENT_LEVEL = 0x0531  # 00-13 (levels 1-1 to 7-2)
WORLD_NUMBER = 0x0635
LEVEL_TILESET = 0x06F7

# Collectibles
CHERRIES = 0x062A
SUBSPACE_COINS = 0x062B
VEGETABLES_PULLED = 0x062C

# Power-ups and status
STARMAN_TIMER = 0x04E0
SUBSPACE_TIMER = 0x04B7
STOPWATCH_TIMER = 0x04FF
SUBSPACE_STATUS = 0x0628  # 00=no, 02=yes
FLOAT_TIMER = 0x0553  # Peach floating ability
DOOR_TRANSITION_TIMER = 0x04BD  # Time counts up for how long the door takes to open
LEVEL_TRANSITION = 0x04EC  # Level transition state

# Items and inventory
ITEM_HOLDING = 0x009C  # 00=no item, 01=holding item
ITEM_PULLED = 0x0096  # Item pulled from ground
CONTINUES = 0x05C5  # Number of continues

# Character movement and status
PLAYER_SPEED = 0x003C  # Horizontal speed
ON_VINE = 0x0050  # 00=no, 01=on vine

# Level completion tracking (per character)
LEVELS_FINISHED_MARIO = 0x062D
LEVELS_FINISHED_PEACH = 0x062E
LEVELS_FINISHED_TOAD = 0x062F
LEVELS_FINISHED_LUIGI = 0x0630

# Enemies (first 5 slots)
ENEMY_X_POS = [0x0029, 0x002A, 0x002B, 0x002C, 0x002D]
ENEMY_Y_POS = [0x0033, 0x0034, 0x0035, 0x0036, 0x0037]
ENEMY_ID = [0x0090, 0x0091, 0x0092, 0x0093, 0x0094]
ENEMY_HEALTH = [0x0465, 0x0466, 0x0467, 0x0468, 0x0469]
ENEMIES_DEFEATED = 0x04AD  # Count of enemies defeated (for heart spawning)

# Input
CONTROLLER_1_PRESSED = 0x00F5
CONTROLLER_1_HELD = 0x00F7

# Character names for display
CHARACTER_NAMES = {0: "Mario", 1: "Peach", 2: "Toad", 3: "Luigi"}

# Level names
LEVEL_NAMES = {
    0x00: "1-1",
    0x01: "1-2",
    0x02: "1-3",
    0x03: "2-1",
    0x04: "2-2",
    0x05: "2-3",
    0x06: "3-1",
    0x07: "3-2",
    0x08: "3-3",
    0x09: "4-1",
    0x0A: "4-2",
    0x0B: "4-3",
    0x0C: "5-1",
    0x0D: "5-2",
    0x0E: "5-3",
    0x0F: "6-1",
    0x10: "6-2",
    0x11: "6-3",
    0x12: "7-1",
    0x13: "7-2"
}
