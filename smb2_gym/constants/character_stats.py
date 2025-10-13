"""Character statistics and abilities for SMB2 characters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CharacterStats:
    """Character statistics and abilities."""

    # Pick-up speeds for 6 frames of pulling animation
    pickup_speeds: tuple[int, ...]

    # Jump speeds under different conditions
    jump_speed_still_no_object: int
    jump_speed_still_with_object: int
    jump_speed_charged_no_object: int
    jump_speed_charged_with_object: int
    jump_speed_running_no_object: int
    jump_speed_running_with_object: int
    jump_speed_quicksand: int

    # Floating ability (Princess only)
    floating_time: int

    # Gravity values
    gravity_without_jump: int
    gravity_with_jump: int
    gravity_quicksand: int

    # Running speeds
    running_speed_right_no_object: int
    running_speed_right_with_object: int
    running_speed_right_quicksand: int
    running_speed_left_no_object: int
    running_speed_left_with_object: int
    running_speed_left_quicksand: int


# ------------------------------------------------------------------------------
# ---- Character Instances -----------------------------------------------------
# ------------------------------------------------------------------------------

MARIO_STATS = CharacterStats(
    pickup_speeds=(0, 4, 2, 1, 4, 7),
    jump_speed_still_no_object=176,
    jump_speed_still_with_object=176,
    jump_speed_charged_no_object=152,
    jump_speed_charged_with_object=152,
    jump_speed_running_no_object=166,
    jump_speed_running_with_object=170,
    jump_speed_quicksand=224,
    floating_time=0,
    gravity_without_jump=7,
    gravity_with_jump=4,
    gravity_quicksand=8,
    running_speed_right_no_object=24,
    running_speed_right_with_object=24,
    running_speed_right_quicksand=4,
    running_speed_left_no_object=-24,  # 0xE8 as signed = -24
    running_speed_left_with_object=-24,  # 0xE8 as signed = -24
    running_speed_left_quicksand=-4,  # 0xFC as signed = -4
)

TOAD_STATS = CharacterStats(
    pickup_speeds=(0, 1, 1, 1, 1, 2),
    jump_speed_still_no_object=178,
    jump_speed_still_with_object=178,
    jump_speed_charged_no_object=152,
    jump_speed_charged_with_object=152,
    jump_speed_running_no_object=173,
    jump_speed_running_with_object=173,
    jump_speed_quicksand=224,
    floating_time=0,
    gravity_without_jump=7,
    gravity_with_jump=4,
    gravity_quicksand=8,
    running_speed_right_no_object=24,
    running_speed_right_with_object=29,
    running_speed_right_quicksand=4,
    running_speed_left_no_object=-24,  # 0xE8 as signed = -24
    running_speed_left_with_object=-29,  # 0xE3 as signed = -29
    running_speed_left_quicksand=-4,  # 0xFC as signed = -4
)

LUIGI_STATS = CharacterStats(
    pickup_speeds=(0, 4, 2, 1, 4, 7),
    jump_speed_still_no_object=214,
    jump_speed_still_with_object=214,
    jump_speed_charged_no_object=201,
    jump_speed_charged_with_object=201,
    jump_speed_running_no_object=208,
    jump_speed_running_with_object=212,
    jump_speed_quicksand=224,
    floating_time=0,
    gravity_without_jump=2,
    gravity_with_jump=1,
    gravity_quicksand=8,
    running_speed_right_no_object=24,
    running_speed_right_with_object=22,
    running_speed_right_quicksand=4,
    running_speed_left_no_object=-24,  # 0xE8 as signed = -24
    running_speed_left_with_object=-22,  # 0xEA as signed = -22
    running_speed_left_quicksand=-4,  # 0xFC as signed = -4
)

PRINCESS_STATS = CharacterStats(
    pickup_speeds=(0, 6, 4, 2, 6, 12),
    jump_speed_still_no_object=179,
    jump_speed_still_with_object=179,
    jump_speed_charged_no_object=152,
    jump_speed_charged_with_object=152,
    jump_speed_running_no_object=172,
    jump_speed_running_with_object=179,
    jump_speed_quicksand=224,
    floating_time=60,  # 60 frames = 1 second
    gravity_without_jump=7,
    gravity_with_jump=4,
    gravity_quicksand=8,
    running_speed_right_no_object=24,
    running_speed_right_with_object=21,
    running_speed_right_quicksand=4,
    running_speed_left_no_object=-24,  # 0xE8 as signed = -24
    running_speed_left_with_object=-21,  # 0xEB as signed = -21
    running_speed_left_quicksand=-4,  # 0xFC as signed = -4
)

CHARACTER_STATS: dict[int, CharacterStats] = {
    0: MARIO_STATS,
    1: PRINCESS_STATS,
    2: TOAD_STATS,
    3: LUIGI_STATS,
}


def get_character_stats(character_id: int) -> CharacterStats:
    """Get character statistics by character ID.

    Args:
        character_id: Character ID (0=Mario, 1=Princess, 2=Toad, 3=Luigi)

    Returns:
        CharacterStats instance for the specified character

    Raises:
        ValueError: If character_id is not valid
    """
    if character_id not in CHARACTER_STATS:
        raise ValueError(f"Invalid character_id {character_id}. Must be 0-3.")
    return CHARACTER_STATS[character_id]

