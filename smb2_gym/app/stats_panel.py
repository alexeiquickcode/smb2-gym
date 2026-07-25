"""Tabbed stats panel for the human-play interface.

The old panel dumped 27 rows at once regardless of window size. This groups the
same data into tabs — OVERVIEW, PLAYER, CHARACTER, ENEMIES — and reflows each
into as many columns as the current width allows.

CHARACTER is the odd one out: its numbers come from the game's static per-
character tables rather than RAM, so they never change during a run.
"""

from typing import (
    Any,
)

import pygame

from ..constants import CHARACTER_NAMES
from ..constants.character_stats import (
    CHARACTER_STATS,
    get_character_stats,
)
from ..constants.object_ids import PlayerState
from .info_display import (
    format_collision_flags,
    format_enemy_name,
    format_enemy_state,
    format_sprite_flags,
)
from .theme import (
    ACCENT,
    BAD,
    BORDER,
    GOOD,
    PANEL_ALT,
    SECTION_ACCENTS,
    TEXT,
    TEXT_DIM,
    TEXT_FAINT,
    WARN,
    Fonts,
    Metrics,
    blit_clipped,
    draw_card,
    truncate,
)


TABS = ["OVERVIEW", "PLAYER", "CHARACTER", "ENEMIES"]


# ------------------------------------------------------------------------------
# ---- Formatters --------------------------------------------------------------
# ------------------------------------------------------------------------------


def _yes_no(value: Any) -> str:
    return "Yes" if value else "No"


def _player_state_name(state: int) -> str:
    try:
        return PlayerState(state).name
    except ValueError:
        return str(state)


# ------------------------------------------------------------------------------
# ---- Data assembly -----------------------------------------------------------
# ------------------------------------------------------------------------------

# Each group is (title, [(label, value, emphasis)]) where emphasis picks a colour
Group = tuple[str, list[tuple[str, str, str | None]]]


def _overview_groups(info: dict[str, Any]) -> list[Group]:
    pc, pos, game = info['pc'], info['pos'], info['game']
    return [
        (
            "POSITION",
            [
                ("World / Level", f"{game.world} · {game.level}", None),
                ("Area", f"{pos.area}-{pos.sub_area}", None),
                ("Local X,Y", f"{pos.x_local}, {pos.y_local}", None),
                ("Global X,Y", f"{pos.x_global}, {pos.y_global}", None),
                ("Page X,Y", f"{pos.x_page}, {pos.y_page}", None),
                ("Page", f"{pos.current_page}/{pos.total_pages}", None),
                ("Spawn Page", str(pos.spawn_page), None),
                ("Vertical", _yes_no(pos.is_vertical), None),
            ],
        ),
        (
            "PLAYER",
            [
                ("Character", CHARACTER_NAMES.get(pc.character, "Unknown"), "accent"),
                ("Lives", str(pc.lives), "bad" if pc.lives <= 1 else None),
                ("Hearts", f"{pc.hearts}/4", "bad" if pc.hearts <= 1 else "good"),
                ("Coins", str(pc.coins), None),
                ("Cherries", str(pc.cherries), None),
                ("Speed", str(pc.speed), None),
                ("State", _player_state_name(pc.state), None),
                ("Holding Item", _yes_no(pc.holding_item), None),
            ],
        ),
        (
            "PROGRESS",
            [
                (
                    "Level Complete",
                    _yes_no(pc.level_completed),
                    "good" if pc.level_completed else None,
                ),
                ("Subspace", str(pc.subspace_status), None),
                ("Mario", str(pc.levels_finished['mario']), None),
                ("Luigi", str(pc.levels_finished['luigi']), None),
                ("Peach", str(pc.levels_finished['peach']), None),
                ("Toad", str(pc.levels_finished['toad']), None),
            ],
        ),
    ]


def _player_groups(info: dict[str, Any]) -> list[Group]:
    pc = info['pc']
    return [
        (
            "PLAYER",
            [
                ("Character", CHARACTER_NAMES.get(pc.character, "Unknown"), "accent"),
                ("State", _player_state_name(pc.state), None),
                ("Lives", str(pc.lives), "bad" if pc.lives <= 1 else None),
                ("Hearts", f"{pc.hearts}/4", "bad" if pc.hearts <= 1 else "good"),
                ("Speed", str(pc.speed), None),
                ("On Vine", _yes_no(pc.on_vine), None),
            ],
        ),
        (
            "ITEMS",
            [
                ("Coins", str(pc.coins), None),
                ("Cherries", str(pc.cherries), None),
                ("Holding Item", _yes_no(pc.holding_item), None),
                ("Item Pulled", str(pc.item_pulled), None),
                ("Big Veggies", str(pc.big_vegetables_pulled), None),
            ],
        ),
        (
            "TIMERS",
            [
                ("Starman", str(pc.starman_timer), "good" if pc.starman_timer else None),
                ("Stopwatch", str(pc.stopwatch_timer), "good" if pc.stopwatch_timer else None),
                ("Subspace", str(pc.subspace_timer), None),
                # Live countdown over the static budget. Both are needed: the
                # timer alone reads 0 for everyone, so a character with no
                # float looks identical to Peach standing on the ground.
                (
                    "Float",
                    f"{pc.float_timer}/{pc.float_length}" if pc.float_length else "None",
                    "good" if pc.float_timer else None,
                ),
                (
                    "Carpet",
                    str(pc.pidgit_carpet_timer),
                    "good" if pc.pidgit_carpet_timer else None,
                ),
                (
                    "Invulnerable",
                    str(pc.invulnerability_timer),
                    "warn" if pc.invulnerability_timer else None,
                ),
                ("Door", str(pc.door_transition_timer), None),
            ],
        ),
    ]


def _signed(value: int) -> int:
    """Read a stat byte as a signed 8-bit value.

    Velocities are stored as two's-complement 8.8 fixed point with Y growing
    downwards, so a jump impulse is negative and the raw byte reads high.
    """
    return value - 256 if value > 127 else value


def _velocity(raw: int) -> str:
    """Format a velocity byte as its raw value and pixels per frame."""
    return f"{raw} ({_signed(raw) / 256:+.2f}px/f)"


def _jump_score(stats: Any) -> float:
    """Relative standing jump height, for ranking the characters against each other.

    Rising under constant deceleration covers v^2 / 2a. The absolute result is
    not in pixels - the game applies gravity on its own schedule and cuts the
    rise short when the button is released - so this is only ever compared
    between characters, never shown as a distance.

    It exists because the impulse alone is misleading: Luigi has the weakest of
    the four but the lowest gravity, and out-jumps everyone.
    """
    velocity = abs(_signed(stats.jump_speed_still_no_object))
    gravity = stats.gravity_with_jump or 1
    return (velocity * velocity) / (2 * gravity)


def _jump_reach_label(stats: Any, everyone: list[Any]) -> str:
    """Rank this character's jump height as a place out of the four.

    Deliberately a rank rather than a distance: the score behind it is only
    meaningful as a comparison (see `_jump_score`).
    """
    scores = sorted({round(_jump_score(s), 3) for s in everyone}, reverse=True)
    place = scores.index(round(_jump_score(stats), 3)) + 1
    names = {1: "Highest", 2: "2nd", 3: "3rd", 4: "Lowest"}
    return names.get(place, f"{place}th")


def _rank(value: float, others: list[float], higher_is_better: bool) -> str | None:
    """Emphasis for a stat compared against the other three characters."""
    if len(set(others)) <= 1:
        return None  # every character is identical, so nothing to rank
    best = max(others) if higher_is_better else min(others)
    worst = min(others) if higher_is_better else max(others)
    if value == best:
        return "good"
    if value == worst:
        return "bad"
    return None


def _character_groups(info: dict[str, Any]) -> list[Group]:
    """Static per-character stats, with each value ranked against the others.

    These come from the game's tables rather than RAM, so they never change
    during play. The raw byte is shown alongside a readable reading because the
    stored values are counter-intuitive on their own: a *lower* jump byte is a
    *stronger* jump, since it is a negative (upward) velocity.
    """
    pc = info['pc']
    stats = pc.stats
    everyone = [get_character_stats(i) for i in sorted(CHARACTER_STATS)]

    jump = stats.jump_speed_still_no_object
    run = stats.running_speed_right_no_object
    pull = stats.pickup_speeds[-1] if stats.pickup_speeds else 0

    # Rank on how high the jump actually goes, not on the impulse alone. Luigi
    # has the weakest impulse of the four but the lowest gravity, and ends up
    # the best jumper - ranking the raw byte would paint that as his weakness.
    jump_rank = _rank(_jump_score(stats), [_jump_score(s) for s in everyone], True)
    # Less gravity means more hang time.
    gravity_rank = _rank(stats.gravity_with_jump, [s.gravity_with_jump for s in everyone], False)
    # Fewer frames on the last pull step means a faster pick-up.
    pull_rank = _rank(pull, [s.pickup_speeds[-1] for s in everyone if s.pickup_speeds], False)

    return [
        (
            "JUMP",
            [
                # Ranked on impulse *and* gravity together, so the colour
                # reflects who actually jumps highest rather than who has the
                # biggest number. Luigi's weak impulse plus low gravity wins.
                ("Standing", _velocity(jump), None),
                ("Jump reach", _jump_reach_label(stats, everyone), jump_rank),
                ("Charged", _velocity(stats.jump_speed_charged_no_object), None),
                ("Running", _velocity(stats.jump_speed_running_no_object), None),
                ("Carrying", _velocity(stats.jump_speed_running_with_object), None),
                ("Quicksand", _velocity(stats.jump_speed_quicksand), None),
                (
                    "Float time",
                    f"{stats.floating_time} frames" if stats.floating_time else "None",
                    "accent" if stats.floating_time else None,
                ),
            ],
        ),
        (
            "PHYSICS",
            [
                ("Gravity", str(stats.gravity_with_jump), gravity_rank),
                ("Gravity (fall)", str(stats.gravity_without_jump), None),
                ("Gravity (sand)", str(stats.gravity_quicksand), None),
                ("Run speed", _velocity(run), None),
                ("Run carrying", _velocity(stats.running_speed_right_with_object), None),
                ("Run in sand", _velocity(stats.running_speed_right_quicksand), None),
            ],
        ),
        (
            "PICK UP",
            [
                ("Pull frames", ", ".join(str(v) for v in stats.pickup_speeds), None),
                ("Slowest step", f"{pull} frames", pull_rank),
                ("Character", CHARACTER_NAMES.get(pc.character, "Unknown"), "accent"),
            ],
        ),
    ]


ENEMY_COLUMNS: list[tuple[str, float]] = [
    ("#", 0.5),
    ("NAME", 2.0),
    ("HP", 0.5),
    ("X,Y", 1.2),
    ("REL X,Y", 1.2),
    ("VEL X,Y", 1.1),
    ("STATE", 1.1),
    ("TMR", 0.6),
    ("FLAGS", 2.4),
    ("COLLISION", 2.4),
]


def _enemy_rows(info: dict[str, Any]) -> list[list[str]]:
    """Build one display row per enemy slot."""
    pos = info['pos']
    rows = []
    for e in info['enemies']:
        present = e.object_type is not None
        has_xy = e.x_position is not None and e.y_position is not None
        rel_x, rel_y = e.relative_x(pos.x_global), e.relative_y(pos.y_global)
        has_vel = e.x_velocity is not None and e.y_velocity is not None
        rows.append(
            [
                str(e.slot_number),
                format_enemy_name(e.object_type),
                str(e.health) if e.health is not None else "",
                f"{e.x_position},{e.y_position}" if has_xy else "",
                f"{rel_x},{rel_y}" if rel_x is not None and rel_y is not None else "",
                f"{e.x_velocity},{e.y_velocity}" if has_vel else "",
                format_enemy_state(e.state, present),
                str(e.object_timer) if e.object_timer is not None else "",
                format_sprite_flags(e.sprite_flags),
                format_collision_flags(e.collision),
            ]
        )
    return rows


# ------------------------------------------------------------------------------
# ---- Rendering ---------------------------------------------------------------
# ------------------------------------------------------------------------------

_EMPHASIS_COLORS = {
    "accent": ACCENT,
    "good": GOOD,
    "warn": WARN,
    "bad": BAD,
    None: TEXT,
}


def draw_stats_panel(
    surface: pygame.Surface,
    card: pygame.Rect,
    fonts: Fonts,
    metrics: Metrics,
    info: dict[str, Any],
    active_tab: int,
) -> list[pygame.Rect]:
    """Draw the tabbed stats card.

    Returns:
        The clickable rect of each tab, so the caller can route mouse clicks.
    """
    draw_card(surface, card, metrics)
    tab_rects = _draw_tabs(surface, card, fonts, metrics, active_tab)

    content = pygame.Rect(
        card.x + metrics.pad,
        card.y + metrics.tab_height + metrics.pad,
        card.width - 2 * metrics.pad,
        card.height - metrics.tab_height - 2 * metrics.pad,
    )
    if content.height <= 0 or content.width <= 0:
        return tab_rects

    name = TABS[active_tab % len(TABS)]
    if name == "ENEMIES":
        _draw_enemy_table(surface, content, fonts, metrics, _enemy_rows(info))
    elif name == "PLAYER":
        _draw_groups(surface, content, fonts, metrics, _player_groups(info))
    elif name == "CHARACTER":
        _draw_groups(surface, content, fonts, metrics, _character_groups(info))
    else:
        _draw_groups(surface, content, fonts, metrics, _overview_groups(info))

    return tab_rects


def _draw_tabs(
    surface: pygame.Surface,
    card: pygame.Rect,
    fonts: Fonts,
    metrics: Metrics,
    active_tab: int,
) -> list[pygame.Rect]:
    """Draw the tab strip along the top of the stats card."""
    rects: list[pygame.Rect] = []
    x = card.x + metrics.pad
    y = card.y + metrics.pad // 2
    height = metrics.tab_height - metrics.pad // 2

    for i, name in enumerate(TABS):
        width = fonts.small.size(name)[0] + 3 * metrics.pad
        rect = pygame.Rect(x, y, width, height)
        rects.append(rect)

        active = i == active_tab % len(TABS)
        if active:
            pygame.draw.rect(surface, PANEL_ALT, rect, border_radius=metrics.radius // 2)
        label = fonts.small.render(name, True, TEXT if active else TEXT_DIM)
        surface.blit(
            label,
            (rect.centerx - label.get_width() // 2, rect.centery - label.get_height() // 2),
        )
        if active:
            pygame.draw.line(
                surface,
                ACCENT,
                (rect.x + metrics.pad, rect.bottom - 1),
                (rect.right - metrics.pad, rect.bottom - 1),
                2,
            )
        x += width

    hint = fonts.tiny.render("[Tab] or click  ·  [I] hides", True, TEXT_FAINT)
    hint_x = card.right - metrics.pad - hint.get_width()
    if hint_x > x + metrics.pad:
        surface.blit(hint, (hint_x, y + (height - hint.get_height()) // 2))

    pygame.draw.line(
        surface,
        BORDER,
        (card.x + metrics.pad, y + height),
        (card.right - metrics.pad, y + height),
        1,
    )
    return rects


def _draw_groups(
    surface: pygame.Surface,
    content: pygame.Rect,
    fonts: Fonts,
    metrics: Metrics,
    groups: list[Group],
) -> None:
    """Lay out label/value groups side by side, one column per group.

    When the window is too narrow for every group, the ones that don't fit are
    dropped rather than squashed into unreadable slivers.
    """
    row_h = max(15, int(18 * metrics.ui_scale))
    min_col = int(190 * metrics.ui_scale)
    max_cols = max(1, content.width // min_col)
    visible = groups[:max_cols]
    col_w = content.width // len(visible)

    for i, (title, rows) in enumerate(visible):
        x = content.x + i * col_w
        y = content.y
        inner_w = col_w - metrics.pad

        accent = SECTION_ACCENTS.get(title, ACCENT)
        surface.blit(fonts.small.render(title, True, accent), (x, y))
        y += row_h
        pygame.draw.line(surface, BORDER, (x, y - 3), (x + inner_w - metrics.pad, y - 3), 1)

        label_w = int(inner_w * 0.55)
        for label, value, emphasis in rows:
            if y + row_h > content.bottom:
                break
            blit_clipped(surface, fonts.small.render(label, True, TEXT_DIM), (x, y), label_w)
            color = _EMPHASIS_COLORS.get(emphasis, TEXT)
            value_text = truncate(fonts.small, value, inner_w - label_w - metrics.pad)
            surface.blit(fonts.small.render(value_text, True, color), (x + label_w, y))
            y += row_h

        if i < len(visible) - 1:
            line_x = x + col_w - metrics.pad // 2
            pygame.draw.line(surface, BORDER, (line_x, content.y), (line_x, content.bottom), 1)


def _draw_enemy_table(
    surface: pygame.Surface,
    content: pygame.Rect,
    fonts: Fonts,
    metrics: Metrics,
    rows: list[list[str]],
) -> None:
    """Draw the enemy slot table with zebra striping and dimmed empty slots.

    Columns are dropped from the right when the panel is too narrow, so the
    identifying columns (slot, name, position) always survive.
    """
    row_h = max(15, int(18 * metrics.ui_scale))

    # Weighted widths, then trim columns that don't fit
    total_weight = sum(w for _, w in ENEMY_COLUMNS)
    unit = content.width / total_weight
    widths, headers = [], []
    used = 0
    for header, weight in ENEMY_COLUMNS:
        w = int(unit * weight)
        if used + w > content.width and headers:
            break
        widths.append(w)
        headers.append(header)
        used += w

    gutter = max(6, int(8 * metrics.ui_scale))

    y = content.y
    x = content.x
    for header, width in zip(headers, widths, strict=True):
        label = truncate(fonts.tiny, header, width - gutter)
        surface.blit(fonts.tiny.render(label, True, ACCENT), (x, y))
        x += width
    y += row_h
    pygame.draw.line(surface, BORDER, (content.x, y - 3), (content.x + used, y - 3), 1)

    for index, row in enumerate(rows):
        if y + row_h > content.bottom:
            break
        present = bool(row[1])

        if index % 2 == 1:
            stripe = pygame.Rect(content.x - 2, y - 2, used + 4, row_h)
            pygame.draw.rect(surface, PANEL_ALT, stripe, border_radius=3)

        x = content.x
        # Deliberately ragged: `widths` is trimmed to the columns that fit, so
        # the zip stops early and drops the overflow cells. Not `strict=True`.
        for col, (cell, width) in enumerate(zip(row, widths)):  # noqa: B905
            if not present and col > 0:
                cell = cell or "—" if col == 1 else cell
            if col == 0:
                color = TEXT_DIM if present else TEXT_FAINT
            elif col == 1:
                color = TEXT if present else TEXT_FAINT
            else:
                color = TEXT_DIM if present else TEXT_FAINT
            if cell:
                text = truncate(fonts.tiny, cell, width - gutter)
                surface.blit(fonts.tiny.render(text, True, color), (x, y))
            x += width
        y += row_h
