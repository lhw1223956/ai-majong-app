from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.lyl_expected_value import (
    ALL_34_TILES_EV,
    TILE_ORDER_EV,
    build_deck_counts_ev,
    tie_score_ev,
)


@dataclass(frozen=True)
class ProgressEvaluation:
    score: int
    meld_count: int
    pair_used: int
    useful_tatsu_count: int
    distance: int


def _clean_codes(codes: Iterable[str] | None) -> list[str]:
    if codes is None:
        return []
    return [code for code in codes if code in TILE_ORDER_EV]


def _hand_counts_key(hand: Iterable[str]) -> tuple[int, ...]:
    counts = [0] * len(ALL_34_TILES_EV)
    for tile in hand:
        counts[TILE_ORDER_EV[tile]] += 1
    return tuple(counts)


def _deck_counts_key(deck_counts: dict[str, int]) -> tuple[int, ...]:
    return tuple(max(0, int(deck_counts.get(tile, 0))) for tile in ALL_34_TILES_EV)


def _has_live_consecutive_wait(left: int, deck_key: tuple[int, ...]) -> bool:
    rank = left % 9
    return (rank > 0 and deck_key[left - 1] > 0) or (rank + 2 <= 8 and deck_key[left + 2] > 0)


def _progress_sort_key(stats: tuple[int, int, int]) -> tuple[int, int, int]:
    melds, pairs, tatsu = stats
    return (melds, min(pairs, 1), min(tatsu, max(0, 5 - melds)))


@lru_cache(maxsize=200_000)
def _best_from_counts(counts_key: tuple[int, ...], deck_key: tuple[int, ...]) -> tuple[int, int, int]:
    work = list(counts_key)
    first = next((idx for idx, value in enumerate(work) if value > 0), -1)
    if first < 0:
        return (0, 0, 0)

    best = (0, 0, 0)

    def consider(candidate: tuple[int, int, int]) -> None:
        nonlocal best
        if _progress_sort_key(candidate) > _progress_sort_key(best):
            best = candidate

    work[first] -= 1
    consider(_best_from_counts(tuple(work), deck_key))
    work[first] += 1

    if work[first] >= 3:
        work[first] -= 3
        melds, pairs, tatsu = _best_from_counts(tuple(work), deck_key)
        consider((melds + 1, pairs, tatsu))
        work[first] += 3

    if first < 27 and first % 9 <= 6 and work[first + 1] >= 1 and work[first + 2] >= 1:
        work[first] -= 1
        work[first + 1] -= 1
        work[first + 2] -= 1
        melds, pairs, tatsu = _best_from_counts(tuple(work), deck_key)
        consider((melds + 1, pairs, tatsu))
        work[first] += 1
        work[first + 1] += 1
        work[first + 2] += 1

    if work[first] >= 2:
        work[first] -= 2
        melds, pairs, tatsu = _best_from_counts(tuple(work), deck_key)
        consider((melds, pairs + 1, tatsu))
        if deck_key[first] > 0:
            consider((melds, pairs, tatsu + 1))
        work[first] += 2

    if first < 27 and first % 9 <= 7 and work[first + 1] >= 1 and _has_live_consecutive_wait(first, deck_key):
        work[first] -= 1
        work[first + 1] -= 1
        melds, pairs, tatsu = _best_from_counts(tuple(work), deck_key)
        consider((melds, pairs, tatsu + 1))
        work[first] += 1
        work[first + 1] += 1

    if first < 27 and first % 9 <= 6 and work[first + 2] >= 1 and deck_key[first + 1] > 0:
        work[first] -= 1
        work[first + 2] -= 1
        melds, pairs, tatsu = _best_from_counts(tuple(work), deck_key)
        consider((melds, pairs, tatsu + 1))
        work[first] += 1
        work[first + 2] += 1

    return best


def _best_live_progress_stats(hand: list[str], deck_counts: dict[str, int]) -> tuple[int, int, int]:
    return _best_from_counts(_hand_counts_key(hand), _deck_counts_key(deck_counts))


def _evaluate_progress_from_counts_key(
    counts_key: tuple[int, ...],
    deck_key: tuple[int, ...],
    fixed_melds: int = 0,
) -> ProgressEvaluation:
    concealed_meld_count, pair_count, useful_tatsu_count = _best_from_counts(counts_key, deck_key)
    fixed_meld_count = min(max(int(fixed_melds), 0), 5)
    meld_count = min(5, fixed_meld_count + concealed_meld_count)
    pair_used = 1 if pair_count > 0 else 0
    useful_tatsu = min(useful_tatsu_count, max(0, 5 - meld_count))
    if meld_count >= 5 and pair_used:
        return ProgressEvaluation(
            score=100_000_000,
            meld_count=5,
            pair_used=1,
            useful_tatsu_count=0,
            distance=-1,
        )

    missing_melds = max(0, 5 - meld_count)
    missing_pair = max(0, 1 - pair_used)
    distance = missing_melds * 2 + missing_pair - useful_tatsu
    score = meld_count * 100_000 + pair_used * 10_000 + useful_tatsu * 1_000
    return ProgressEvaluation(
        score=score,
        meld_count=meld_count,
        pair_used=pair_used,
        useful_tatsu_count=useful_tatsu_count,
        distance=distance,
    )


def evaluate_progress(
    hand_codes: Iterable[str],
    deck_counts: dict[str, int] | None = None,
    visible_tiles: Iterable[str] | None = None,
    fixed_melds: int = 0,
) -> ProgressEvaluation:
    hand = _clean_codes(hand_codes)
    if deck_counts is None:
        deck_counts = build_deck_counts_ev(hand, _clean_codes(visible_tiles))

    return _evaluate_progress_from_counts_key(
        _hand_counts_key(hand),
        _deck_counts_key(deck_counts),
        fixed_melds=fixed_melds,
    )


def count_progress_improving_tiles(
    hand_codes: Iterable[str],
    visible_tiles: Iterable[str] | None = None,
    deck_counts: dict[str, int] | None = None,
    fixed_melds: int = 0,
) -> int:
    hand = _clean_codes(hand_codes)
    visible = _clean_codes(visible_tiles)
    if deck_counts is None:
        deck_counts = build_deck_counts_ev(hand, visible)
    hand_counts_key = _hand_counts_key(hand)
    deck_key = _deck_counts_key(deck_counts)
    base = _evaluate_progress_from_counts_key(hand_counts_key, deck_key, fixed_melds=fixed_melds)
    improving_count = 0
    for draw_index, count in enumerate(deck_key):
        if count <= 0:
            continue
        next_counts = list(hand_counts_key)
        next_counts[draw_index] += 1
        next_deck_key = list(deck_key)
        next_deck_key[draw_index] = max(0, int(count) - 1)
        after = _evaluate_progress_from_counts_key(
            tuple(next_counts),
            tuple(next_deck_key),
            fixed_melds=fixed_melds,
        )
        if (after.distance, -after.score) < (base.distance, -base.score):
            improving_count += int(count)
    return improving_count


def _calculate_progress_ev_detail(
    hand_counts_key: tuple[int, ...],
    discard: str,
    deck_key: tuple[int, ...],
    fixed_melds: int = 0,
) -> float:
    remaining_counts = list(hand_counts_key)
    remaining_counts[TILE_ORDER_EV[discard]] -= 1
    ev = 0.0
    for draw_index, count in enumerate(deck_key):
        if count <= 0:
            continue
        next_counts = list(remaining_counts)
        next_counts[draw_index] += 1
        next_deck_key = list(deck_key)
        next_deck_key[draw_index] = max(0, int(count) - 1)
        ev += int(count) * _evaluate_progress_from_counts_key(
            tuple(next_counts),
            tuple(next_deck_key),
            fixed_melds=fixed_melds,
        ).score
    return ev


def _normalize_ev(rows: list[dict]) -> None:
    values = [float(row["ev"]) for row in rows if row.get("ev_used", True)]
    if not values:
        for row in rows:
            row["ev_normalized"] = 0.0
        return
    ev_min = min(values)
    ev_max = max(values)
    if ev_max == ev_min:
        for row in rows:
            row["ev_normalized"] = 1.0 if row.get("ev_used", True) else 0.0
        return
    for row in rows:
        row["ev_normalized"] = (
            (float(row["ev"]) - ev_min) / (ev_max - ev_min)
            if row.get("ev_used", True)
            else 0.0
        )


def rank_discards_by_progress_ev(
    hand_codes: Iterable[str],
    visible_tiles: Iterable[str] | None = None,
    legal_discards: Iterable[str] | None = None,
    acceptance_ev_close_ratio: float = 0.995,
    acceptance_ev_close_abs: float = 1_000.0,
    fixed_melds: int = 0,
    compute_improving_tiles: bool = True,
    normalize_ev: bool = True,
    max_improving_tile_candidates: int = 2,
) -> dict:
    hand = _clean_codes(hand_codes)
    visible = _clean_codes(visible_tiles)
    legal_set = set(_clean_codes(legal_discards)) if legal_discards is not None else set(hand)
    candidates = [
        tile
        for tile in sorted(set(hand), key=lambda code: TILE_ORDER_EV[code])
        if tile in legal_set
    ]
    deck_counts = build_deck_counts_ev(hand, visible)
    deck_key = _deck_counts_key(deck_counts)
    hand_counts_key = _hand_counts_key(hand)
    rows = []
    for discard in candidates:
        remaining_counts = list(hand_counts_key)
        remaining_counts[TILE_ORDER_EV[discard]] -= 1
        progress = _evaluate_progress_from_counts_key(
            tuple(remaining_counts),
            deck_key,
            fixed_melds=fixed_melds,
        )
        rows.append(
            {
                "tile": discard,
                "ev": 0.0,
                "ev_used": False,
                "tie_score": tie_score_ev(discard),
                "progress_score": progress.score,
                "progress_distance": progress.distance,
                "progress_meld_count": progress.meld_count,
                "progress_pair_used": progress.pair_used,
                "progress_useful_tatsu_count": progress.useful_tatsu_count,
                "progress_improving_tiles": 0,
                "progress_improving_tiles_computed": False,
            }
        )

    min_distance = min((int(row["progress_distance"]) for row in rows), default=0)
    top_progress_score = max(
        (
            int(row["progress_score"])
            for row in rows
            if int(row["progress_distance"]) == min_distance
        ),
        default=0,
    )
    ev_candidate_tiles = {
        row["tile"]
        for row in rows
        if int(row["progress_distance"]) == min_distance
        and int(row["progress_score"]) == top_progress_score
    }
    should_compute_ev = len(ev_candidate_tiles) > 1
    for row in rows:
        if not should_compute_ev or row["tile"] not in ev_candidate_tiles:
            continue
        row["ev"] = _calculate_progress_ev_detail(
            hand_counts_key,
            row["tile"],
            deck_key,
            fixed_melds=fixed_melds,
        )
        row["ev_used"] = True

    if normalize_ev:
        _normalize_ev(rows)
    if compute_improving_tiles:
        _fill_near_ev_acceptance(
            rows=rows,
            hand=hand,
            visible=visible,
            deck_counts=deck_counts,
            ev_close_ratio=acceptance_ev_close_ratio,
            ev_close_abs=acceptance_ev_close_abs,
            fixed_melds=fixed_melds,
            max_candidates=max_improving_tile_candidates,
        )
    rows.sort(
        key=lambda row: (
            int(row["progress_distance"]),
            -int(row["progress_score"]),
            -float(row["ev"]),
            -int(row["progress_improving_tiles"]),
            int(tie_score_ev(row["tile"])),
            TILE_ORDER_EV[row["tile"]],
        )
    )
    _annotate_progress_ev_ranks(rows)
    return {
        "best_tiles": [row["tile"] for row in rows if row.get("progress_ev_rank") == 1],
        "results": rows,
        "summary": {
            "candidate_count": len(rows),
            "source": "lyl_live_progress_then_ev",
            "priority": [
                "progress_distance",
                "top_progress_score_group",
                "discard_ev_within_top_progress_group",
                "progress_improving_tiles_near_ev_only",
                "tie_score",
                "tile_order",
            ],
            "ev_policy": "only_when_min_distance_top_progress_score_group_has_multiple_candidates",
            "acceptance_ev_close_ratio": float(acceptance_ev_close_ratio),
            "acceptance_ev_close_abs": float(acceptance_ev_close_abs),
            "fixed_melds": int(fixed_melds),
            "compute_improving_tiles": bool(compute_improving_tiles),
            "normalize_ev": bool(normalize_ev),
            "max_improving_tile_candidates": int(max_improving_tile_candidates),
        },
    }


def _fill_near_ev_acceptance(
    rows: list[dict],
    hand: list[str],
    visible: list[str],
    deck_counts: dict[str, int],
    ev_close_ratio: float,
    ev_close_abs: float,
    fixed_melds: int = 0,
    max_candidates: int = 2,
) -> None:
    if not rows:
        return

    ev_rows = [row for row in rows if row.get("ev_used", False)]
    if not ev_rows:
        return

    best_ev = max(float(row["ev"]) for row in ev_rows)
    candidate_rows = []
    for row in ev_rows:
        if not row.get("ev_used", False):
            continue
        ev = float(row["ev"])
        close_by_ratio = ev >= best_ev * float(ev_close_ratio)
        close_by_abs = (best_ev - ev) <= float(ev_close_abs)
        if close_by_ratio or close_by_abs:
            candidate_rows.append(row)

    candidate_rows.sort(
        key=lambda row: (
            int(row["progress_distance"]),
            -int(row["progress_score"]),
            -float(row["ev"]),
            int(tie_score_ev(row["tile"])),
            TILE_ORDER_EV[row["tile"]],
        )
    )
    for row in candidate_rows[: max(0, int(max_candidates))]:

        remaining = list(hand)
        remaining.remove(row["tile"])
        row["progress_improving_tiles"] = count_progress_improving_tiles(
            remaining,
            visible,
            deck_counts=deck_counts,
            fixed_melds=fixed_melds,
        )
        row["progress_improving_tiles_computed"] = True


def _annotate_progress_ev_ranks(rows: list[dict], ev_tolerance: float = 1e-6) -> None:
    previous_key = None
    previous_rank = 0
    group = 0
    for position, row in enumerate(rows, start=1):
        key = (
            int(row["progress_distance"]),
            int(row["progress_score"]),
            round(float(row["ev"]) / max(ev_tolerance, 1e-12)),
            int(row["progress_improving_tiles"]),
        )
        if key != previous_key:
            group += 1
            previous_rank = position
            previous_key = key
        row["progress_ev_rank"] = previous_rank
        row["progress_ev_rank_group"] = group


def get_best_discard_by_progress_ev(
    hand_codes: Iterable[str],
    visible_tiles: Iterable[str] | None = None,
    legal_discards: Iterable[str] | None = None,
    acceptance_ev_close_ratio: float = 0.995,
    acceptance_ev_close_abs: float = 1_000.0,
    fixed_melds: int = 0,
    compute_improving_tiles: bool = True,
    normalize_ev: bool = True,
    max_improving_tile_candidates: int = 2,
) -> tuple[str | None, list[dict]]:
    ranking = rank_discards_by_progress_ev(
        hand_codes=hand_codes,
        visible_tiles=visible_tiles,
        legal_discards=legal_discards,
        acceptance_ev_close_ratio=acceptance_ev_close_ratio,
        acceptance_ev_close_abs=acceptance_ev_close_abs,
        fixed_melds=fixed_melds,
        compute_improving_tiles=compute_improving_tiles,
        normalize_ev=normalize_ev,
        max_improving_tile_candidates=max_improving_tile_candidates,
    )
    results = ranking["results"]
    if not results:
        return None, []
    return results[0]["tile"], results


if __name__ == "__main__":
    sample = "1w 2w 3w 7w 7w 1D 5D 5D 7D 9D 9D 3s 4s 5s 9s zhong fa".split()
    best, details = get_best_discard_by_progress_ev(sample)
    print(f"best discard: {best}")
    for item in details:
        print(
            f"{item['progress_ev_rank']}. {item['tile']} "
            f"distance={item['progress_distance']} "
            f"progress={item['progress_score']} "
            f"ev={int(item['ev'])} "
            f"acceptance={item['progress_improving_tiles']}"
        )
