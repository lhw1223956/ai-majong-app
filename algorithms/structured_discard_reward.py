from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from algorithms import rl_env

MELD_WEIGHT = 0.30
PAIR_PRIMARY_WEIGHT = 0.12
PAIR_EXTRA_WEIGHT = 0.04
TATSU_WEIGHT = 0.06
BREAK_MELD_PENALTY = 0.15
BREAK_PAIR_PENALTY = 0.08
BREAK_TATSU_PENALTY = 0.04
STRUCTURE_REWARD_CLIP = 0.30


@dataclass(frozen=True)
class StructureEvaluation:
    meld_count: int
    pair_count: int
    effective_pair_score: float
    tatsu_count: int
    break_penalty: float
    score: float


def _is_suited(tile: int) -> bool:
    return tile < rl_env.HONOR_START


def _same_suit(tile_a: int, tile_b: int) -> bool:
    return rl_env.tile_suit(tile_a) == rl_env.tile_suit(tile_b)


def count_effective_pairs(pair_count: int) -> float:
    if pair_count <= 0:
        return 0.0
    return 1.0 + max(0, pair_count - 1) * (PAIR_EXTRA_WEIGHT / PAIR_PRIMARY_WEIGHT)


@lru_cache(maxsize=200_000)
def _best_structure_stats(counts_key: tuple[int, ...]) -> tuple[int, int, int]:
    counts = list(counts_key)

    first = -1
    for idx, value in enumerate(counts):
        if value > 0:
            first = idx
            break

    if first == -1:
        return (0, 0, 0)

    best = (0, 0, 0)

    def consider(candidate: tuple[int, int, int]) -> None:
        nonlocal best
        if _structure_sort_key(candidate) > _structure_sort_key(best):
            best = candidate

    counts[first] -= 1
    consider(_best_structure_stats(tuple(counts)))
    counts[first] += 1

    if counts[first] >= 3:
        counts[first] -= 3
        melds, pairs, tatsu = _best_structure_stats(tuple(counts))
        consider((melds + 1, pairs, tatsu))
        counts[first] += 3

    if (
        _is_suited(first)
        and first + 2 < rl_env.NUM_TILE_TYPES
        and _same_suit(first, first + 1)
        and _same_suit(first, first + 2)
        and counts[first + 1] >= 1
        and counts[first + 2] >= 1
    ):
        counts[first] -= 1
        counts[first + 1] -= 1
        counts[first + 2] -= 1
        melds, pairs, tatsu = _best_structure_stats(tuple(counts))
        consider((melds + 1, pairs, tatsu))
        counts[first] += 1
        counts[first + 1] += 1
        counts[first + 2] += 1

    if counts[first] >= 2:
        counts[first] -= 2
        melds, pairs, tatsu = _best_structure_stats(tuple(counts))
        consider((melds, pairs + 1, tatsu))
        counts[first] += 2

    if (
        _is_suited(first)
        and first + 1 < rl_env.NUM_TILE_TYPES
        and _same_suit(first, first + 1)
        and counts[first + 1] >= 1
    ):
        counts[first] -= 1
        counts[first + 1] -= 1
        melds, pairs, tatsu = _best_structure_stats(tuple(counts))
        consider((melds, pairs, tatsu + 1))
        counts[first] += 1
        counts[first + 1] += 1

    if (
        _is_suited(first)
        and first + 2 < rl_env.NUM_TILE_TYPES
        and _same_suit(first, first + 2)
        and counts[first + 2] >= 1
    ):
        counts[first] -= 1
        counts[first + 2] -= 1
        melds, pairs, tatsu = _best_structure_stats(tuple(counts))
        consider((melds, pairs, tatsu + 1))
        counts[first] += 1
        counts[first + 2] += 1

    return best


def _structure_sort_key(stats: tuple[int, int, int]) -> tuple[float, int, int, int]:
    melds, pairs, tatsu = stats
    pair_score = count_effective_pairs(pairs)
    numeric_score = (
        melds * MELD_WEIGHT
        + pair_score * PAIR_PRIMARY_WEIGHT
        + tatsu * TATSU_WEIGHT
    )
    return (numeric_score, melds, min(pairs, 1), tatsu)


def count_tatsu(hand_counts: np.ndarray) -> int:
    _, _, tatsu = _best_structure_stats(tuple(int(v) for v in hand_counts.tolist()))
    return tatsu


def evaluate_hand_structure(
    hand_counts: np.ndarray,
    melds_count: int = 0,
) -> StructureEvaluation:
    concealed_melds, pair_count, tatsu_count = _best_structure_stats(
        tuple(int(v) for v in hand_counts.tolist())
    )
    meld_count = melds_count + concealed_melds
    effective_pair_score = count_effective_pairs(pair_count)
    score = (
        meld_count * MELD_WEIGHT
        + effective_pair_score * PAIR_PRIMARY_WEIGHT
        + tatsu_count * TATSU_WEIGHT
    )
    return StructureEvaluation(
        meld_count=meld_count,
        pair_count=pair_count,
        effective_pair_score=effective_pair_score,
        tatsu_count=tatsu_count,
        break_penalty=0.0,
        score=score,
    )


def compute_discard_structure_reward(
    hand_before: np.ndarray,
    hand_after: np.ndarray,
    melds_count: int = 0,
    reward_scale: float = 1.0,
) -> dict:
    before = evaluate_hand_structure(hand_before, melds_count)
    after = evaluate_hand_structure(hand_after, melds_count)

    meld_drop = max(0, before.meld_count - after.meld_count)
    pair_drop = max(0.0, before.effective_pair_score - after.effective_pair_score)
    tatsu_drop = max(0, before.tatsu_count - after.tatsu_count)
    break_penalty = (
        meld_drop * BREAK_MELD_PENALTY
        + pair_drop * BREAK_PAIR_PENALTY
        + tatsu_drop * BREAK_TATSU_PENALTY
    )

    reward = (after.score - before.score) - break_penalty
    reward = float(np.clip(reward * reward_scale, -STRUCTURE_REWARD_CLIP, STRUCTURE_REWARD_CLIP))

    return {
        "reward": reward,
        "before": before,
        "after": StructureEvaluation(
            meld_count=after.meld_count,
            pair_count=after.pair_count,
            effective_pair_score=after.effective_pair_score,
            tatsu_count=after.tatsu_count,
            break_penalty=break_penalty,
            score=after.score,
        ),
        "break_penalty": break_penalty,
    }
