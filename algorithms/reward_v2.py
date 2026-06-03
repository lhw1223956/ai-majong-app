from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from algorithms import rl_env
from algorithms.structured_discard_reward import evaluate_hand_structure


DEFICIENCY_DELTA_REWARD = 0.06
FIRST_TENPAI_REWARD = 0.25
KEEP_TENPAI_REWARD = 0.02
BREAK_TENPAI_PENALTY = 0.20

BREAK_MELD_PENALTY_V2 = 0.20
BREAK_PAIR_PENALTY_V2 = 0.08
BREAK_TATSU_PENALTY_V2 = 0.04

MELD_INCREASE_BONUS = 0.08
MELD_3_TO_4_BONUS = 0.08
MELD_4_TO_5_BONUS = 0.12

CREATE_PAIR_REWARD = 0.04
KEEP_SINGLE_PAIR_REWARD = 0.01
BREAK_SINGLE_PAIR_PENALTY = 0.08

TATSU_WITH_MELD_BONUS = 0.04
TATSU_TO_MELD_BONUS = 0.06
NEW_TATSU_SMALL_BONUS = 0.01

STALL_TATSU_PENALTY = 0.03
STALL_PAIR_PENALTY = 0.03

DISCARD_REWARD_V2_CLIP = 0.35

EXHAUSTED_TENPAI_PENALTY = -0.10
EXHAUSTED_DEFICIENCY_1_PENALTY = -0.30
EXHAUSTED_DEFICIENCY_2_PENALTY = -0.50
EXHAUSTED_DEFICIENCY_3_PLUS_PENALTY = -0.80


@dataclass(frozen=True)
class RewardV2Result:
    reward: float
    clipped_reward: float
    deficiency_delta_reward: float
    tenpai_transition_reward: float
    structure_delta_reward: float
    structure_break_penalty: float
    meld_progress_bonus: float
    pair_shape_reward: float
    tatsu_to_meld_reward: float
    bad_stall_penalty: float
    before_deficiency: int
    after_deficiency: int
    before_tenpai: bool
    after_tenpai: bool
    first_tenpai: bool
    structure_score_before: float
    structure_score_after: float
    structure_meld_before: int
    structure_meld_after: int
    structure_pair_before: int
    structure_pair_after: int
    structure_effective_pair_before: float
    structure_effective_pair_after: float
    structure_tatsu_before: int
    structure_tatsu_after: int


def _deficiency_delta_reward(before_deficiency: int, after_deficiency: int) -> float:
    if after_deficiency < before_deficiency:
        return DEFICIENCY_DELTA_REWARD
    if after_deficiency > before_deficiency:
        return -DEFICIENCY_DELTA_REWARD
    return 0.0


def _tenpai_transition_reward(
    before_tenpai: bool,
    after_tenpai: bool,
    already_reached_tenpai: bool,
) -> tuple[float, bool]:
    if after_tenpai and not before_tenpai and not already_reached_tenpai:
        return FIRST_TENPAI_REWARD, True
    if before_tenpai and after_tenpai:
        return KEEP_TENPAI_REWARD, False
    if before_tenpai and not after_tenpai:
        return -BREAK_TENPAI_PENALTY, False
    return 0.0, False


def _structure_break_penalty(before, after) -> float:
    meld_drop = max(0, before.meld_count - after.meld_count)
    pair_drop = max(0.0, before.effective_pair_score - after.effective_pair_score)
    tatsu_drop = max(0, before.tatsu_count - after.tatsu_count)
    return -(
        meld_drop * BREAK_MELD_PENALTY_V2
        + pair_drop * BREAK_PAIR_PENALTY_V2
        + tatsu_drop * BREAK_TATSU_PENALTY_V2
    )


def _meld_progress_bonus(before, after) -> float:
    if after.meld_count <= before.meld_count:
        return 0.0

    bonus = MELD_INCREASE_BONUS
    if before.meld_count <= 3 <= after.meld_count - 1:
        bonus += MELD_3_TO_4_BONUS
    if before.meld_count <= 4 <= after.meld_count - 1:
        bonus += MELD_4_TO_5_BONUS
    return bonus


def _pair_shape_reward(before, after) -> float:
    reward = 0.0
    if before.pair_count <= 0 and after.pair_count >= 1:
        reward += CREATE_PAIR_REWARD
    if before.pair_count == 1 and after.pair_count == 1:
        reward += KEEP_SINGLE_PAIR_REWARD
    if before.pair_count >= 1 and after.pair_count <= 0:
        reward -= BREAK_SINGLE_PAIR_PENALTY
    return reward


def _tatsu_to_meld_reward(before, after) -> float:
    meld_increased = after.meld_count > before.meld_count
    reward = 0.0
    if meld_increased and after.tatsu_count < before.tatsu_count:
        reward += TATSU_TO_MELD_BONUS
    elif meld_increased and 0 < after.tatsu_count <= 1:
        reward += TATSU_WITH_MELD_BONUS
    elif not meld_increased and after.tatsu_count > before.tatsu_count:
        reward += NEW_TATSU_SMALL_BONUS
    return reward


def _bad_stall_penalty(after) -> float:
    penalty = 0.0
    if after.meld_count <= 2 and after.tatsu_count >= 2:
        penalty -= STALL_TATSU_PENALTY
    if after.meld_count <= 2 and after.pair_count >= 2:
        penalty -= STALL_PAIR_PENALTY
    return penalty


def compute_exhausted_deficiency_penalty(hand_counts: np.ndarray, melds_count: int = 0) -> tuple[float, int]:
    final_deficiency = rl_env.calc_deficiency(hand_counts, melds_count)
    if final_deficiency <= 0:
        return EXHAUSTED_TENPAI_PENALTY, final_deficiency
    if final_deficiency == 1:
        return EXHAUSTED_DEFICIENCY_1_PENALTY, final_deficiency
    if final_deficiency == 2:
        return EXHAUSTED_DEFICIENCY_2_PENALTY, final_deficiency
    return EXHAUSTED_DEFICIENCY_3_PLUS_PENALTY, final_deficiency


def compute_discard_reward_v2(
    hand_before: np.ndarray,
    hand_after: np.ndarray,
    melds_count: int = 0,
    already_reached_tenpai: bool = False,
    structure_reward_scale: float = 1.0,
    reward_scale: float = 1.0,
) -> RewardV2Result:
    before = evaluate_hand_structure(hand_before, melds_count)
    after = evaluate_hand_structure(hand_after, melds_count)

    before_deficiency = rl_env.calc_deficiency(hand_before, melds_count)
    after_deficiency = rl_env.calc_deficiency(hand_after, melds_count)
    before_tenpai = before_deficiency <= 0
    after_tenpai = after_deficiency <= 0

    deficiency_reward = _deficiency_delta_reward(before_deficiency, after_deficiency)
    tenpai_reward, first_tenpai = _tenpai_transition_reward(
        before_tenpai=before_tenpai,
        after_tenpai=after_tenpai,
        already_reached_tenpai=already_reached_tenpai,
    )
    structure_delta = (after.score - before.score) * structure_reward_scale
    break_penalty = _structure_break_penalty(before, after)
    meld_bonus = _meld_progress_bonus(before, after)
    pair_reward = _pair_shape_reward(before, after)
    tatsu_reward = _tatsu_to_meld_reward(before, after)
    stall_penalty = _bad_stall_penalty(after)

    reward = (
        deficiency_reward
        + tenpai_reward
        + structure_delta
        + break_penalty
        + meld_bonus
        + pair_reward
        + tatsu_reward
        + stall_penalty
    ) * reward_scale
    clipped_reward = float(np.clip(reward, -DISCARD_REWARD_V2_CLIP, DISCARD_REWARD_V2_CLIP))

    return RewardV2Result(
        reward=float(reward),
        clipped_reward=clipped_reward,
        deficiency_delta_reward=float(deficiency_reward),
        tenpai_transition_reward=float(tenpai_reward),
        structure_delta_reward=float(structure_delta),
        structure_break_penalty=float(break_penalty),
        meld_progress_bonus=float(meld_bonus),
        pair_shape_reward=float(pair_reward),
        tatsu_to_meld_reward=float(tatsu_reward),
        bad_stall_penalty=float(stall_penalty),
        before_deficiency=int(before_deficiency),
        after_deficiency=int(after_deficiency),
        before_tenpai=bool(before_tenpai),
        after_tenpai=bool(after_tenpai),
        first_tenpai=bool(first_tenpai),
        structure_score_before=float(before.score),
        structure_score_after=float(after.score),
        structure_meld_before=int(before.meld_count),
        structure_meld_after=int(after.meld_count),
        structure_pair_before=int(before.pair_count),
        structure_pair_after=int(after.pair_count),
        structure_effective_pair_before=float(before.effective_pair_score),
        structure_effective_pair_after=float(after.effective_pair_score),
        structure_tatsu_before=int(before.tatsu_count),
        structure_tatsu_after=int(after.tatsu_count),
    )
