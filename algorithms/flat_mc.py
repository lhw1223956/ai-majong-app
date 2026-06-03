import random
import collections
import numpy as np
# 引入既有的特徵對照表與向聽數計算函數
from algorithms.rl_env import calc_deficiency
from algorithms.rl_env import HAND_SIZE
from core.config import CODE_TO_IDX


def _codes_to_counts(codes):
    counts = np.zeros(34, dtype=np.int32)
    for code in codes:
        if code in CODE_TO_IDX:
            counts[CODE_TO_IDX[code]] += 1
    return counts


def _build_unknown_deck(hand_codes, discard_pool, TILE_INFO, exp_codes=None):
    visible_tiles = (hand_codes or []) + (discard_pool or []) + (exp_codes or [])
    visible_tiles = [c for c in visible_tiles if c in TILE_INFO and TILE_INFO[c].get('type') != 'h']
    visible_counts = collections.Counter(visible_tiles)

    deck = []
    for code, info in TILE_INFO.items():
        if info['type'] != 'h':
            remains = 4 - visible_counts.get(code, 0)
            if remains > 0:
                deck.extend([code] * remains)
    return deck


def flat_mc_evaluate_details(
    hand_codes,
    candidate_discard,
    discard_pool,
    TILE_INFO,
    melds_count=0,
    exp_codes=None,
    num_simulations=40,
    max_depth=20,
    seed=None,
):
    """
    回傳 FlatMC 的細部評分，用於 discard Q / Critic 輔助資料集。
    """
    rng = random.Random(seed) if seed is not None else random
    deck = _build_unknown_deck(hand_codes, discard_pool, TILE_INFO, exp_codes)

    base_hand = [c for c in (hand_codes or []) if c in CODE_TO_IDX]
    if candidate_discard in base_hand:
        base_hand.remove(candidate_discard)

    base_deficiency = int(calc_deficiency(_codes_to_counts(base_hand), melds_count))
    wins = 0
    tenpai_hits = 0
    deficiency_improvement_sum = 0.0

    for _ in range(num_simulations):
        sim_hand = list(base_hand)
        sim_deck = list(deck)
        rng.shuffle(sim_deck)
        best_deficiency = base_deficiency
        reached_tenpai = base_deficiency <= 0
        won = False

        for _depth in range(min(max_depth, len(sim_deck))):
            draw = sim_deck.pop()
            sim_hand.append(draw)

            hand_arr = _codes_to_counts(sim_hand)
            deficiency = int(calc_deficiency(hand_arr, melds_count))
            best_deficiency = min(best_deficiency, deficiency)
            if deficiency <= 0:
                reached_tenpai = True
            if deficiency == -1:
                won = True
                break

            unique_sim_tiles = list(set(sim_hand))
            best_pop_val = -1
            min_d = 10
            rng.shuffle(unique_sim_tiles)

            for tile in unique_sim_tiles:
                temp_list = list(sim_hand)
                temp_list.remove(tile)
                d = int(calc_deficiency(_codes_to_counts(temp_list), melds_count))
                if d < min_d:
                    min_d = d
                    best_pop_val = tile
                    if d == 0:
                        break

            if best_pop_val != -1:
                sim_hand.remove(best_pop_val)
            else:
                sim_hand.pop(rng.randint(0, len(sim_hand) - 1))

        wins += int(won)
        tenpai_hits += int(reached_tenpai)
        deficiency_improvement_sum += max(0.0, float(base_deficiency - best_deficiency))

    simulation_count = max(num_simulations, 1)
    win_rate = wins / simulation_count
    tenpai_rate = tenpai_hits / simulation_count
    avg_deficiency_improvement = deficiency_improvement_sum / simulation_count
    normalized_improvement = avg_deficiency_improvement / max(float(base_deficiency), 1.0)
    mc_score = (0.50 * win_rate) + (0.50 * normalized_improvement)

    return {
        'tile': candidate_discard,
        'win_rate': win_rate,
        'tenpai_rate': tenpai_rate,
        'avg_deficiency_improvement': avg_deficiency_improvement,
        'normalized_deficiency_improvement': normalized_improvement,
        'base_deficiency': base_deficiency,
        'mc_score': mc_score,
        'num_simulations': num_simulations,
        'max_depth': max_depth,
    }


def flat_mc_evaluate(hand_codes, candidate_discard, discard_pool, TILE_INFO, melds_count=0, exp_codes=None, num_simulations=40, max_depth=20):
    """
    FlatMC 平坦蒙地卡羅模擬器 - 增強版
    """
    details = flat_mc_evaluate_details(
        hand_codes,
        candidate_discard,
        discard_pool,
        TILE_INFO,
        melds_count,
        exp_codes,
        num_simulations,
        max_depth,
    )
    return details['win_rate']

def get_best_discard_with_flatmc(hand_codes, cnn_top_candidates, discard_pool, TILE_INFO, exp_codes=None):
    """
    結合 CNN 的前 N 個候選，交給 FlatMC 決選。
    """
    best_candidate = None
    best_win_rate = -1.0
    
    results = []
    for candidate in cnn_top_candidates:
        # 用「暗牌張數」反推出面子數，避免槓牌(4張)讓 exp_codes // 3 失真
        base_hand = [c for c in (hand_codes or []) if c in CODE_TO_IDX]
        if candidate in base_hand:
            base_hand.remove(candidate)
        melds_count = max(0, (HAND_SIZE - len(base_hand)) // 3)

        win_rate = flat_mc_evaluate(hand_codes, candidate, discard_pool, TILE_INFO, melds_count, exp_codes)
        results.append({'tile': candidate, 'win_rate': win_rate})
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_candidate = candidate
            
    return best_candidate, best_win_rate, results
