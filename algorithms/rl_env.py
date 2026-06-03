"""
台灣麻將強化學習環境 (Taiwan Mahjong MARL Environment)
基於 PettingZoo AECEnv，實作完整台灣麻將規則
- 136張牌（34種×4，不含花牌）
- 16張起手，摸牌後17張，胡牌結構：5面子+1眼
- 完整台分計算、吃碰槓胡邏輯
"""

import functools
import random
from collections import defaultdict
import numpy as np
from gymnasium.spaces import Discrete, Box
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils import wrappers
from core.calculator import run_full_logic
from core.config import IDX_TO_CODE
from algorithms.lyl_progress_ev_judgement import evaluate_progress

# ─────────────────────────────────────────
# 常數定義
# ─────────────────────────────────────────
NUM_TILE_TYPES = 34        # 34種牌（萬1-9, 筒1-9, 索1-9, 東南西北中發白）
TOTAL_TILES    = 136       # 136張牌（不含花牌）
HAND_SIZE      = 16        # 起手牌數
WIN_HAND_SIZE  = 17        # 胡牌時手牌數

# 花色區段
MAN_START, MAN_END     = 0,  8   # 萬子 0-8
PIN_START, PIN_END     = 9,  17  # 筒子 9-17
SOU_START, SOU_END     = 18, 26  # 索子 18-26
HONOR_START, HONOR_END = 27, 33  # 字牌 27-33（東南西北中發白）
EAST, SOUTH, WEST, NORTH, CHUN, HATSU, HAKU = 27, 28, 29, 30, 31, 32, 33

# 動作空間
# 0-33: 打出牌 i
# 34: 吃（上家）
# 35: 碰（任意家）
# 36: 明槓
# 37: 暗槓
# 38: 加槓
# 39: 胡
# 40: 過（跳過鳴牌機會）
ACTION_DISCARD_START = 0
ACTION_DISCARD_END   = 33
ACTION_CHI   = 34
ACTION_PONG  = 35
ACTION_OPEN_KONG  = 36
ACTION_DARK_KONG  = 37
ACTION_ADD_KONG   = 38
ACTION_HU    = 39
ACTION_PASS  = 40
NUM_ACTIONS  = 41

EXHAUSTED_TENPAI_PENALTY = -0.10
EXHAUSTED_DEFICIENCY_1_PENALTY = -0.30
EXHAUSTED_DEFICIENCY_2_PENALTY = -0.50
EXHAUSTED_DEFICIENCY_3_PLUS_PENALTY = -0.80

# 觀察空間維度 [8 × 34]
# Plane 0-3：已知牌（自己手牌 + 自己鳴牌 + 所有棄牌 + 場上公開鳴牌）
# Plane 4-7：未知牌（每種牌最多 4 張 - 已知數量）
#
# ─── 手牌張數說明（台灣16張麻將）───
# 設 m = 已完成鳴牌面子數（吃/碰各=1，槓=1）
# Draw Phase（摸牌後，模型需決定捨牌）: 手牌 = 3(5-m)+2 張
#   - 0面子: 17張  |  1碰: 14張  |  2碰: 11張  |  3碰: 8張
# 捨牌後（等待進張/聽牌中）        : 手牌 = 3(5-m)+1 張
#   - 0面子: 16張  |  1碰: 13張  |  2碰: 10張  |  3碰: 7張
# CNN 模型的觀察與動作皆發生在 Draw Phase（3n+2 狀態）
OBS_PLANES = 8

# ─────────────────────────────────────────
# 牌張工具函式
# ─────────────────────────────────────────

def tile_suit(tile: int) -> int:
    """回傳花色: 0=萬, 1=筒, 2=索, 3=字"""
    if tile <= MAN_END:   return 0
    if tile <= PIN_END:   return 1
    if tile <= SOU_END:   return 2
    return 3

def tile_rank(tile: int) -> int:
    """回傳牌號（花色內的序號，0-indexed）"""
    if tile <= MAN_END:   return tile - MAN_START
    if tile <= PIN_END:   return tile - PIN_START
    if tile <= SOU_END:   return tile - SOU_START
    return tile - HONOR_START

def is_honor(tile: int) -> bool:
    return tile >= HONOR_START

def is_terminal(tile: int) -> bool:
    """么九牌（1/9萬筒索）"""
    if is_honor(tile):
        return True
    r = tile_rank(tile)
    return r == 0 or r == 8

def tile_name(tile: int) -> str:
    suits = ["萬", "筒", "索", ""]
    ranks_num = ["一","二","三","四","五","六","七","八","九"]
    honors = ["東","南","西","北","中","發","白"]
    s = tile_suit(tile)
    r = tile_rank(tile)
    if s == 3:
        return honors[r]
    return ranks_num[r] + suits[s]

def make_deck() -> list:
    """建立136張牌的牌山（不含花牌）"""
    return list(range(NUM_TILE_TYPES)) * 4

# ─────────────────────────────────────────
# 胡牌判斷
# ─────────────────────────────────────────

def _can_form_melds(counts: list, n_melds: int) -> bool:
    """DFS：判斷 counts（長度34計數陣列）能否湊成 n_melds 個面子"""
    if n_melds == 0:
        return sum(counts) == 0
    if sum(counts) == 0:
        return False
    # 找第一張有牌的
    for i in range(NUM_TILE_TYPES):
        if counts[i] == 0:
            continue
        # 嘗試刻子
        if counts[i] >= 3:
            counts[i] -= 3
            if _can_form_melds(counts, n_melds - 1):
                counts[i] += 3
                return True
            counts[i] += 3
        # 嘗試順子（只有數字牌）
        if not is_honor(i):
            s = tile_suit(i)
            if (i + 2 < NUM_TILE_TYPES and
                    tile_suit(i+1) == s and tile_suit(i+2) == s and
                    counts[i+1] >= 1 and counts[i+2] >= 1):
                counts[i] -= 1; counts[i+1] -= 1; counts[i+2] -= 1
                if _can_form_melds(counts, n_melds - 1):
                    counts[i] += 1; counts[i+1] += 1; counts[i+2] += 1
                    return True
                counts[i] += 1; counts[i+1] += 1; counts[i+2] += 1
        # 若第一張牌無法湊成任何面子，必然失敗
        return False
    return True  # counts 全為0，n_melds 應也為0

def is_winning_hand(hand_counts: np.ndarray, melds_count: int = 0) -> bool:
    """
    判斷是否胡牌（台灣麻將：5面子+1眼）
    hand_counts: 34維計數陣列（手上的牌，不含已鳴牌）
    melds_count: 已鳴牌的面子數（每個吃/碰/槓算1個面子）
    """
    total = int(hand_counts.sum())
    needed_melds = 5 - melds_count  # 還需手牌湊幾個面子
    expected_total = needed_melds * 3 + 2
    if total != expected_total:
        return False

    # 十三么已移除（台灣16張麻將較少使用此規則）

    # 一般胡牌：枚舉眼
    counts = list(hand_counts)
    for i in range(NUM_TILE_TYPES):
        if counts[i] >= 2:
            counts[i] -= 2
            if _can_form_melds(counts[:], needed_melds):
                counts[i] += 2
                return True
            counts[i] += 2
    return False

def get_winning_tiles(hand_counts: np.ndarray, melds_count: int = 0) -> list:
    """回傳所有能讓手牌胡牌的聽牌（有效張）"""
    result = []
    for t in range(NUM_TILE_TYPES):
        if hand_counts[t] < 4:
            hand_counts[t] += 1
            if is_winning_hand(hand_counts, melds_count):
                result.append(t)
            hand_counts[t] -= 1
    return result

# ─────────────────────────────────────────
# 進胡數計算（Deficiency Number）
# ─────────────────────────────────────────

def calc_deficiency(hand_counts: np.ndarray, melds_count: int = 0) -> int:
    """
    使用 LYL progress distance 計算進度距離。
    -1 = 已胡牌（17張滿足胡牌條件）
    正整數 = 離 5 面子 + 1 眼仍差多少有效進度
    melds_count 為已鳴牌面子數，會直接計入既有面子。
    """
    if is_winning_hand(hand_counts, melds_count):
        return -1

    hand_codes = []
    for tile_idx, count in enumerate(hand_counts[:NUM_TILE_TYPES]):
        code = IDX_TO_CODE.get(tile_idx)
        if code:
            hand_codes.extend([code] * int(count))
    progress = evaluate_progress(hand_codes, fixed_melds=melds_count)
    return int(progress.distance)


def compute_exhausted_deficiency_penalty(
    hand_counts: np.ndarray,
    melds_count: int = 0,
) -> tuple[float, int]:
    final_deficiency = int(calc_deficiency(hand_counts, melds_count))
    if final_deficiency <= 0:
        return EXHAUSTED_TENPAI_PENALTY, final_deficiency
    if final_deficiency == 1:
        return EXHAUSTED_DEFICIENCY_1_PENALTY, final_deficiency
    if final_deficiency == 2:
        return EXHAUSTED_DEFICIENCY_2_PENALTY, final_deficiency
    return EXHAUSTED_DEFICIENCY_3_PLUS_PENALTY, final_deficiency




def _shanten_melds(counts: list, needed: int) -> int:
    """
    計算：手牌 counts 要湊成 needed 個完整面子，還差幾張牌？
    當 needed==0 時，多餘的牌不影響結果，直接返回0。
    """
    if needed <= 0:
        return 0  # 不需要更多面子，多餘的牌不管

    # 找第一張有牌
    i = 0
    while i < NUM_TILE_TYPES and counts[i] == 0:
        i += 1
    if i >= NUM_TILE_TYPES:
        # 沒牌但還需要面子：每個面子最樂觀差1張（搭子完成），最差差2張
        return needed  # 悲觀估計1張/面子（最佳情況有搭子）

    best = needed * 2  # worst case（全孤張，每個面子差2張）

    # 試完整面子：刻子
    if counts[i] >= 3:
        counts[i] -= 3
        best = min(best, _shanten_melds(counts, needed - 1))
        counts[i] += 3

    # 試完整面子：順子
    if not is_honor(i) and i + 2 < NUM_TILE_TYPES:
        s = tile_suit(i)
        if (tile_suit(i+1) == s and tile_suit(i+2) == s
                and counts[i+1] >= 1 and counts[i+2] >= 1):
            counts[i] -= 1; counts[i+1] -= 1; counts[i+2] -= 1
            best = min(best, _shanten_melds(counts, needed - 1))
            counts[i] += 1; counts[i+1] += 1; counts[i+2] += 1

    # 試搭子（差1張）：對子搭
    if counts[i] >= 2:
        counts[i] -= 2
        best = min(best, 1 + _shanten_melds(counts, needed - 1))
        counts[i] += 2

    # 試搭子：連張/嵌張搭
    if not is_honor(i):
        s = tile_suit(i)
        for j in [i+1, i+2]:
            if j < NUM_TILE_TYPES and tile_suit(j) == s and counts[j] >= 1:
                counts[i] -= 1; counts[j] -= 1
                best = min(best, 1 + _shanten_melds(counts, needed - 1))
                counts[i] += 1; counts[j] += 1

    # 孤張（差2張才能湊面子）：直接跳過，遞歸繼續
    saved = counts[i]
    counts[i] = 0
    best = min(best, 2 + _shanten_melds(counts, needed - 1))
    counts[i] = saved

    return best





# ─────────────────────────────────────────
# 台分計算系統
# ─────────────────────────────────────────

class TaiScoreCalculator:
    """完整台灣麻將台分計算"""

    WIND_NAMES = ["東", "南", "西", "北"]
    DEALER_RELATIVE_NAMES = {
        0: "我",
        1: "下家(右)",
        2: "對家(對面)",
        3: "上家(左)",
    }

    @staticmethod
    def _counts_to_codes(hand_counts: np.ndarray) -> list[str]:
        codes = []
        for tile, count in enumerate(hand_counts):
            code = IDX_TO_CODE.get(int(tile))
            if code is None:
                continue
            codes.extend([code] * int(count))
        return codes

    @staticmethod
    def _melds_to_codes(melds: list) -> list[str]:
        codes = []
        for meld in melds:
            for tile in meld.get("tiles", []):
                code = IDX_TO_CODE.get(int(tile))
                if code is not None:
                    codes.append(code)
        return codes

    @staticmethod
    def _detail_to_tuple(detail: str) -> tuple[str, int]:
        marker = "台"
        if marker not in detail:
            return (detail, 0)
        before_marker = detail.rsplit(marker, 1)[0].rstrip()
        parts = before_marker.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return (parts[0], int(parts[1]))
        return (detail, 0)

    @staticmethod
    def _dealer_relative_name(player_wind: int, game_state=None) -> str:
        dealer_idx = int(getattr(game_state, "dealer_idx", 0))
        relative_idx = (dealer_idx - int(player_wind)) % 4
        return TaiScoreCalculator.DEALER_RELATIVE_NAMES[relative_idx]

    @staticmethod
    def calc_score(hand_counts: np.ndarray, melds: list, win_tile: int,
                   is_tsumo: bool, is_menzen: bool, player_wind: int,
                   round_wind: int, game_state=None) -> dict:
        """
        計算台分
        hand_counts: 手牌計數（含摸入張）
        melds: 鳴牌面子列表 [{'type':'pong','tiles':[i,i,i],'from':1}, ...]
        win_tile: 胡的那張牌
        is_tsumo: 是否自摸
        is_menzen: 是否門前清（未鳴牌）
        player_wind: 玩家自風 (0=東,1=南,2=西,3=北)
        round_wind: 場風
        """
        con = TaiScoreCalculator._counts_to_codes(hand_counts)
        exp = TaiScoreCalculator._melds_to_codes(melds)
        win_code = IDX_TO_CODE.get(int(win_tile))
        if win_code is not None:
            dealer_p = TaiScoreCalculator._dealer_relative_name(player_wind, game_state)
            wind_circle = TaiScoreCalculator.WIND_NAMES[int(round_wind) % 4]
            loser = getattr(game_state, "last_discard_player", None)
            dealer_agent = f"player_{int(getattr(game_state, 'dealer_idx', 0))}"
            win_on_dealer = bool(not is_tsumo and loser == dealer_agent and dealer_p != "我")
            ok, app_tai, app_details, _ = run_full_logic(
                con=con,
                exp=exp,
                win_tile=win_code,
                streak=0,
                dealer_p=dealer_p,
                is_zm=is_tsumo,
                win_on_dealer=win_on_dealer,
                f_mode="莊家位置(莊家抓花)",
                dice=0,
                manual_list=[],
                base_tai=3,
                wind_circle=wind_circle,
            )
            if ok and app_tai != "相公":
                return {
                    "tai": int(app_tai),
                    "details": [
                        TaiScoreCalculator._detail_to_tuple(detail)
                        for detail in app_details
                    ],
                }
            return {
                "tai": 0,
                "details": [("相公", 0), *[
                    TaiScoreCalculator._detail_to_tuple(detail)
                    for detail in app_details
                ]],
            }

        tai = 0
        details = []
        all_tiles = list(hand_counts)

        # ── 基本台數 ──
        # 平胡
        if TaiScoreCalculator._is_normal_hand(hand_counts, melds):
            tai += 1
            details.append(("平胡", 1))

        # 門前清（未鳴牌胡牌額外+1）
        if is_menzen and not is_tsumo:
            tai += 1
            details.append(("門前清", 1))

        # 自摸（額外+1台）
        if is_tsumo:
            tai += 1
            details.append(("自摸", 1))

        # ── 特殊牌型 ──
        # 碰碰胡（全刻子）
        if TaiScoreCalculator._is_all_pungs(hand_counts, melds):
            tai += 3
            details.append(("碰碰胡", 3))

        # 清一色（同花色）
        suit = TaiScoreCalculator._get_all_one_suit(hand_counts, melds)
        if suit is not None and suit != 3:
            tai += 5
            details.append(("清一色", 5))

        # 字一色（全字牌）
        if TaiScoreCalculator._is_all_honor(hand_counts, melds):
            tai += 5
            details.append(("字一色", 5))

        # 混一色（一種數字色+字牌）
        if TaiScoreCalculator._is_half_flush(hand_counts, melds):
            tai += 2
            details.append(("混一色", 2))

        # 全帶么九（每組面子含么九）
        if TaiScoreCalculator._is_all_terminal_or_honor(hand_counts, melds):
            tai += 3
            details.append(("全帶么九", 3))

        # 么九牌（混老頭：全么九不含字）
        if TaiScoreCalculator._is_all_terminals(hand_counts, melds):
            tai += 5
            details.append(("清老頭", 5))

        # 大四喜（四組風牌刻子）
        if TaiScoreCalculator._is_big_four_winds(hand_counts, melds):
            tai += 8
            details.append(("大四喜", 8))

        # 小四喜（三組風牌刻子+一眼風牌）
        if TaiScoreCalculator._is_small_four_winds(hand_counts, melds):
            tai += 5
            details.append(("小四喜", 5))

        # 大三元（三組三元牌刻子）
        if TaiScoreCalculator._is_big_three_dragons(hand_counts, melds):
            tai += 5
            details.append(("大三元", 5))

        # 小三元（兩組三元刻子+一眼三元）
        if TaiScoreCalculator._is_small_three_dragons(hand_counts, melds):
            tai += 3
            details.append(("小三元", 3))

        # 十三么已移除

        # 四暗刻（四組暗刻）
        if TaiScoreCalculator._is_four_concealed_pungs(hand_counts, melds, is_tsumo):
            tai += 8
            details.append(("四暗刻", 8))

        # ── 花色台數 ──
        # 三元牌（中發白刻子各+1）
        for t, name in [(CHUN,"中"), (HATSU,"發"), (HAKU,"白")]:
            if TaiScoreCalculator._has_triplet(hand_counts, melds, t):
                tai += 1
                details.append((f"三元牌({name})", 1))

        # 自風刻子
        wind_tiles = [EAST, SOUTH, WEST, NORTH]
        pw_tile = wind_tiles[player_wind % 4]
        if TaiScoreCalculator._has_triplet(hand_counts, melds, pw_tile):
            tai += 1
            details.append(("自風刻子", 1))

        # 場風刻子
        rw_tile = wind_tiles[round_wind % 4]
        if rw_tile != pw_tile and TaiScoreCalculator._has_triplet(hand_counts, melds, rw_tile):
            tai += 1
            details.append(("場風刻子", 1))

        # 槓牌台數（每個槓+1）
        kong_count = sum(1 for m in melds if m['type'] in ('dark_kong','open_kong','add_kong'))
        if kong_count > 0:
            tai += kong_count
            details.append((f"槓牌×{kong_count}", kong_count))

        # 門前摸（門清自摸）
        if is_menzen and is_tsumo:
            tai += 1
            details.append(("門清自摸加成", 1))

        return {"tai": tai, "details": details}

    # ── 輔助判斷 ──

    @staticmethod
    def _is_normal_hand(hand_counts, melds):
        return True

    @staticmethod
    def _is_all_pungs(hand_counts, melds):
        for m in melds:
            if m['type'] not in ('pong','dark_kong','open_kong','add_kong'):
                return False
        # 手牌中只能有刻子+眼
        c = list(hand_counts)
        eye_found = False
        for i in range(NUM_TILE_TYPES):
            if c[i] == 4 or c[i] == 3:
                c[i] = 0
            elif c[i] == 2:
                if eye_found: return False
                eye_found = True
                c[i] = 0
            elif c[i] != 0:
                return False
        return eye_found

    @staticmethod
    def _get_all_one_suit(hand_counts, melds):
        suits = set()
        for i in range(NUM_TILE_TYPES):
            if hand_counts[i] > 0:
                suits.add(tile_suit(i))
        for m in melds:
            for t in m['tiles']:
                suits.add(tile_suit(t))
        if len(suits) == 1:
            return list(suits)[0]
        return None

    @staticmethod
    def _is_all_honor(hand_counts, melds):
        for i in range(NUM_TILE_TYPES):
            if hand_counts[i] > 0 and i < HONOR_START:
                return False
        for m in melds:
            for t in m['tiles']:
                if t < HONOR_START: return False
        return True

    @staticmethod
    def _is_half_flush(hand_counts, melds):
        suits = set()
        for i in range(NUM_TILE_TYPES):
            if hand_counts[i] > 0:
                suits.add(tile_suit(i))
        for m in melds:
            for t in m['tiles']:
                suits.add(tile_suit(t))
        if 3 not in suits: return False
        non_honor = suits - {3}
        return len(non_honor) == 1

    @staticmethod
    def _is_all_terminal_or_honor(hand_counts, melds):
        for i in range(NUM_TILE_TYPES):
            if hand_counts[i] > 0 and not is_terminal(i):
                return False
        for m in melds:
            if not any(is_terminal(t) for t in m['tiles']):
                return False
        return True

    @staticmethod
    def _is_all_terminals(hand_counts, melds):
        for i in range(NUM_TILE_TYPES):
            if hand_counts[i] > 0 and (is_honor(i) or not is_terminal(i)):
                return False
        for m in melds:
            for t in m['tiles']:
                if is_honor(t) or not is_terminal(t): return False
        return True

    @staticmethod
    def _has_triplet(hand_counts, melds, tile):
        if hand_counts[tile] >= 3: return True
        for m in melds:
            if m['type'] in ('pong','dark_kong','open_kong','add_kong'):
                if m['tiles'][0] == tile: return True
        return False

    @staticmethod
    def _is_big_four_winds(hand_counts, melds):
        return all(TaiScoreCalculator._has_triplet(hand_counts, melds, w)
                   for w in [EAST, SOUTH, WEST, NORTH])

    @staticmethod
    def _is_small_four_winds(hand_counts, melds):
        triplet_winds = sum(1 for w in [EAST,SOUTH,WEST,NORTH]
                            if TaiScoreCalculator._has_triplet(hand_counts, melds, w))
        pair_winds = sum(1 for w in [EAST,SOUTH,WEST,NORTH] if hand_counts[w] == 2)
        return triplet_winds == 3 and pair_winds >= 1

    @staticmethod
    def _is_big_three_dragons(hand_counts, melds):
        return all(TaiScoreCalculator._has_triplet(hand_counts, melds, d)
                   for d in [CHUN, HATSU, HAKU])

    @staticmethod
    def _is_small_three_dragons(hand_counts, melds):
        triplets = sum(1 for d in [CHUN,HATSU,HAKU]
                       if TaiScoreCalculator._has_triplet(hand_counts, melds, d))
        pairs = sum(1 for d in [CHUN,HATSU,HAKU] if hand_counts[d] == 2)
        return triplets == 2 and pairs >= 1

    # _is_thirteen_orphans 已移除

    @staticmethod
    def _is_four_concealed_pungs(hand_counts, melds, is_tsumo):
        open_melds = [m for m in melds if m['type'] != 'dark_kong']
        if open_melds: return False
        triplets = sum(1 for c in hand_counts if c >= 3)
        return triplets >= 4

# ─────────────────────────────────────────
# 防守工具
# ─────────────────────────────────────────

def estimate_tile_danger(tile: int, all_discards: list, ponged_tiles: list,
                          wall_remaining: int) -> float:
    """
    估算打出某牌的放槍危險度（0=安全, 1=危險）
    """
    if tile in ponged_tiles:
        return 0.05  # 有人碰過 → 只剩1張，幾乎安全

    seen_count = all_discards.count(tile)
    remaining = 4 - seen_count  # 場上最多還有幾張

    if remaining <= 0:
        return 0.0  # 已全數可見，最安全

    danger = remaining / 4.0
    # 牌局後期放大危險度
    if wall_remaining < 20:
        danger *= (1.5 if wall_remaining < 10 else 1.2)

    return min(danger, 1.0)

def get_safe_tiles(hand_counts: np.ndarray, all_discards: list, ponged_tiles: list,
                   wall_remaining: int) -> list:
    """回傳手牌中按安全度排序的牌（最安全在前）"""
    tiles = [i for i in range(NUM_TILE_TYPES) if hand_counts[i] > 0]
    tiles.sort(key=lambda t: estimate_tile_danger(t, all_discards, ponged_tiles, wall_remaining))
    return tiles

# ─────────────────────────────────────────
# 主環境類別
# ─────────────────────────────────────────

class TaiwanMahjongEnv(AECEnv):
    """
    台灣16張麻將環境 (PettingZoo AECEnv)
    
    動作空間 (41個動作):
        0-33: 打出第 i 種牌
        34: 吃（上家打出的牌，湊順子）
        35: 碰（任意家打出的牌，湊刻子）
        36: 明槓（任意家打出的牌，手中有3張）
        37: 暗槓（手中有4張相同牌）
        38: 加槓（已碰的刻子+自摸同張）
        39: 胡牌
        40: 過（跳過鳴牌/胡牌機會）
    
    觀察空間 [8 × 34]:
        0-3:   已知牌（自己手牌＋自己鳴牌＋所有棄牌＋場上公開鳴牌）
        4-7:   未知牌（每種牌最多 4 張 - 已知數量）

    手牌張數規則（設 m = 已鳴牌面子數）：
        Draw Phase  （摸牌後，CNN 決策時）: 手牌 = 3(5-m)+2 張
            m=0: 17張 | m=1(碰): 14張 | m=2: 11張 | m=3: 8張
        捨牌後 / 聽牌等待狀態            : 手牌 = 3(5-m)+1 張
            m=0: 16張 | m=1(碰): 13張 | m=2: 10張 | m=3: 7張

        ※ CNN 模型的觀察與動作發生在 Draw Phase（3n+2 張狀態）。
        ※ 若送入 3n+1 張（捨牌後狀態），因分佈偏移會影響模型精度。
    """

    metadata = {"render_modes": ["ansi"], "name": "tw_mahjong_v1", "is_parallelizable": True}

    BASE_SCORE    = 1000  # 底分
    TAI_SCORE     = 500   # 每台分數
    TSUMO_MULT    = 3     # 自摸三倍效果（三家各付）

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.agents          = ["player_0", "player_1", "player_2", "player_3"]
        self.possible_agents = self.agents[:]

        self.action_spaces = {
            a: Discrete(NUM_ACTIONS) for a in self.possible_agents
        }
        self.observation_spaces = {
            a: Box(low=0, high=4, shape=(OBS_PLANES, NUM_TILE_TYPES), dtype=np.float32)
            for a in self.possible_agents
        }
        self._score_calc = TaiScoreCalculator()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    # ─── reset ───

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.rewards              = {a: 0.0  for a in self.agents}
        self._cumulative_rewards  = {a: 0.0  for a in self.agents}
        self.terminations         = {a: False for a in self.agents}
        self.truncations          = {a: False for a in self.agents}
        self.infos                = {a: {}   for a in self.agents}

        # 建牌山並洗牌
        deck = make_deck()
        random.shuffle(deck)
        self.wall = deck

        # 發牌（每人16張）
        self.hands   = {a: np.zeros(NUM_TILE_TYPES, dtype=np.int32) for a in self.agents}
        self.melds   = {a: [] for a in self.agents}   # 鳴牌面子
        self.discards= {a: [] for a in self.agents}   # 棄牌堆
        self.eaten_discards = {a: [] for a in self.agents}
        self.scores  = {a: 0  for a in self.agents}   # 本局得分

        for a in self.agents:
            for _ in range(HAND_SIZE):
                t = self.wall.pop()
                self.hands[a][t] += 1

        # 遊戲狀態
        self.round_wind   = 0      # 0=東場
        self.dealer_idx   = 0      # 莊家索引
        self.turn_idx     = 0      # 目前輪到的玩家索引
        self.last_discard = None   # 上一張打出的牌
        self.last_discard_player = None
        self.phase        = "draw" # draw / discard_after_claim / claim
        self.claim_tile   = None   # 正在詢問的牌
        self.claim_responses = {}  # 各玩家的回應
        self.pending_claims = []
        self.forbidden_discard_after_claim = {a: None for a in self.agents}
        self.selected_chi_option_by_agent = {}
        self.drawn_tile   = None   # 剛摸到的牌
        self.num_moves    = 0
        self.game_over    = False
        self.player_prev_deficiency = {
            a: calc_deficiency(self.hands[a], len(self.melds[a]))
            for a in self.agents
        }

        # 自動摸牌給莊家
        self._draw_tile(self.possible_agents[self.dealer_idx])

        self._selector = agent_selector(self.agents)
        self.agent_selection = self.possible_agents[self.turn_idx]

        return self.observe(self.agent_selection), {}

    # ─── 合法動作遮蔽 ───

    def action_masks(self, agent: str = None) -> np.ndarray:
        if agent is None:
            agent = self.agent_selection
        mask = np.zeros(NUM_ACTIONS, dtype=bool)

        if self.phase == "discard_after_claim":
            # 吃/碰後不得摸牌、胡牌或槓牌，只能從現有手牌中打一張。
            forbidden_tile = self.forbidden_discard_after_claim.get(agent)
            for t in range(NUM_TILE_TYPES):
                if self.hands[agent][t] > 0 and t != forbidden_tile:
                    mask[t] = True

        elif self.phase == "draw":
            # 自己摸牌後：可打牌或暗槓/加槓/胡
            for t in range(NUM_TILE_TYPES):
                if self.hands[agent][t] > 0:
                    mask[t] = True
            # 暗槓
            for t in range(NUM_TILE_TYPES):
                if self.hands[agent][t] >= 4:
                    mask[ACTION_DARK_KONG] = True
            # 加槓（已碰的牌自摸到第4張）
            ponged = {m['tiles'][0] for m in self.melds[agent] if m['type'] == 'pong'}
            for t in ponged:
                if self.hands[agent][t] >= 1:
                    mask[ACTION_ADD_KONG] = True
            # 自摸胡
            total = int(self.hands[agent].sum())
            melds_count = len(self.melds[agent])
            expected_draw_total = 3 * (5 - melds_count) + 2
            if total == expected_draw_total and is_winning_hand(self.hands[agent], melds_count):
                mask[ACTION_HU] = True

        elif self.phase == "claim":
            # 別人打牌後：可吃/碰/槓/胡/過
            tile = self.claim_tile
            player_idx = self.possible_agents.index(agent)
            discard_player_idx = self.possible_agents.index(self.last_discard_player)

            # 胡（放炮胡）
            test = self.hands[agent].copy()
            test[tile] += 1
            if is_winning_hand(test, len(self.melds[agent])):
                mask[ACTION_HU] = True

            # 吃（只能吃上家，也就是 discard_player 是自己的上家）
            upper_idx = (player_idx - 1) % 4
            if discard_player_idx == upper_idx:
                for seq in self._get_chi_options(agent, tile):
                    mask[ACTION_CHI] = True
                    break

            # 碰（任意家）
            if discard_player_idx != player_idx and self.hands[agent][tile] >= 2:
                mask[ACTION_PONG] = True

            # 明槓（任意家）
            if discard_player_idx != player_idx and self.hands[agent][tile] >= 3:
                mask[ACTION_OPEN_KONG] = True

            # 過
            mask[ACTION_PASS] = True

        return mask

    # ─── step ───

    def step(self, action: int):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self.rewards = {a: 0.0 for a in self.agents}

        # 合法動作檢查
        legal = self.action_masks(agent)
        if not legal[action]:
            self.rewards[agent] = -2.0
            for a in self.possible_agents:
                self.terminations[a] = True
            self._accumulate_rewards()
            self.game_over = True
            return

        if self.phase in ("draw", "discard_after_claim"):
            self._handle_draw_phase(agent, action)
        elif self.phase == "claim":
            self._handle_claim_phase(agent, action)

        self.num_moves += 1

        # 荒牌判斷
        if len(self.wall) == 0 and self.phase != "claim":
            self._handle_exhausted_wall()
            return

        self._accumulate_rewards()

    def _get_chi_options(self, agent: str, tile: int) -> list:
        """回傳所有合法的吃牌組合（(partner1, partner2) 清單）"""
        if is_honor(tile):
            return []
        options = []
        s = tile_suit(tile)
        r = tile_rank(tile)
        base = tile - r
        for seq in [(-2,-1), (-1,1), (1,2)]:
            r1 = r + seq[0]
            r2 = r + seq[1]
            if 0 <= r1 <= 8 and 0 <= r2 <= 8:
                p1 = base + r1
                p2 = base + r2
                if (tile_suit(p1) == s and tile_suit(p2) == s and
                        self.hands[agent][p1] >= 1 and self.hands[agent][p2] >= 1):
                    options.append((p1, p2))
        return options

    def _get_opponents(self, agent: str) -> list:
        """回傳其他3名玩家（以相對位置：下家/對家/上家）"""
        idx = self.possible_agents.index(agent)
        return [self.possible_agents[(idx + i) % 4] for i in range(1, 4)]

    def _resolve_win(self, winner: str, win_tile: int, is_tsumo: bool):
        """處理胡牌：計算台分並分配獎勵"""
        idx = self.possible_agents.index(winner)
        is_menzen = len(self.melds[winner]) == 0
        player_wind = idx % 4
        round_wind  = self.round_wind % 4

        # 若是放炮胡，把牌加進手牌計算（計算後移除）
        if not is_tsumo:
            self.hands[winner][win_tile] += 1

        result = TaiScoreCalculator.calc_score(
            hand_counts  = self.hands[winner],
            melds        = self.melds[winner],
            win_tile     = win_tile,
            is_tsumo     = is_tsumo,
            is_menzen    = is_menzen,
            player_wind  = player_wind,
            round_wind   = round_wind,
            game_state   = self,
        )

        if not is_tsumo:
            self.hands[winner][win_tile] -= 1

        tai = result["tai"]
        pay = self.BASE_SCORE + tai * self.TAI_SCORE

        if is_tsumo:
            # 三家各付
            loser = None
            total_gain = 0
            for loser in self.agents:
                if loser != winner:
                    self.scores[loser] -= pay
                    total_gain += pay
            self.scores[winner] += total_gain
            self.rewards[winner] = tai * self.TSUMO_MULT
        else:
            # 放炮者付
            loser = self.last_discard_player
            self.scores[loser]  -= pay
            self.scores[winner] += pay
            self.rewards[winner] = tai * 1.0
            self.rewards[loser]  = 0.0

        self.infos[winner] = {
            "win": True, "tai": tai, "tsumo": is_tsumo,
            "details": result["details"], "pay": pay,
            "loser": loser if not is_tsumo else None,
        }

        for a in self.agents:
            self.terminations[a] = True
        self.game_over = True
        self._accumulate_rewards()

    def _handle_exhausted_wall(self):
        """荒牌：無人胡牌，平局"""
        for a in self.agents:
            penalty, final_deficiency = compute_exhausted_deficiency_penalty(
                self.hands[a],
                len(self.melds[a]),
            )
            self.truncations[a] = True
            self.rewards[a] = float(penalty)
            self.infos[a] = {
                "win": False,
                "exhausted": True,
                "exhausted_deficiency_penalty": float(penalty),
                "final_deficiency": int(final_deficiency),
            }
        self._accumulate_rewards()

    def _advance_agent(self):
        """推進到下一位玩家"""
        if not any(self.terminations.values()) and not any(self.truncations.values()):
            self.turn_idx = (self.turn_idx + 1) % 4
            self.agent_selection = self.possible_agents[self.turn_idx]

    # ─── render ───

    def observe(self, agent: str) -> np.ndarray:
        obs = np.zeros((OBS_PLANES, NUM_TILE_TYPES), dtype=np.float32)

        known_visible = self.hands[agent].copy()
        for a in self.agents:
            for t in self.discards[a]:
                known_visible[t] += 1
            for m in self.melds[a]:
                if a != agent and m["type"] == "dark_kong":
                    continue
                for t in m["tiles"]:
                    known_visible[t] += 1

        known_visible = np.minimum(known_visible, 4)
        unseen = np.maximum(0, 4 - known_visible)

        for k in range(4):
            obs[k] = (known_visible > k).astype(np.float32)
            obs[4 + k] = (unseen > k).astype(np.float32)

        return obs

    def state(self) -> np.ndarray:
        return np.stack([self.observe(a) for a in self.agents])

    def _refresh_player_deficiency(self, agent: str):
        self.player_prev_deficiency[agent] = calc_deficiency(
            self.hands[agent], len(self.melds[agent])
        )

    def _consume_last_discard(self, tile: int):
        src = self.last_discard_player
        if src is None:
            return
        if self.discards[src] and self.discards[src][-1] == tile:
            self.discards[src].pop()
        self.eaten_discards[src].append(tile)

    def _clear_claim_state(self):
        self.claim_tile = None
        self.claim_responses = {}
        self.pending_claims = []
        self.selected_chi_option_by_agent = {}

    def _resolve_claim_priority(self):
        priorities = {
            ACTION_HU: 0,
            ACTION_OPEN_KONG: 1,
            ACTION_PONG: 1,
            ACTION_CHI: 2,
        }
        candidates = []
        for order_idx, claimant in enumerate(self.pending_claims):
            action = self.claim_responses.get(claimant, ACTION_PASS)
            if action == ACTION_PASS:
                continue
            candidates.append((priorities.get(action, 99), order_idx, claimant, action))
        if not candidates:
            return None, None
        _, _, claimant, action = min(candidates)
        return claimant, action

    def _finalize_no_claim(self):
        discard_idx = self.possible_agents.index(self.last_discard_player)
        next_idx = (discard_idx + 1) % 4
        next_agent = self.possible_agents[next_idx]
        self.turn_idx = next_idx
        self.phase = "draw"
        self.agent_selection = next_agent
        self._clear_claim_state()
        self._draw_tile(next_agent)

    def _commit_claim(self, agent: str, action: int, tile: int):
        if action == ACTION_HU:
            self._clear_claim_state()
            self._resolve_win(agent, tile, is_tsumo=False)
            return

        selected_chi_option = int(getattr(self, "selected_chi_option_by_agent", {}).get(agent, 0))
        self._consume_last_discard(tile)
        self.turn_idx = self.possible_agents.index(agent)
        self.agent_selection = agent
        self._clear_claim_state()

        if action == ACTION_PONG:
            self.phase = "discard_after_claim"
            self.forbidden_discard_after_claim[agent] = tile
            self.hands[agent][tile] -= 2
            self.melds[agent].append({
                "type": "pong",
                "tiles": [tile, tile, tile],
                "claim_tile": tile,
                "from": self.possible_agents.index(self.last_discard_player),
            })
            self.rewards[agent] += 0.02
            self._refresh_player_deficiency(agent)
            return

        if action == ACTION_OPEN_KONG:
            self.phase = "draw"
            self.forbidden_discard_after_claim[agent] = None
            self.hands[agent][tile] -= 3
            self.melds[agent].append({
                "type": "open_kong",
                "tiles": [tile, tile, tile, tile],
                "claim_tile": tile,
                "from": self.possible_agents.index(self.last_discard_player),
            })
            self.rewards[agent] += 0.02
            self._draw_tile(agent)
            return

        if action == ACTION_CHI:
            self.phase = "discard_after_claim"
            opts = self._get_chi_options(agent, tile)
            if not opts:
                self.rewards[agent] -= 0.5
                self._finalize_no_claim()
                return
            option_index = min(max(selected_chi_option, 0), len(opts) - 1)
            partner1, partner2 = opts[option_index]
            self.hands[agent][partner1] -= 1
            self.hands[agent][partner2] -= 1
            self.forbidden_discard_after_claim[agent] = tile
            self.melds[agent].append({
                "type": "chi",
                "tiles": sorted([tile, partner1, partner2]),
                "claim_tile": tile,
                "from": self.possible_agents.index(self.last_discard_player),
            })
            self.rewards[agent] += 0.01
            self._refresh_player_deficiency(agent)

    def _resolve_claim_round(self):
        agent, action = self._resolve_claim_priority()
        if agent is None:
            self._finalize_no_claim()
            return
        tile = self.claim_tile
        self._commit_claim(agent, action, tile)

    def _handle_draw_phase(self, agent: str, action: int):
        if action == ACTION_HU:
            self._resolve_win(agent, self.drawn_tile, is_tsumo=True)
            return

        if action == ACTION_DARK_KONG:
            for t in range(NUM_TILE_TYPES):
                if self.hands[agent][t] >= 4:
                    self.hands[agent][t] -= 4
                    self.melds[agent].append(
                        {"type": "dark_kong", "tiles": [t, t, t, t], "from": None}
                    )
                    self.rewards[agent] += 0.02
                    break
            self._draw_tile(agent)
            return

        if action == ACTION_ADD_KONG:
            for m in self.melds[agent]:
                if m["type"] == "pong" and self.hands[agent][m["tiles"][0]] >= 1:
                    t = m["tiles"][0]
                    self.hands[agent][t] -= 1
                    m["type"] = "add_kong"
                    m["tiles"].append(t)
                    self.rewards[agent] += 0.02
                    break
            self._draw_tile(agent)
            return

        if action < NUM_TILE_TYPES:
            tile = action
            if self.hands[agent][tile] <= 0:
                self.rewards[agent] = -2.0
                for a in self.possible_agents:
                    self.terminations[a] = True
                self.game_over = True
                return

            prev_deficiency = self.player_prev_deficiency.get(agent)
            self.hands[agent][tile] -= 1
            self.forbidden_discard_after_claim[agent] = None
            self.discards[agent].append(tile)
            self.last_discard = tile
            self.last_discard_player = agent

            d_after = calc_deficiency(self.hands[agent], len(self.melds[agent]))
            if prev_deficiency is not None:
                if d_after < prev_deficiency:
                    self.rewards[agent] += 0.05
                elif d_after > prev_deficiency:
                    self.rewards[agent] -= 0.05
            self.player_prev_deficiency[agent] = d_after
            self._start_claim_phase(tile, agent)

    def _handle_claim_phase(self, agent: str, action: int):
        self.claim_responses[agent] = ACTION_PASS if action == ACTION_PASS else action
        self._check_claim_resolved()

    def _draw_tile(self, agent: str):
        if self.wall:
            t = self.wall.pop()
            self.hands[agent][t] += 1
            self.drawn_tile = t
        else:
            self.drawn_tile = None
        self._refresh_player_deficiency(agent)

    def _start_claim_phase(self, tile: int, discard_agent: str):
        self.phase = "claim"
        self.claim_tile = tile
        self.last_discard = tile
        self.last_discard_player = discard_agent
        self.claim_responses = {}
        discard_idx = self.possible_agents.index(discard_agent)
        self.pending_claims = [
            self.possible_agents[(discard_idx + i) % 4]
            for i in range(1, 4)
        ]
        self.agent_selection = self.pending_claims[0]

    def _check_claim_resolved(self):
        for claimant in self.pending_claims:
            if claimant not in self.claim_responses:
                self.agent_selection = claimant
                return
        self._resolve_claim_round()

    def render(self):
        if self.render_mode == "ansi":
            for a in self.agents:
                tiles = [tile_name(i) for i in range(NUM_TILE_TYPES)
                         for _ in range(self.hands[a][i])]
                print(f"{a}: {''.join(tiles)} | 棄牌: {''.join(tile_name(t) for t in self.discards[a])}")
            print(f"牌山剩餘: {len(self.wall)} 張")

    def close(self):
        pass


# ─────────────────────────────────────────
# 工廠函式
# ─────────────────────────────────────────

def env(**kwargs):
    environment = TaiwanMahjongEnv(**kwargs)
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment


# ─────────────────────────────────────────
# 快速測試
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=== 台灣麻將環境自我測試 ===")

    # 測試胡牌判斷
    hand = np.zeros(NUM_TILE_TYPES, dtype=np.int32)
    # 手牌：123萬 456萬 789萬 123筒 11筒（聽牌）
    for t in [0,1,2, 3,4,5, 6,7,8, 9,10,11, 9,9]:
        hand[t] += 1
    # 再加一張9筒（第3張）
    hand[9] += 1
    print("胡牌測試:", is_winning_hand(hand, 0))  # 應為 True
    print("進胡數:", calc_deficiency(hand, 0))      # 應為 -1（已胡）

    # 測試環境建立
    e = TaiwanMahjongEnv(render_mode="ansi")
    obs, _ = e.reset(seed=42)
    print(f"\n觀察空間形狀: {obs.shape}")
    e.render()

    # 測試合法動作
    agent = e.agent_selection
    masks = e.action_masks(agent)
    legal = np.where(masks)[0]
    print(f"\n{agent} 合法動作數: {len(legal)}")
    print(f"合法動作: {legal}")

    # 測試台分計算
    win_hand = np.zeros(NUM_TILE_TYPES, dtype=np.int32)
    for t in [0,0,0, 1,2,3, 4,5,6, 7,8,0, 9,9]:
        win_hand[t] += 1
    win_hand[9] += 1
    result = TaiScoreCalculator.calc_score(
        win_hand, [], 9, is_tsumo=True, is_menzen=True,
        player_wind=0, round_wind=0
    )
    print(f"\n台分計算: {result['tai']} 台")
    for name, pts in result['details']:
        print(f"  {name}: +{pts}")

    print("\n=== 測試完成 ===")
