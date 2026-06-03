import collections
from core.config import TILE_INFO

def recursive_decompose_main(counts, sets_needed, win_tile, current_sets=None):
    if current_sets is None: current_sets = []
    if sum(counts.values()) == 0: return (sets_needed == 0), current_sets
    if sets_needed <= 0: return False, []
    tile = next(k for k, v in sorted(counts.items(), key=lambda x: TILE_INFO[x[0]]['w']) if v > 0)
    for take in [4, 3]:
        if counts[tile] >= take:
            temp = counts.copy(); temp[tile] -= take
            ok, res = recursive_decompose_main(temp, sets_needed - 1, win_tile, current_sets + [(f'set_{take}', tile)])
            if ok: return True, res
    info = TILE_INFO[tile]
    if info['type'] in ['w', 'D', 's'] and info.get('val', 0) <= 7:
        t2 = next((k for k,v in TILE_INFO.items() if v.get('type')==info['type'] and v.get('val')==info['val']+1), None)
        t3 = next((k for k,v in TILE_INFO.items() if v.get('type')==info['type'] and v.get('val')==info['val']+2), None)
        if t2 and t3 and counts.get(t2,0) > 0 and counts.get(t3,0) > 0:
            temp = counts.copy(); temp[tile]-=1; temp[t2]-=1; temp[t3]-=1
            seq = [tile, t2, t3]; pos = seq.index(win_tile) if win_tile in seq else -1
            ok, res = recursive_decompose_main(temp, sets_needed - 1, win_tile, current_sets + [('seq', seq, pos)])
            if ok: return True, res
    return False, []

def recursive_decompose_waiting(counts, sets_needed):
    if sum(counts.values()) == 0: return (sets_needed == 0)
    if sets_needed <= 0: return False
    tile = next(k for k, v in sorted(counts.items(), key=lambda x: TILE_INFO[x[0]]['w']) if v > 0)
    if counts[tile] >= 3:
        temp = counts.copy(); temp[tile] -= 3
        if recursive_decompose_waiting(temp, sets_needed - 1): return True
    info = TILE_INFO[tile]
    if info['type'] in ['w', 'D', 's'] and info.get('val', 0) <= 7:
        t2 = next((k for k,v in TILE_INFO.items() if v.get('type')==info['type'] and v.get('val')==info['val']+1), None)
        t3 = next((k for k,v in TILE_INFO.items() if v.get('type')==info['type'] and v.get('val')==info['val']+2), None)
        if t2 and t3 and counts.get(t2,0) > 0 and counts.get(t3,0) > 0:
            temp = counts.copy(); temp[tile]-=1; temp[t2]-=1; temp[t3]-=1
            if recursive_decompose_waiting(temp, sets_needed - 1): return True
    return False

def check_hu_for_waiting(counts):
    for eye in counts:
        if counts[eye] >= 2:
            temp = counts.copy(); temp[eye] -= 2
            rem_tiles = sum(temp.values())
            if rem_tiles % 3 != 0: continue
            sets_needed = rem_tiles // 3
            if recursive_decompose_waiting(temp, sets_needed): return True
    return False

def get_waiting_tiles(hand_codes):
    counts = collections.Counter(hand_codes)
    waiting = []
    all_tiles = [k for k,v in TILE_INFO.items() if v['type'] != 'h']
    for t in all_tiles:
        temp = counts.copy()
        temp[t] += 1
        if temp[t] > 4: continue
        if check_hu_for_waiting(temp):
            waiting.append(t)
    return waiting

def analyze_waiting_status(con):
    hand_only = [c for c in con if TILE_INFO[c]['type'] != 'h']
    total_counts = collections.Counter(con)
    for code, count in total_counts.items():
        info = TILE_INFO[code]
        if info['type'] != 'h' and count > 4:
            return "error", f"牌數錯誤：**{info['name']}** 有 {count} 張 (單一牌種上限為 4)", []
    hand_len = len(hand_only)
    if hand_len > 16:
        return "error", f"手牌數量為 {hand_len} 張。<br>手牌上限限制為 n=5 (最多 16 張)。", []
    if hand_len % 3 == 0:
        return "error", f"手牌數量為 {hand_len} 張 (相公)。<br>若要聽牌，手牌應為 3n+1 張。", []
    elif hand_len % 3 == 2:
        return "error", f"手牌數量為 {hand_len} 張 (3n+2)。<br>這是已經胡牌或未打牌的數量，請移除一張多餘的牌以計算聽牌。", []
    waiting_list = get_waiting_tiles(hand_only)
    if waiting_list: return "waiting", "聽牌中！", waiting_list
    else: return "not_waiting", "尚未聽牌", []

def run_full_logic(con, exp, win_tile, streak, dealer_p, is_zm, win_on_dealer, f_mode, dice, manual_list, base_tai, wind_circle):
    all_codes = con + exp
    hand_only = [c for c in all_codes if TILE_INFO[c]['type'] != 'h']
    hua_codes = [c for c in all_codes if TILE_INFO[c]['type'] == 'h']
    total_counts = collections.Counter(all_codes)
    for code, count in total_counts.items():
        info = TILE_INFO[code]
        limit = 1 if info['type'] == 'h' else 4
        if count > limit: return False, "相公", [f"偵測到 **{info['name']}** 有 {count} 張 (上限 {limit})"], None
    if any(TILE_INFO[c]['type'] == 'h' for c in con):
        return False, "相公", ["手牌區不可含花牌"], None
    all_counts = collections.Counter(hand_only)
    hu_ok, best_sets, win_is_eye = False, [], False
    for eye, count in all_counts.items():
        if count >= 2:
            temp = all_counts.copy(); temp[eye] -= 2
            ok, res = recursive_decompose_main(temp, 5, win_tile)
            if ok: hu_ok = True; best_sets = res; win_is_eye = (eye == win_tile); break
    if not hu_ok: return False, "相公", ["結構錯誤 (無法湊成5面子+1眼)"], None
    hand_minus_win = list([c for c in con if TILE_INFO[c]['type'] != 'h'])
    if win_tile in hand_minus_win: hand_minus_win.remove(win_tile)
    waiting_list = get_waiting_tiles(hand_minus_win)
    is_strict_single_wait = (len(waiting_list) == 1 and waiting_list[0] == win_tile)
    tai, details = 0, []
    n_counts = collections.Counter([TILE_INFO[c]['name'] for c in hand_only])
    suits = set([TILE_INFO[c]['type'] for c in hand_only])
    if all(t == 'z' for t in suits): tai += 16; details.append("字一色 16台")
    elif len(suits - {'z'}) == 1:
        if 'z' in suits: tai += 4; details.append("混一色 4台")
        else: tai += 8; details.append("清一色 8台")

    d_tri = sum(1 for d in ['中','發','白'] if n_counts[d] >= 3)
    d_pair = sum(1 for d in ['中','發','白'] if n_counts[d] == 2)
    if d_tri == 3: tai += 8; details.append("大三元 8台")
    elif d_tri == 2 and d_pair == 1: tai += 4; details.append("小三元 4台")
    else:
        if n_counts['中'] >= 3: tai += 1; details.append("紅中 1台")
        if n_counts['發'] >= 3: tai += 1; details.append("發財 1台")
        if n_counts['白'] >= 3: tai += 1; details.append("白板 1台")

    w_tri = sum(1 for w in ['東','南','西','北'] if n_counts[w] >= 3)
    is_big_four = (w_tri == 4)
    if is_big_four: tai += 16; details.append("大四喜 16台 (不加計圈風與字牌門風)")
    elif w_tri == 3 and any(n_counts[w] == 2 for w in ['東','南','西','北']): tai += 8; details.append("小四喜 8台")

    if all(s[0].startswith('set') for s in best_sets): tai += 4; details.append("碰碰胡 4台")

    con_hand = [c for c in con if TILE_INFO[c]['type'] != 'h']
    is_quan_qiu = (len(con_hand) == 2 and win_is_eye)
    wait_type = None
    if not is_quan_qiu:
        if is_strict_single_wait and win_is_eye: wait_type = "單吊 1台"
        elif is_strict_single_wait:
            for s in best_sets:
                if s[0] == 'seq' and s[2] != -1:
                    v = TILE_INFO[win_tile]['val']
                    if s[2] == 1: wait_type = "中洞 1台"
                    elif (s[2] == 0 and v == 7) or (s[2] == 2 and v == 3): wait_type = "邊張 1台"
                    else: wait_type = "單吊 1台"
    if wait_type: tai += 1; details.append(wait_type)
    anke_count = 0
    exposed_counts = collections.Counter(exp)
    for s in best_sets:
        ctype = s[0]
        if ctype in ['set_3', 'set_4']:
            tile = s[1]
            if exposed_counts[tile] >= 3: exposed_counts[tile] -= 3
            else: anke_count += 1
    if anke_count >= 3:
        tm={3:2, 4:5, 5:8}; tai += tm.get(anke_count, 0); details.append(f"{anke_count}暗刻 {tm.get(anke_count,0)}台")

    is_menqing = (len(exp) == 0)
    if is_quan_qiu: tai += 2; details.append("全求人 2台 (含單吊)")
    elif is_menqing:
        if is_zm and "槓上開花" not in manual_list and "海底撈月" not in manual_list: tai += 3; details.append("門清一摸三 3台")
        elif not is_zm: tai += 1; details.append("門清 1台")
        elif (is_zm and ("槓上開花" in manual_list or "海底撈月" in manual_list)): tai += 1; details.append("門清 1台")

    if len(hua_codes)==0 and not any(TILE_INFO[c]['type']=='z' for c in hand_only) and all(s[0]=='seq' for s in best_sets) and not is_menqing and not wait_type:
        tai += 2; details.append("平胡 2台")

    dealer_map_idx = {"我": 0, "下家(右)": 1, "對家(對面)": 2, "上家(左)": 3}
    dealer_idx_rel = dealer_map_idx[dealer_p]
    if f_mode.startswith("莊家"):
        logical_east_idx_rel = dealer_idx_rel
        calc_note = "莊家位置"
    else:
        dice_offset = (dice - 1) % 4
        logical_east_idx_rel = (dealer_idx_rel + dice_offset) % 4
        calc_note = f"骰子{dice}點開門位置"

    my_wind_idx = (4 - logical_east_idx_rel) % 4
    wind_names = ["東", "南", "西", "北"]
    my_wind_name = wind_names[my_wind_idx]
    my_flower_num = my_wind_idx + 1
    wind_debug_info = f"判斷基準：{calc_note} <br> 我的門風：<b>{my_wind_name}風</b> (對應花牌：{my_flower_num}花)"

    if not is_big_four:
        if n_counts[wind_circle] >= 3: tai += 1; details.append(f"圈風({wind_circle}風) 1台")
        if n_counts[my_wind_name] >= 3: tai += 1; details.append(f"門風({my_wind_name}風) 1台")

    if len(hua_codes) == 8: tai += 8; details.append("八仙過海 8台")
    elif len(hua_codes) == 7: tai += 7; details.append("七搶一 7台")
    else:
        h_suits = collections.Counter([TILE_INFO[c]['suit'] for c in hua_codes])
        gang_suits = [s for s, c in h_suits.items() if c == 4]
        for g in gang_suits: tai += 2; details.append(f"花槓 ({'春夏秋冬' if g=='rf' else '梅蘭竹菊'}) 2台")
        loose_flowers = [c for c in hua_codes if TILE_INFO[c]['suit'] not in gang_suits]
        for c in loose_flowers:
            if TILE_INFO[c]['v'] == my_flower_num: tai += 1; details.append(f"方位花牌({TILE_INFO[c]['name']}) 1台")

    tai += base_tai; details.insert(0, f"底台 {base_tai}台")
    has_streak = False
    if dealer_p == "我": has_streak = True
    elif is_zm: has_streak = True
    elif win_on_dealer: has_streak = True; tai += 1; details.append("胡莊家 1台")
    if has_streak and streak > 0: tai += (2*streak); details.append(f"連{streak}拉{streak} {2*streak}台")
    if dealer_p == "我": tai += 1; details.append("莊家 1台")

    if "槓上開花" in manual_list: tai += 2; details.append("槓上自摸 2台")
    elif "海底撈月" in manual_list: tai += 2; details.append("海底自摸 2台")
    elif is_zm and not is_menqing: tai += 1; details.append("自摸 1台")

    manual_score_map = {"天胡": 16, "搶槓": 1, "河底撈魚": 1, "咪幾": 8, "天地人胡": 16}
    for m in manual_list:
        if m in ["槓上開花", "海底撈月"]: continue
        pts = manual_score_map.get(m, 0)
        if m == "天胡": pts = 16
        if m in ["搶槓", "河底撈魚", "咪幾", "天地人胡"]:
            real_pts = pts if pts > 0 else 1
            if m == "地人胡": real_pts = 16
            if m == "咪幾": real_pts = 8
            tai += real_pts; details.append(f"{m} {real_pts}台")

    return True, tai, details, wind_debug_info
