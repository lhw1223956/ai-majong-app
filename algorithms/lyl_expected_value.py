from __future__ import annotations

from collections.abc import Iterable


# Ported from 模型訓練lyl/ai/ev.py.  The extra ranking metadata below is only
# for training reward inspection; the hand EV scoring logic follows that file.


SUITS_EV = ["w", "D", "s"]
HONORS_EV = ["ew", "sw", "ww", "nw", "zhong", "fa", "wd"]
ALL_34_TILES_EV = [f"{number}{suit}" for suit in SUITS_EV for number in range(1, 10)] + HONORS_EV
TILE_ORDER_EV = {tile: index for index, tile in enumerate(ALL_34_TILES_EV)}
HONOR_INDEX_EV = {"ew": 1, "sw": 2, "ww": 3, "nw": 4, "zhong": 5, "fa": 6, "wd": 7}
WIN_SCORE_EV = 100_000_000


def _clean_codes(codes: Iterable[str] | None) -> list[str]:
    if codes is None:
        return []
    return [code for code in codes if code in TILE_ORDER_EV]


def tie_score_ev(tile: str) -> int:
    if tile in HONORS_EV:
        return 0
    number = int(tile[0])
    if number in (1, 9):
        return 1
    return 2


def get_max_blocks_from_suit_ev(arr: list[int], is_honor: bool = False) -> list[str]:
    best_score = -1
    best_blocks: list[str] = []

    def dfs(idx: int, current_blocks: list[str], current_arr: list[int]) -> None:
        nonlocal best_score, best_blocks
        if (is_honor and idx > 7) or (not is_honor and idx > 9):
            meld_count = current_blocks.count("m")
            pair_count = current_blocks.count("p")
            tatsu_count = current_blocks.count("t")
            score = meld_count * 100_000 + pair_count * 10_000 + tatsu_count * 1_000
            if score > best_score:
                best_score = score
                best_blocks = list(current_blocks)
            return

        if current_arr[idx] == 0:
            dfs(idx + 1, current_blocks, current_arr)
            return

        if current_arr[idx] >= 3:
            current_arr[idx] -= 3
            current_blocks.append("m")
            dfs(idx, current_blocks, current_arr)
            current_blocks.pop()
            current_arr[idx] += 3

        if (
            not is_honor
            and idx <= 7
            and current_arr[idx] >= 1
            and current_arr[idx + 1] >= 1
            and current_arr[idx + 2] >= 1
        ):
            current_arr[idx] -= 1
            current_arr[idx + 1] -= 1
            current_arr[idx + 2] -= 1
            current_blocks.append("m")
            dfs(idx, current_blocks, current_arr)
            current_blocks.pop()
            current_arr[idx] += 1
            current_arr[idx + 1] += 1
            current_arr[idx + 2] += 1

        if current_arr[idx] >= 2:
            current_arr[idx] -= 2
            current_blocks.append("p")
            dfs(idx, current_blocks, current_arr)
            current_blocks.pop()
            current_arr[idx] += 2

        if not is_honor and idx <= 8 and current_arr[idx] >= 1 and current_arr[idx + 1] >= 1:
            current_arr[idx] -= 1
            current_arr[idx + 1] -= 1
            current_blocks.append("t")
            dfs(idx, current_blocks, current_arr)
            current_blocks.pop()
            current_arr[idx] += 1
            current_arr[idx + 1] += 1

        if not is_honor and idx <= 7 and current_arr[idx] >= 1 and current_arr[idx + 2] >= 1:
            current_arr[idx] -= 1
            current_arr[idx + 2] -= 1
            current_blocks.append("t")
            dfs(idx, current_blocks, current_arr)
            current_blocks.pop()
            current_arr[idx] += 1
            current_arr[idx + 2] += 1

        current_arr[idx] -= 1
        dfs(idx, current_blocks, current_arr)
        current_arr[idx] += 1

    dfs(1, [], list(arr))
    return best_blocks


def score_hand_ev(hand: list[str]) -> int:
    w_arr, d_arr, s_arr, h_arr = [0] * 10, [0] * 10, [0] * 10, [0] * 8
    for tile in hand:
        if tile in HONORS_EV:
            h_arr[HONOR_INDEX_EV[tile]] += 1
        elif tile.endswith("w"):
            w_arr[int(tile[0])] += 1
        elif tile.endswith("D"):
            d_arr[int(tile[0])] += 1
        elif tile.endswith("s"):
            s_arr[int(tile[0])] += 1

    blocks: list[str] = []
    blocks.extend(get_max_blocks_from_suit_ev(w_arr))
    blocks.extend(get_max_blocks_from_suit_ev(d_arr))
    blocks.extend(get_max_blocks_from_suit_ev(s_arr))
    blocks.extend(get_max_blocks_from_suit_ev(h_arr, is_honor=True))

    meld_count = blocks.count("m")
    pair_count = blocks.count("p")
    tatsu_count = blocks.count("t")

    max_melds = len(hand) // 3
    max_blocks = max_melds + 1
    actual_melds = min(meld_count, max_melds)
    remaining_slots = max_blocks - actual_melds

    actual_pairs = 0
    actual_tatsu = 0
    if remaining_slots > 0:
        if pair_count > 0:
            actual_pairs = 1
            remaining_slots -= 1
        leftover_pairs = max(0, pair_count - actual_pairs)
        actual_tatsu = min(tatsu_count + leftover_pairs, remaining_slots)

    if actual_melds == max_melds and actual_pairs == 1:
        return WIN_SCORE_EV
    return actual_melds * 100_000 + actual_pairs * 10_000 + actual_tatsu * 1_000


def build_deck_counts_ev(hand: Iterable[str], visible_tiles: Iterable[str] | None = None) -> dict[str, int]:
    deck_counts = {tile: 4 for tile in ALL_34_TILES_EV}
    for tile in _clean_codes(hand):
        deck_counts[tile] -= 1
    for tile in _clean_codes(visible_tiles):
        if deck_counts.get(tile, 0) > 0:
            deck_counts[tile] -= 1
    return deck_counts


def calculate_ev_detail(hand: list[str], discard: str, deck_counts: dict[str, int]) -> float:
    remaining_hand = list(hand)
    remaining_hand.remove(discard)
    ev = 0.0
    for draw in ALL_34_TILES_EV:
        count = deck_counts.get(draw, 4)
        if count > 0:
            ev += count * score_hand_ev(remaining_hand + [draw])
    return ev


def _normalize_results(results: list[dict]) -> None:
    if not results:
        return
    values = [float(row["ev"]) for row in results]
    ev_min = min(values)
    ev_max = max(values)
    if ev_max == ev_min:
        for row in results:
            row["ev_normalized"] = 1.0
        return
    for row in results:
        row["ev_normalized"] = (float(row["ev"]) - ev_min) / (ev_max - ev_min)


def _annotate_results(results: list[dict], tie_abs_tol: float = 1e-6) -> list[str]:
    if not results:
        return []
    best_ev = float(results[0]["ev"])
    best_tiles = [row["tile"] for row in results if abs(float(row["ev"]) - best_ev) <= tie_abs_tol]
    previous_ev: float | None = None
    rank_group = 0
    competition_rank = 0
    previous_competition_rank = 0
    for position, row in enumerate(results, start=1):
        ev = float(row["ev"])
        if previous_ev is None or abs(ev - previous_ev) > tie_abs_tol:
            rank_group += 1
            competition_rank = position
            previous_ev = ev
            previous_competition_rank = competition_rank
        else:
            competition_rank = previous_competition_rank
        is_best = row["tile"] in best_tiles
        row["is_ev_best"] = is_best
        row["is_ev_best_tied"] = is_best
        row["ev_best_count"] = len(best_tiles)
        row["ev_gap_to_best"] = best_ev - ev
        row["ev_rank"] = competition_rank
        row["ev_rank_group"] = rank_group
    return best_tiles


def get_discard_ev_ranking(
    hand_codes: Iterable[str],
    visible_tiles: Iterable[str] | None = None,
    legal_discards: Iterable[str] | None = None,
    tie_abs_tol: float = 1e-6,
) -> dict:
    hand = _clean_codes(hand_codes)
    legal_set = set(_clean_codes(legal_discards)) if legal_discards is not None else set(hand)
    candidates = [
        tile
        for tile in sorted(set(hand), key=lambda code: TILE_ORDER_EV[code])
        if tile in legal_set
    ]
    if not candidates:
        return {
            "best_tiles": [],
            "results": [],
            "summary": {
                "candidate_count": 0,
                "best_count": 0,
                "all_equal": False,
                "source": "lyl_expected_value",
            },
        }

    hand_counts = {tile: hand.count(tile) for tile in set(hand)}
    forced_honor = next(
        (tile for tile in HONORS_EV if hand_counts.get(tile, 0) == 4 and tile in candidates),
        None,
    )

    deck_counts = build_deck_counts_ev(hand, visible_tiles)
    results = [
        {
            "tile": discard,
            "ev": calculate_ev_detail(hand, discard, deck_counts),
            "tie_score": tie_score_ev(discard),
        }
        for discard in candidates
    ]
    if forced_honor is not None:
        for row in results:
            if row["tile"] == forced_honor:
                row["ev"] = max(float(item["ev"]) for item in results) + 1.0
                break
    results.sort(key=lambda row: (-float(row["ev"]), int(row["tie_score"]), TILE_ORDER_EV[row["tile"]]))
    _normalize_results(results)
    best_tiles = _annotate_results(results, tie_abs_tol=tie_abs_tol)
    ev_values = {float(row["ev"]) for row in results}

    return {
        "best_tiles": best_tiles,
        "results": results,
        "summary": {
            "candidate_count": len(results),
            "best_count": len(best_tiles),
            "all_equal": len(ev_values) == 1,
            "source": "lyl_expected_value",
        },
    }


def get_best_discard_ev(
    hand: Iterable[str],
    visible_tiles: Iterable[str] | None = None,
    legal_discards: Iterable[str] | None = None,
) -> tuple[str | None, list[dict]]:
    ranking = get_discard_ev_ranking(
        hand_codes=hand,
        visible_tiles=visible_tiles,
        legal_discards=legal_discards,
    )
    results = ranking["results"]
    if not results:
        return None, []
    return results[0]["tile"], results


if __name__ == "__main__":
    sample_hand = ["7w", "7w", "8w", "9w", "9w", "2s", "3s", "4s", "6s", "7s", "8s"]
    visible = ["2w", "3w", "4w", "ww", "ww", "ww"]
    best_tile, details = get_best_discard_ev(sample_hand, visible_tiles=visible)
    print(f"best discard: {best_tile}")
    for rank, item in enumerate(details[:5], 1):
        print(f"{rank}. discard {item['tile']} -> EV: {item['ev']:.0f}")
