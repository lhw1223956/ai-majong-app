import google.generativeai as genai
import collections
import os
from pathlib import Path
from core.config import TILE_INFO
from algorithms.lyl_progress_ev_judgement import rank_discards_by_progress_ev


def _load_gemini_api_key_from_env():
    api_key = str(os.environ.get("GEMINI_API_KEY", "")).strip()
    if api_key:
        return api_key, ""
    return "", ""


def _load_gemini_api_key_from_streamlit():
    try:
        import streamlit as st

        api_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        if api_key:
            return api_key, ""
    except Exception:
        pass
    return "", ""


def _parse_toml(text):
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    return tomllib.loads(text)


def _load_gemini_api_key_from_file():
    api_key, key_error = _load_gemini_api_key_from_env()
    if api_key or key_error:
        return api_key, key_error

    api_key, key_error = _load_gemini_api_key_from_streamlit()
    if api_key or key_error:
        return api_key, key_error

    secrets_paths = [
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml",
    ]
    checked_paths = []

    for path in dict.fromkeys(secrets_paths):
        display_path = str(path).replace("\\", "/")
        checked_paths.append(display_path)
        if not path.exists():
            continue

        try:
            data = _parse_toml(path.read_text(encoding="utf-8-sig"))
        except Exception as e:
            return "", f"⚠️ Gemini 金鑰檔案讀取失敗：`{display_path}`，錯誤：{e}"

        api_key = str(data.get("GEMINI_API_KEY", "")).strip()
        if api_key:
            return api_key, ""
        return "", f"⚠️ Gemini 金鑰檔案缺少 `GEMINI_API_KEY`：`{display_path}`"

    checked_text = "、".join(f"`{path}`" for path in checked_paths)
    return "", (
        "⚠️ **Gemini 尚未啟用**：請在 Hugging Face Space 的 Secret 設定 "
        "`GEMINI_API_KEY`，或建立 `.streamlit/secrets.toml` 並設定 "
        f"`GEMINI_API_KEY`。按下 LLM 教練按鈕時才會讀取金鑰。已檢查：{checked_text}"
    )


def get_majiang_coach_advice(hand_codes, exp_codes, discard_pool=None):
    missing_key_message = (
        "⚠️ **Gemini 尚未啟用**：請在 Hugging Face Space 的 Secret 設定 `GEMINI_API_KEY`。"
        "按下 LLM 教練按鈕時才會讀取金鑰。"
    )
    try:
        api_key, key_error = _load_gemini_api_key_from_file()
        if key_error:
            return key_error
        if not api_key:
            return missing_key_message
        genai.configure(api_key=api_key)
    except KeyError:
        return missing_key_message
    except Exception as e:
        return f"⚠️ Gemini 設定失敗：{e}"

    try:
        # 使用 2.5 flash 模型
        llm_model = genai.GenerativeModel('gemini-2.5-flash')
        hand_names = sorted([f"{TILE_INFO[c]['icon']}{TILE_INFO[c]['name']}" for c in hand_codes if c in TILE_INFO])
        exp_names = [f"{TILE_INFO[c]['icon']}{TILE_INFO[c]['name']}" for c in exp_codes if c in TILE_INFO]
        
        # 計算特徵以降低 AI 幻覺，提高建議精確度
        counts = collections.Counter([TILE_INFO[c]['name'] for c in hand_codes if c in TILE_INFO])
        pairs = [k for k, v in counts.items() if v >= 2]
        triplets = [k for k, v in counts.items() if v >= 3]
        stats_info = f"對子數量: {len(pairs)} 組, 刻子/暗槓數量: {len(triplets)} 組"
        
        # 整合所有已知牌（門前牌 + 桌面明牌）用於 EV 計算
        if discard_pool is None:
            discard_pool = []
        hand_for_ev = [c for c in hand_codes if TILE_INFO[c]['type'] != 'h']
        exp_for_ev = [c for c in exp_codes if TILE_INFO[c]['type'] != 'h']
        pool_for_ev = [c for c in discard_pool if c in TILE_INFO and TILE_INFO[c]['type'] != 'h']
        all_visible = exp_for_ev + pool_for_ev
        legal_discards = sorted(set(hand_for_ev), key=lambda code: TILE_INFO[code]["w"])
        fixed_melds = len(exp_for_ev) // 3
        ev_report = rank_discards_by_progress_ev(
            hand_for_ev,
            visible_tiles=all_visible,
            legal_discards=legal_discards,
            fixed_melds=fixed_melds,
        )
        ev_details = ev_report.get("results", [])
        
        ev_info_str = "演算法分析 (lyl_progress_ev_judgement.py，已扣除桌面明牌)：\n"
        if ev_details:
            top_5_ev = ev_details[:5]
            for item in top_5_ev:
                tile_n = TILE_INFO.get(item['tile'], {}).get('name', item['tile'])
                tile_i = TILE_INFO.get(item['tile'], {}).get('icon', '')
                rank = item.get("progress_ev_rank", "-")
                distance = item.get("progress_distance", "-")
                progress = item.get("progress_score", "-")
                ev_value = item.get("ev", 0)
                ev_used = "有啟用EV" if item.get("ev_used", False) else "依進度排序"
                improving = item.get("progress_improving_tiles", 0)
                ev_info_str += (
                    f"- 第{rank}名 打【{tile_i}{tile_n}】: 距離 {distance}, "
                    f"進度 {progress}, EV {ev_value}, 有效進牌 {improving} ({ev_used})\n"
                )
        else:
            ev_info_str += "- 目前無法計算期望值。\n"
        
        # 桌面明牌資訊整理
        pool_names = [f"{TILE_INFO[c]['icon']}{TILE_INFO[c]['name']}" for c in pool_for_ev if c in TILE_INFO]
        pool_counts = collections.Counter([TILE_INFO[c]['name'] for c in pool_for_ev if c in TILE_INFO])
        # 找出已出現 3 張以上（幾乎死掉）的牌
        dead_tiles = [n for n, cnt in pool_counts.items() if cnt >= 3]
        # 找出已出現 2 張的牌（剩張數少）
        scarce_tiles = [n for n, cnt in pool_counts.items() if cnt == 2]
        
        pool_summary = ""
        if pool_names:
            pool_summary = f"- 桌面明牌（已打出/吃碰槓 共 {len(pool_for_ev)} 張）：{', '.join(pool_names[:30])}{'...(太多省略)' if len(pool_names) > 30 else ''}\n"
            if dead_tiles:
                pool_summary += f"- ⚠️ 幾乎死牌（已出3張以上）：{', '.join(dead_tiles)}\n"
            if scarce_tiles:
                pool_summary += f"- 🔶 稀少牌（已出2張）：{', '.join(scarce_tiles)}\n"
        else:
            pool_summary = "- 桌面明牌：玩家未提供（或尚未辨識）\n"
        
        prompt = f"""
        你現在是一位「極度精簡且視角清晰」的台灣 16 張麻將助教。
        
        【牌局狀態】：
        - 未打出的暗牌：{', '.join(hand_names) if hand_names else '無'}
        - 已吃碰的明牌：{', '.join(exp_names) if exp_names else '無'}
        - 牌型狀態：{stats_info}
        
        【桌面資訊（所有玩家已打出/吃碰槓的牌）】：
        {pool_summary}
        {ev_info_str}
        
        【嚴格輸出限制】：
        - 1. **全圖示化：當提到任何麻將牌時，絕對只能輸出麻將圖示 (例如 🀇, 🀈, 🀀)，嚴禁出現中文牌名 (如 一萬、二條、東風)！**
        - 2. **極簡文字：所有分析或原因文字必須限縮在「50字以內」，越短越精準越好。**
        - 3. 請在分析捨牌時，參考死牌/稀少牌資訊，避免等待幾乎不可能出現的牌。
        - 4. 務必「完全」按照以下雙表格 Markdown 格式輸出，不要有任何開頭或結尾碎語：

        ### 🎯 快速捨牌決策
        | 捨牌  | 推薦度 | 核心理由  | 期待進牌 |
        | :---: | :---: | :--- | :--- |
        | [圖示] | ⭐⭐⭐⭐⭐ | [精簡理由] | [多個圖示] |
        | [圖示] | ⭐⭐⭐⭐ | [精簡理由] | [多個圖示] |

        ---
        ### ⚡ 戰術與台型 
        | 分析項目 | 重點說明  |
        | :---: | :--- |
        | **進牌規劃** | [包含圖示的戰略解說] |
        | **可能台型** | [如：碰碰胡 或 無特定台型] |
        | **桌面警示** | [根據桌面資訊，提示哪些牌已死/稀少，應避免等待] |
        """
        response = llm_model.generate_content(prompt)
        import re
        enlarged_text = re.sub('([\U0001F000-\U0001F02F]+)', r'<span style="font-size: 70px; line-height: 1; vertical-align: middle; font-family: \'Segoe UI Emoji\';">\1</span>', response.text)
        return enlarged_text
    except Exception as e:
        return f"教練連線失敗... (錯誤原因: {str(e)})"
