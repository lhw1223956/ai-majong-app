import streamlit as st
import collections
import re
from PIL import Image
from core.config import TILE_INFO
from core.calculator import analyze_waiting_status, run_full_logic
from ai.vision import process_discard_pool
from ai.agent import get_model_load_error, get_structured8x34_ev_flatmc_recommendation
from ai.coach import get_majiang_coach_advice
from ui.scoreboard import init_scoreboard_state

def render_main_ui(mode, model_choice, flower_mode, dice_val, ppo_structured_8x34_model):
    init_scoreboard_state()
    if 'current_image' not in st.session_state:
        st.info("  ☝️   請先上傳照片或使用相機拍照，AI 將自動辨識手牌。")
        return

    all_codes = st.session_state.con_manual + st.session_state.exp_manual
    
    # 建立雙欄排版
    col_left, col_right = st.columns([1.6, 1.0], gap="large")
    
    # ==========================
    # 左側：牌面管理與辨識結果
    # ==========================
    with col_left:
        st.image(st.session_state.current_plot, caption=f"AI 辨識結果 ({model_choice})", use_container_width=True)
        st.markdown(f'<div class="section-header" style="margin-top: 10px;">  🎴   牌面管理 <span class="count-badge">偵測總數：{len(all_codes)} 張</span></div>', unsafe_allow_html=True)
        
        if mode == "台數計算":
            with st.container():
                st.markdown('<div class="win-tile-box">', unsafe_allow_html=True)
                if 'win_tile' not in st.session_state: st.session_state.win_tile = all_codes[0] if all_codes else '1w'
                win_info = TILE_INFO.get(st.session_state.win_tile, {'icon':'?', 'name':'未知'})

                st.write(f"#### 目前胡牌張：{win_info['name']}")
                st.button(win_info['icon'], key="win_now", use_container_width=True, type="tertiary")
                with st.popover("  🔄   更改胡牌張", use_container_width=True):
                    st.write("選擇新的胡牌張：")
                    all_keys = sorted([i for i in TILE_INFO.items() if i[1]['type'] != 'h'], key=lambda x: x[1]['w'])
                    grid = st.columns(4)
                    counts_all = collections.Counter(st.session_state.con_manual + st.session_state.exp_manual)
                    for idx, (k, v) in enumerate(all_keys):
                        with grid[idx % 4]:
                            if st.button(v['icon'], key=f"sw_{k}", disabled=(counts_all[k] >= 4), type="tertiary"):
                                st.session_state.win_tile = k; st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            st.write(f" 🐹  手牌：")
            codes = st.session_state.con_manual
            s_idx = sorted(range(len(codes)), key=lambda k: TILE_INFO[codes[k]]['w'])
            cols = st.columns(11)
            for i, idx in enumerate(s_idx):
                with cols[i % 11]:
                    if st.button(TILE_INFO[codes[idx]]['icon'], key=f"h_{i}", type="tertiary"): st.session_state.con_manual.pop(idx); st.rerun()
            with st.popover(f"  ➕   新增手牌"):
                p_c = st.columns(8); all_keys = sorted(TILE_INFO.items(), key=lambda x: x[1]['w'])
                counts_all = collections.Counter(st.session_state.con_manual + st.session_state.exp_manual)
                for k, v in all_keys:
                    limit = 1 if v['type'] == 'h' else 4
                    if st.button(v['icon'], key=f"add_h_{k}", disabled=(counts_all[k] >= limit), type="tertiary"): st.session_state.con_manual.append(k); st.rerun()

            st.markdown('<div class="swap-btn-container">', unsafe_allow_html=True)
            if st.button("  🔃   交換手牌與門前牌", help="點擊互換上下兩區的牌"):
                st.session_state.con_manual, st.session_state.exp_manual = st.session_state.exp_manual, st.session_state.con_manual
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

            st.write(f" 🐥  門前牌：")
            codes = st.session_state.exp_manual
            s_idx = sorted(range(len(codes)), key=lambda k: TILE_INFO[codes[k]]['w'])
            cols = st.columns(11)
            for i, idx in enumerate(s_idx):
                with cols[i % 11]:
                    if st.button(TILE_INFO[codes[idx]]['icon'], key=f"d_{i}", type="tertiary"): st.session_state.exp_manual.pop(idx); st.rerun()
            with st.popover(f"  ➕   新增門前"):
                p_c = st.columns(8); all_keys = sorted(TILE_INFO.items(), key=lambda x: x[1]['w'])
                counts_all = collections.Counter(st.session_state.con_manual + st.session_state.exp_manual)
                for k, v in all_keys:
                    limit = 1 if v['type'] == 'h' else 4
                    if st.button(v['icon'], key=f"add_d_{k}", disabled=(counts_all[k] >= limit), type="tertiary"): st.session_state.exp_manual.append(k); st.rerun()

        elif mode == "聽牌分析":
            st.write(f" 🐹  手牌：")
            codes = st.session_state.con_manual
            s_idx = sorted(range(len(codes)), key=lambda k: TILE_INFO[codes[k]]['w'])
            cols = st.columns(11)
            for i, idx in enumerate(s_idx):
                with cols[i % 11]:
                    if st.button(TILE_INFO[codes[idx]]['icon'], key=f"h_{i}", type="tertiary"):
                        st.session_state.con_manual.pop(idx); st.rerun()
            with st.popover(f"  ➕   新增手牌"):
                st.write("點擊圖示加入：")
                all_keys = sorted(TILE_INFO.items(), key=lambda x: x[1]['w'])
                cols_add = st.columns(8)
                counts = collections.Counter(st.session_state.con_manual)
                for idx, (k, v) in enumerate(all_keys):
                    with cols_add[idx % 8]:
                        limit = 1 if v['type'] == 'h' else 4
                        if st.button(v['icon'], key=f"add_h_{k}", disabled=(counts[k] >= limit), type="tertiary"):
                            st.session_state.con_manual.append(k); st.rerun()

        elif mode == "麻將助手":
            st.write(f" 🐹  手牌：")
            codes = st.session_state.con_manual
            s_idx = sorted(range(len(codes)), key=lambda k: TILE_INFO[codes[k]]['w'])
            cols = st.columns(11)
            for i, idx in enumerate(s_idx):
                with cols[i % 11]:
                    if st.button(TILE_INFO[codes[idx]]['icon'], key=f"ast_h_{i}", type="tertiary"): st.session_state.con_manual.pop(idx); st.rerun()
            with st.popover(f"  ➕   新增手牌"):
                p_c = st.columns(8); all_keys = sorted(TILE_INFO.items(), key=lambda x: x[1]['w'])
                counts_all = collections.Counter(st.session_state.con_manual + st.session_state.exp_manual)
                for k, v in all_keys:
                    limit = 1 if v['type'] == 'h' else 4
                    if st.button(v['icon'], key=f"ast_add_h_{k}", disabled=(counts_all[k] >= limit), type="tertiary"): st.session_state.con_manual.append(k); st.rerun()

            st.markdown('<div class="swap-btn-container">', unsafe_allow_html=True)
            if st.button("  🔃   交換手牌與門前牌", help="點擊互換上下兩區的牌", key="swap_ast"):
                st.session_state.con_manual, st.session_state.exp_manual = st.session_state.exp_manual, st.session_state.con_manual
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

            st.write(f" 🐥  門前牌：")
            codes = st.session_state.exp_manual
            s_idx = sorted(range(len(codes)), key=lambda k: TILE_INFO[codes[k]]['w'])
            cols = st.columns(11)
            for i, idx in enumerate(s_idx):
                with cols[i % 11]:
                    if st.button(TILE_INFO[codes[idx]]['icon'], key=f"ast_d_{i}", type="tertiary"): st.session_state.exp_manual.pop(idx); st.rerun()
            with st.popover(f"  ➕   新增明牌"):
                p_c = st.columns(8); all_keys = sorted(TILE_INFO.items(), key=lambda x: x[1]['w'])
                counts_all = collections.Counter(st.session_state.con_manual + st.session_state.exp_manual)
                for k, v in all_keys:
                    limit = 1 if v['type'] == 'h' else 4
                    if st.button(v['icon'], key=f"ast_add_d_{k}", disabled=(counts_all[k] >= limit), type="tertiary"): st.session_state.exp_manual.append(k); st.rerun()

            st.write("") # 增加排版間距
            
            # ─── 桌面明牌區 ───
            import sys
            from ultralytics import YOLO
            from ai.vision import load_yolo_model
            model = load_yolo_model(model_choice)
            
            st.markdown('<div class="section-header"> 🀄️ 桌面明牌區（選填）</div>', unsafe_allow_html=True)
            st.caption("拍攝或上傳所有玩家已打出去的牌、以及其他玩家的吃碰槓，AI 將用此資訊推算剩餘可進牌，提供更精準建議。")

            pool_btn_col1, pool_btn_col2, pool_btn_col3 = st.columns(3)
            with pool_btn_col1:
                with st.popover("📁 上傳圖片", use_container_width=True):
                    pool_up = st.file_uploader("上傳桌面明牌圖片", type=['png', 'jpg', 'jpeg'], key="pool_upload_input", label_visibility="collapsed")
                    if pool_up:
                        process_discard_pool(Image.open(pool_up), 'upload', model_choice, model)
            with pool_btn_col2:
                with st.popover("📷 拍攝相機", use_container_width=True):
                    pool_cam = st.camera_input("拍攝桌面明牌", key="pool_cam_input", label_visibility="collapsed")
                    if pool_cam:
                        process_discard_pool(Image.open(pool_cam), 'camera', model_choice, model)
            with pool_btn_col3:
                if st.button("🗑️ 清空桌面明牌", key="clear_pool_top", use_container_width=True):
                    st.session_state.discard_pool = []
                    if 'pool_plot' in st.session_state: del st.session_state['pool_plot']
                    st.rerun()

            if 'pool_plot' in st.session_state:
                with st.expander("🔍 查看桌面明牌辨識圖", expanded=False):
                    st.image(st.session_state.pool_plot, caption="桌面明牌辨識結果", use_container_width=True)

            if 'discard_pool' not in st.session_state:
                st.session_state.discard_pool = []
            discard_pool = st.session_state.discard_pool

            if discard_pool:
                valid_pool = [c for c in discard_pool if c in TILE_INFO]
                st.write(f"🃏 **已辨識桌面明牌** （共 {len(valid_pool)} 張）：")
                s_idx_pool = sorted(range(len(valid_pool)), key=lambda k: TILE_INFO[valid_pool[k]]['w'])
                cols_pool = st.columns(min(len(valid_pool), 11))
                for i, idx in enumerate(s_idx_pool):
                    with cols_pool[i % 11]:
                        if st.button(TILE_INFO[valid_pool[idx]]['icon'], key=f"pool_d_{i}", type="tertiary",
                                     help=f"點擊移除 {TILE_INFO[valid_pool[idx]]['name']}"):
                            st.session_state.discard_pool.pop(idx)
                            st.rerun()
            else:
                st.info("尚未辨識到桌面明牌，可直接點擊下方按鈕手動加入，或拍攝/上傳圖片辨識。")

            with st.popover("➕ 手動新增桌面明牌"):
                st.write("點擊要新增至桌面明牌的牌：")
                all_keys_pool = sorted([(k, v) for k, v in TILE_INFO.items() if v['type'] != 'h'], key=lambda x: x[1]['w'])
                pool_counts_manual = collections.Counter(st.session_state.discard_pool)
                cols_pool_add = st.columns(8)
                for idx_p, (k, v) in enumerate(all_keys_pool):
                    with cols_pool_add[idx_p % 8]:
                        if st.button(v['icon'], key=f"pool_add_{k}", type="tertiary",
                                     disabled=(pool_counts_manual[k] >= 4)):
                            st.session_state.discard_pool.append(k)
                            st.rerun()

            # 清空按鈕已整合至上方
                

    # ==========================
    # 右側：分析結果與場況設定
    # ==========================
    with col_right:
        if mode == "台數計算":
            st.markdown(f'<div class="section-header" style="margin-top: 0;">  ⚙️  場況設定</div>', unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                base_t = st.number_input("底台數", min_value=0, value=3)
                dealer = st.selectbox("誰是莊家", ["我", "下家(右)", "對家(對面)", "上家(左)"])
                wind_r = st.selectbox("目前風圈", ["東", "南", "西", "北"])
            with c2:
                m_list = st.multiselect("手動加台：", ["搶槓", "海底撈月", "河底撈魚", "槓上開花", "咪幾", "天地人胡"])
                force_zm = "海底撈月" in m_list or "槓上開花" in m_list
                if force_zm:
                    st.info("  💡   已選海底/槓上開花")
                    is_zm = True; win_on_dealer = False
                else:
                    is_zm = st.checkbox("我是自摸", value=False)
                    win_on_dealer = st.checkbox("胡莊家") if dealer != "我" and not is_zm else False

            streak = st.number_input("連莊次數", min_value=0, value=0) if dealer == "我" or win_on_dealer else 0

            hu_ok, res_tai, details, wind_info = run_full_logic(st.session_state.con_manual, st.session_state.exp_manual, st.session_state.win_tile, streak, dealer, is_zm, win_on_dealer, flower_mode, dice_val, m_list, base_t, wind_r)

            if res_tai == "相公":
                html_content = f'''<div style="background-color:#f8d7da; color:#721c24; padding:20px; border-radius:12px; text-align:center;"><div class="result-label">  🏆️  預估台數</div><div style="margin-top: 10px;"><span class="tai-number">相公  👻  </span></div></div>'''
            else:
                html_content = f'''<div style="background-color:rgba(255,255,255,0.6); color:#155724; padding:20px; border-radius:12px; text-align:center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);"><div class="result-label">  🏆️  預估台數</div><div style="display: flex; justify-content: center; align-items: baseline; margin-top: 10px;"><span class="tai-number" style="background: -webkit-linear-gradient(#db2777, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{res_tai}</span><span class="tai-text">台</span></div></div>'''

            st.write("")
            st.markdown(html_content, unsafe_allow_html=True)
            
            st.write("")
            if st.button("  🔄   重新計算", key="refresh_btn", use_container_width=True, type="primary"): st.rerun()
            
            if wind_info: st.markdown(f'<div class="wind-info" style="margin-top: 10px;">{wind_info}</div>', unsafe_allow_html=True)
            for d in details: st.write(f"  📌   {d}")
            
            # --- 結算至計分板 ---
            if res_tai != "相公" and int(res_tai) >= 0:
                st.write("")
                with st.expander("💰 一鍵匯入計分板", expanded=True):
                    p_keys = list(st.session_state.players.keys())
                    checkout_c1, checkout_c2 = st.columns(2)
                    with checkout_c1:
                        winner = st.selectbox("🏆 胡牌者", p_keys, index=0)
                    with checkout_c2:
                        loser_options = ["(自摸) 三家賠"] + p_keys
                        loser = st.selectbox("💔 放槍者", loser_options, index=0 if is_zm else 1)
                    
                    if st.button("💸 確認結算並更新", type="primary", use_container_width=True):
                        tai_count = int(res_tai)
                        base_m = st.session_state.base_money
                        tai_m = st.session_state.tai_money
                        total_win = base_m + tai_count * tai_m
                        
                        if loser == "(自摸) 三家賠":
                            for pk in p_keys:
                                if pk == winner:
                                    st.session_state.players[pk] += total_win * 3
                                else:
                                    st.session_state.players[pk] -= total_win
                            st.session_state.history_logs.append(f"🎉 結算：{winner} 自摸 {tai_count}台 (+{total_win*3})")
                            st.success(f"更新成功！{winner} 贏得 {total_win * 3} 分。")
                        else:
                            if winner == loser:
                                st.error("胡牌與放槍不能同人！")
                            else:
                                st.session_state.players[winner] += total_win
                                st.session_state.players[loser] -= total_win
                                st.session_state.history_logs.append(f"💥 結算：{winner} 胡 {loser} {tai_count}台 (+{total_win})")
                                st.success(f"更新成功！{winner} 贏 {total_win} 分。")

        elif mode == "聽牌分析":
            st.markdown(f'<div class="section-header" style="margin-top: 0;"> 📊 分析結果</div>', unsafe_allow_html=True)
            status, title, data = analyze_waiting_status(st.session_state.con_manual)
            if status == "waiting":
                bg_color, text_color = "rgba(255,255,255,0.7)", "#1E3A8A"
                pool = st.session_state.get('discard_pool', [])
                visible_counts = collections.Counter(st.session_state.con_manual + st.session_state.exp_manual + pool)
                total_visible = sum(visible_counts.values())
                
                total_remains = 0
                icon_html_list = []
                for t in data:
                    remains = max(0, 4 - visible_counts.get(t, 0))
                    total_remains += remains
                    remains_color = "#EF4444" if remains == 0 else "#2563EB"
                    remains_text = "死牌" if remains == 0 else f"剩 {remains} 張"
                    bg_remains = "#FEE2E2" if remains == 0 else "#DBEAFE"
                    tag_html = f'<div style="font-size: 11px; font-weight: bold; color: {remains_color}; background-color: {bg_remains}; border-radius: 4px; padding: 2px 4px; margin-top: 4px; text-align: center;">{remains_text}</div>'
                    icon_html_list.append(f'<div class="waiting-tile" style="display: inline-block; margin: 5px; text-align: center;"><div>{TILE_INFO[t]["icon"]}</div><div class="waiting-name">{TILE_INFO[t]["name"]}</div>{tag_html}</div>')
                icon_html = "".join(icon_html_list)
                
                prob = (total_remains / max(1, 136 - total_visible)) * 100
                pool_hint = f" (已扣除已知 {total_visible} 張)" if total_visible > len(st.session_state.con_manual) else " (未輸入桌面明牌)"
                prob_html = f'<div style="background: rgba(255,255,255,0.8); border-radius: 8px; padding: 12px; margin-top: 15px; text-align: center; border: 1px solid #93C5FD;"><div style="font-size: 16px; font-weight: bold; color: #1E3A8A; margin-bottom: 4px;">📊 聽牌存活率</div><div style="font-size: 14px; color: #475569;">海底剩餘 <b style="color: #E63946; font-size: 18px;">{total_remains}</b> 張{pool_hint}</div><div style="font-size: 14px; color: #475569; margin-top: 4px;">🎯 一發自摸機率：<span style="color: #E63946; font-size: 20px; font-weight: 900;">{prob:.1f}%</span></div></div>'
                
                html_content = f"""<div class="result-box" style="background-color: {bg_color}; color: {text_color}; padding: 20px; border-radius: 12px; border: 1px solid #93C5FD;"><div class="result-title" style="font-size: 18px; font-weight: bold; margin-bottom: 8px;"> 👀  聽牌狀況</div><div class="result-content" style="font-size: 20px; font-weight: bold; margin-bottom: 12px;"> 🔥  {title}</div><div style="margin-top: 10px; font-size: 14px; font-weight: 600; margin-bottom: 8px;">聽以下這些牌：</div><div class="waiting-tiles-container">{icon_html}</div>{prob_html}</div>"""
            elif status == "not_waiting":
                bg_color, text_color = "#fff3cd", "#856404"
                html_content = f"""<div class="result-box" style="background-color: {bg_color}; color: {text_color};"><div class="result-title"> 👀  聽牌狀況</div><div class="result-content">{title}  🤡 </div><div class="hint-msg">目前還沒有聽牌呦!</div></div>"""
            else:
                bg_color, text_color = "#f8d7da", "#721c24"
                html_content = f"""<div class="result-box" style="background-color: {bg_color}; color: {text_color};"><div class="result-title"> ⚠️  牌型異常</div><div class="error-msg">相公  👻 </div><div class="hint-msg">{title}</div></div>"""

            st.markdown(html_content, unsafe_allow_html=True)
            st.write("")
            if st.button("  🔄   重新分析", key="refresh_all", use_container_width=True, type="primary"): st.rerun()

        elif mode == "麻將助手":
            st.markdown('<div class="section-header" style="margin-top: 0;"> 🌟 AI 教練 </div>', unsafe_allow_html=True)
            current_pool = st.session_state.get('discard_pool', [])
            coach_input_key = (
                tuple(st.session_state.con_manual),
                tuple(st.session_state.exp_manual),
                tuple(current_pool),
            )
            if st.session_state.get("ai_coach_input_key") != coach_input_key:
                st.session_state.ai_coach_input_key = coach_input_key
                st.session_state.pop("ai_coach_recommendation", None)
                st.session_state.pop("ai_coach_llm_advice", None)

            if st.button("💡 Structured 8x34 + EV + FlatMC 建議", use_container_width=True, type="primary", key="btn_ast"):
                with st.status("🤖 Structured 8x34 + EV + FlatMC 分析中...", expanded=True) as status:
                    st.write("1. Structured 8x34 policy 讀取手牌狀態")
                    st.write("2. EV 選出 top-3 候選")
                    st.write("3. FlatMC 模擬並融合排序")
                    hybrid_tile, hybrid_msg, hybrid_results = get_structured8x34_ev_flatmc_recommendation(
                        st.session_state.con_manual,
                        st.session_state.exp_manual,
                        discard_pool=current_pool,
                        ppo_model=ppo_structured_8x34_model,
                    )
                    st.session_state.ai_coach_recommendation = (hybrid_tile, hybrid_msg, hybrid_results)
                    status.update(label="✅ 分析完成", state="complete", expanded=False)

            def render_hybrid_results(rl_tile, rl_msg, rl_results, highlight_color="#7C3AED"):
                def enlarge_tile_icons(text, size_px=34):
                    return re.sub(
                        r"([\U0001F000-\U0001F02F]\ufe0e?)",
                        rf'<span style="font-size: {size_px}px; line-height: 1; vertical-align: middle; font-family: \'Segoe UI Emoji\';">\1</span>',
                        text,
                    )

                st.markdown("### 🤖 Structured 8x34 + EV + FlatMC")
                if rl_tile:
                    tile_info = TILE_INFO[rl_tile]
                    st.markdown(
                        f"""<div style="background: #DCFCE7; border-radius: 8px; padding: 14px 16px; color: #166534; font-weight: 800; font-size: 20px; line-height: 1.6; margin-bottom: 12px;">建議打出：<span style="font-size: 42px; line-height: 1; vertical-align: middle; font-family: 'Segoe UI Emoji'; margin: 0 6px;">{tile_info['icon']}</span>{tile_info['name']}</div>""",
                        unsafe_allow_html=True,
                    )
                    if rl_msg:
                        st.markdown(
                            f"""<div style="background: #DBEAFE; border-radius: 8px; padding: 14px 16px; color: #1D4ED8; font-size: 17px; line-height: 1.9; margin-bottom: 14px;">{enlarge_tile_icons(rl_msg).replace(chr(10), '<br>')}</div>""",
                            unsafe_allow_html=True,
                        )
                else:
                    st.warning(f"AI 提示：{rl_msg}")

                if rl_tile is None and rl_msg == "模型未載入":
                    model_err = get_model_load_error("models/cnn_structured_8x34/mahjong_cnn_structured_8x34_agent_v1.zip")
                    if model_err:
                        st.code(model_err)

                for res in rl_results or []:
                    t_code = res['tile']
                    t_info = TILE_INFO[t_code]
                    final_pct = float(res.get('score', res.get('win_rate', 0.0))) * 100
                    policy_pct = float(res.get('cnn_prob', 0.0)) * 100
                    ev_pct = float(res.get('ev_normalized', 0.0)) * 100
                    flatmc_pct = float(res.get('flatmc_score', res.get('raw_win_rate', 0.0))) * 100
                    tenpai_tag = '<span style="color: #10B981; font-size: 12px; font-weight: bold;">[聽牌]</span>' if res.get('is_tenpai', False) else ""
                    ev_rank = res.get('ev_rank', '-')
                    html_str = f"""<div style="background: rgba(255,255,255,0.7); border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px; margin-bottom: 10px;"><div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;"><div style="display: flex; align-items: center;"><span style="font-size: 42px; line-height: 1; margin-right: 10px; font-family: 'Segoe UI Emoji';">{t_info['icon']}</span><span style="font-weight: 800; color: #1E293B; font-size: 17px;">{t_info['name']}</span>{tenpai_tag}</div><div style="text-align: right;"><div style="font-size: 20px; font-weight: 900; color: {highlight_color};">{final_pct:.1f}%</div></div></div><div style="font-size: 13px; color: #475569; line-height: 1.7;">Policy {policy_pct:.1f}% · EV rank {ev_rank} / {ev_pct:.1f}% · FlatMC {flatmc_pct:.1f}%</div><div style="background: #F1F5F9; border-radius: 4px; height: 7px; width: 100%; margin-top: 7px;"><div style="background: {highlight_color}; height: 7px; border-radius: 4px; width: {min(100, final_pct)}%;"></div></div></div>"""
                    st.markdown(html_str, unsafe_allow_html=True)

            if "ai_coach_recommendation" in st.session_state:
                render_hybrid_results(*st.session_state.ai_coach_recommendation)

            if st.button("💬 呼叫 LLM 教練", use_container_width=True, type="secondary", key="btn_llm_coach"):
                with st.status("💬 LLM 教練分析中...", expanded=True) as status:
                    advice = get_majiang_coach_advice(
                        st.session_state.con_manual,
                        st.session_state.exp_manual,
                        discard_pool=current_pool,
                    )
                    st.session_state.ai_coach_llm_advice = advice
                    status.update(label="✅ LLM 回覆完成", state="complete", expanded=False)

            if "ai_coach_llm_advice" in st.session_state:
                st.markdown("### 💬 智慧教練")
                st.chat_message("assistant").markdown(st.session_state.ai_coach_llm_advice, unsafe_allow_html=True)
