import streamlit as st

def init_scoreboard_state():
    if 'players' not in st.session_state:
        st.session_state.players = {
            "東風 (玩家A)": 0,
            "南風 (玩家B)": 0,
            "西風 (玩家C)": 0,
            "北風 (玩家D)": 0
        }
    if 'base_money' not in st.session_state:
        st.session_state.base_money = 300
    if 'tai_money' not in st.session_state:
        st.session_state.tai_money = 100
    if 'history_logs' not in st.session_state:
        st.session_state.history_logs = []

def render_scoreboard_ui():
    init_scoreboard_state()
    
    st.markdown('<div class="section-header"> 🏆 四人全局計分板 </div>', unsafe_allow_html=True)
    
    # Settings
    with st.expander("⚙️ 計分設定 (底/台/玩家名稱)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.base_money = st.number_input("每「底」金額", value=st.session_state.base_money, step=10, min_value=0)
        with c2:
            st.session_state.tai_money = st.number_input("每「台」金額", value=st.session_state.tai_money, step=10, min_value=0)
            
        st.write("玩家名稱設定：")
        p_keys = list(st.session_state.players.keys())
        new_names = []
        nc1, nc2, nc3, nc4 = st.columns(4)
        cols = [nc1, nc2, nc3, nc4]
        for i, pk in enumerate(p_keys):
            with cols[i]:
                new_name = st.text_input(f"玩家 {i+1}", value=pk, key=f"pname_{i}")
                new_names.append(new_name)
            
        if st.button("更新名稱設定"):
            new_players = {}
            for i, old_k in enumerate(p_keys):
                new_k = new_names[i]
                if new_k == "" or new_k in new_players: 
                    new_k = f"玩家 {i+1} (重複)" if new_k in new_players else f"玩家 {i+1}"
                new_players[new_k] = st.session_state.players[old_k]
            st.session_state.players = new_players
            st.success("名稱已更新！")
            st.rerun()
            
    # Display Scoreboard
    st.markdown("### 📊 目前戰況")
    cols = st.columns(4)
    p_keys = list(st.session_state.players.keys())
    for i, pk in enumerate(p_keys):
        score = st.session_state.players[pk]
        color = "#10B981" if score >= 0 else "#EF4444"
        sign = "+" if score > 0 else ""
        with cols[i]:
            # 單行 HTML 避免 Streamlit Markdown bug
            st.markdown(f"""<div style="background: white; border: 1px solid #E2E8F0; border-radius: 12px; padding: 15px; text-align: center; margin-bottom: 15px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);"><div style="font-size: 14px; color: #64748B; font-weight: bold; margin-bottom: 5px;">{pk}</div><div style="font-size: 28px; font-weight: 900; color: {color};">{sign}{score}</div></div>""", unsafe_allow_html=True)
            
    # Manual Adjustments
    with st.expander("🛠️ 手動微調加減分"):
        mac1, mac2, mac3 = st.columns(3)
        with mac1:
            adj_p = st.selectbox("選擇玩家", p_keys)
        with mac2:
            adj_amt = st.number_input("金額 (正數加分，負數扣分)", value=0, step=100)
        with mac3:
            st.write("")
            st.write("")
            if st.button("確認調整", type="secondary", use_container_width=True):
                if adj_amt != 0:
                    st.session_state.players[adj_p] += adj_amt
                    st.session_state.history_logs.append(f"🛠️ 手動調整：{adj_p} ({'+' if adj_amt>0 else ''}{adj_amt})")
                    st.rerun()

    # Reset
    st.write("")
    if st.button("🗑️ 清空所有分數與流水帳", type="primary"):
        for pk in p_keys:
            st.session_state.players[pk] = 0
        st.session_state.history_logs = []
        st.rerun()
        
    # History Log
    if st.session_state.history_logs:
        st.markdown("### 📜 流水帳紀錄")
        log_html = "".join([f"<li style='margin-bottom: 4px;'>{log}</li>" for log in reversed(st.session_state.history_logs)])
        st.markdown(f"<ul style='color: #475569; font-size: 14px; background: white; padding: 16px 32px; border-radius: 8px; border: 1px solid #E2E8F0;'>{log_html}</ul>", unsafe_allow_html=True)
