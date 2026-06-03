import streamlit as st
# Trigger hot-reload 1
from PIL import Image
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import requests

from core.config import TILE_INFO
from ai.vision import process_detection, load_yolo_model
from ai.agent import load_ppo_agent
from ui.main_ui import render_main_ui

@st.cache_data
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# --- 0. 介面與 CSS 樣式設定 ---
st.set_page_config(page_title="AI 麻將計算平台", layout="wide", page_icon="🀄️")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans TC', sans-serif;
    color: #1E293B;
}
.stApp {
    background-color: #F8FAFC;
}

/* Card 佈局與通用設定 */
.glass-card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    margin-bottom: 24px;
    border: 1px solid #E2E8F0;
}

/* 🀄 牌按鈕樣式（僅套用 tertiary，避免影響記分板/一般按鈕） */
.stButton > button[kind="tertiary"] {
    border: 1px solid #CBD5E1 !important; 
    background-color: white !important;
    height: 100px !important; width: 80px !important; margin: 2px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    transition: transform 0.1s, box-shadow 0.1s !important;
}
.stButton > button[kind="tertiary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    border-color: #94A3B8 !important;
}
.stButton > button[kind="tertiary"] div p { font-size: 70px !important; color: #1E293B !important; font-family: "Segoe UI Emoji", sans-serif !important; margin: 0 !important; line-height: 1 !important; }

/* 算台模式樣式 */
.win-tile-box { background-color: #FEF3C7; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px; border: 1px solid #FDE68A; }
.section-header { font-size: 22px; font-weight: 700; color: #0F172A; margin: 24px 0 16px 0; border-bottom: 2px solid #E2E8F0; padding-bottom: 8px; display: flex; align-items: center; }
.count-badge { background-color: #3B82F6; color: white; padding: 4px 12px; border-radius: 20px; font-size: 14px; font-weight: 600; margin-left: 12px; display: inline-block;}
.result-label { font-size: 18px; font-weight: 600; margin-bottom: 8px; color: #64748B; }
.wind-info { background-color: #F1F5F9; padding: 12px; border-radius: 8px; font-size: 14px; color: #475569; margin-bottom: 16px; text-align: center; border: 1px solid #E2E8F0; }
.tai-number { font-size: 4rem; font-weight: 900; line-height: 1.1; font-family: 'Noto Sans TC', sans-serif; margin-right: 8px; color: #0F172A; }
.tai-text { font-size: 2rem; font-weight: 700; line-height: 1.2; font-family: 'Noto Sans TC', sans-serif; color: #475569; }
.swap-btn-container { text-align: center; margin: 16px 0; }
.swap-btn-container button { height: 48px !important; width: 280px !important; font-size: 16px !important; font-weight: 600 !important; background-color: #F8FAFC !important; border: 1px solid #CBD5E1 !important; border-radius: 24px !important; color: #475569 !important; }

/* 聽牌模式樣式 */
.stButton > button[kind="primary"] { height: auto !important; width: 100% !important; padding: 14px !important; background-color: #3B82F6 !important; border: none !important; border-radius: 12px !important; color: white !important; box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.4) !important;}
.stButton > button[kind="primary"] div p { font-size: 18px !important; color: white !important; font-family: 'Noto Sans TC', sans-serif !important; font-weight: 600 !important; }
.stButton > button[kind="primary"]:hover { background-color: #2563EB !important; transform: translateY(-1px); }

.result-box { padding: 30px; border-radius: 16px; text-align: center; margin-top: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
.result-title { font-size: 24px; font-weight: 700; margin-bottom: 16px; opacity: 0.9; }
.result-content { font-size: 32px; font-weight: 800; line-height: 1.4; }
.waiting-tiles-container { display: flex; flex-wrap: wrap; justify-content: center; gap: 16px; margin-top: 24px; }
.waiting-tile { background-color: #fff; border: 1px solid #E2E8F0; border-radius: 12px; padding: 12px 24px; font-size: 64px; line-height: 1; display: flex; flex-direction: column; align-items: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05); transition: transform 0.2s; }
.waiting-tile:hover { transform: scale(1.05); border-color: #3B82F6; }
.waiting-name { font-size: 18px; font-weight: 600; margin-top: 8px; color: #475569; font-family: 'Noto Sans TC', sans-serif; }
.error-msg { font-size: 24px; font-weight: 700; color: #EF4444; }
.hint-msg { font-size: 16px; color: #64748B; margin-top: 12px; font-weight: 500;}

/* 一般功能性按鈕覆蓋 */
.action-btn-container > div > div > button {
    height: auto !important;
    width: auto !important;
    min-height: 40px !important;
    padding: 8px 20px !important;
    font-size: 15px !important;
    font-family: 'Noto Sans TC', sans-serif !important;
    font-weight: 600 !important;
    color: #475569 !important;
    border-radius: 8px !important;
    border: 1px solid #CBD5E1 !important;
    background-color: #F8FAFC !important;
    box-shadow: none !important;
    transform: none !important;
}
.action-btn-container > div > div > button:hover {
    background-color: #FEE2E2 !important;
    border-color: #FCA5A5 !important;
    color: #B91C1C !important;
}
</style>
""", unsafe_allow_html=True)

st.title(" 🀄️  自動麻將系統 ")

# --- 1. 核心設定區 ---
with st.sidebar:
    lottie_robot = load_lottieurl("https://lottie.host/170bc0e1-0c58-4fc5-91db-cdef0fb5da7e/N9A0gSIVcQ.json")
    if lottie_robot:
        st_lottie(lottie_robot, height=130, key="robot_sidebar")

    st.title(" ⚙️  核心設定")
    
    app_mode = option_menu(
        menu_title=None,
        options=["台數計算", "聽牌分析", "麻將助手", "🏆 戰局計分板"],
        icons=["calculator", "eye", "robot", "trophy"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#3B82F6", "font-size": "20px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#E2E8F0"},
            "nav-link-selected": {"background-color": "#1E293B", "color": "white", "font-weight": "600"},
        }
    )
    
    if app_mode == "台數計算":
        flower_mode = st.radio("花牌玩法", ["莊家花 (莊家為東)", "開門花 (骰子開門處為東)"])
        dice_val = st.number_input("骰子點數", min_value=3, max_value=18, value=7) if flower_mode == "開門花 (骰子開門處為東)" else 0
        st.info("本工具用於胡牌計算台數，拍攝時須包含手牌以及門前牌區域。")
    elif app_mode == "聽牌分析":
        st.info("本工具進行聽牌分析，請確保手牌符合3n+1 張的聽牌規範。")
        flower_mode, dice_val = None, 0
    else:
        st.info("請拍攝您的手牌與已吃碰的門前牌。\n可額外拍攝桌面明牌區（打出去的牌、其他玩家吃碰槓），AI 教練將為您提供更精準的分析。")
        flower_mode, dice_val = None, 0

# --- 載入模型 ---
ppo_structured_8x34_model = load_ppo_agent(
    "models/cnn_structured_8x34/mahjong_cnn_structured_8x34_agent_v1.zip",
    loader_version="structured8x34-ev-flatmc-2026-05-27-v1",
)

# --- 2. 啟動入口與快取清空 ---
if app_mode == "🏆 戰局計分板":
    from ui.scoreboard import render_scoreboard_ui
    render_scoreboard_ui()
else:
    st.markdown('<div class="glass-card" style="padding: 16px 24px; margin-bottom: 24px;">', unsafe_allow_html=True)
    
    header_col1, header_col2, header_col3 = st.columns([1.5, 1, 2])
    with header_col1:
        st.markdown('<div style="font-size: 18px; font-weight: 700; display: flex; align-items: center; height: 100%; color: #1E293B;">📷 影像辨識</div>', unsafe_allow_html=True)
    with header_col2:
        st.markdown('<div style="font-size: 14px; font-weight: 600; color: #ec4899; display: flex; align-items: center; justify-content: flex-end; height: 100%; padding-right: 10px;">🧠 模型</div>', unsafe_allow_html=True)
    with header_col3:
        model_display = st.selectbox("辨識模型", ("YOLOv8s (標準)", "YOLOv8n (快速)", "YOLOv8s_obb", "YOLOv8n_obb"), label_visibility="collapsed")
        model_map = {"YOLOv8s (標準)": "yolov8s.pt", "YOLOv8n (快速)": "yolov8n.pt", "YOLOv8s_obb": "YOLOv8s_obb.pt", "YOLOv8n_obb": "YOLOv8n_obb.pt"}
        model_choice = model_map.get(model_display, "yolov8s.pt")
        
    model = load_yolo_model(model_choice)
    
    st.write("") # 間距
    
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        with st.popover("📁 上傳圖片", use_container_width=True):
            up = st.file_uploader("選擇照片", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
            if up: process_detection(Image.open(up), 'upload', model_choice, app_mode, model)
    with btn_col2:
        with st.popover("📷 拍攝相機", use_container_width=True):
            cam = st.camera_input("拍照", label_visibility="collapsed")
            if cam: process_detection(Image.open(cam), 'camera', model_choice, app_mode, model)
    with btn_col3:
        if st.button("🗑️ 清空所有牌", use_container_width=True):
            keys_to_clear = [
                'current_image',
                'current_plot',
                'con_manual',
                'exp_manual',
                'current_cache_key',
                'ai_coach_input_key',
                'ai_coach_recommendation',
                'ai_coach_llm_advice',
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)

    # 渲染主介面
    render_main_ui(
        app_mode,
        model_choice,
        flower_mode,
        dice_val,
        ppo_structured_8x34_model,
    )
