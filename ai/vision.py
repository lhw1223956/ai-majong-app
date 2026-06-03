import os
from pathlib import Path

import numpy as np
import streamlit as st

# Ultralytics 會在匯入時讀寫 settings.json；部分 Windows 環境預設位置可能沒有權限（WinError 5）。
# 預先指定到專案內可寫入的資料夾，避免 app 啟動時直接因 PermissionError 中斷。
_DEFAULT_YOLO_CONFIG_DIR = Path(__file__).resolve().parents[1] / ".ultralytics"
os.environ.setdefault("YOLO_CONFIG_DIR", str(_DEFAULT_YOLO_CONFIG_DIR))
_DEFAULT_YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO
from core.config import TILE_INFO

@st.cache_resource
def load_yolo_model(name):
    folder_path = os.path.join("影像辨識模型", name)
    if os.path.exists(folder_path):
        return YOLO(folder_path)
    return YOLO(name)

def _prepare_image_for_yolo(image_obj):
    if getattr(image_obj, "mode", "RGB") != "RGB":
        return image_obj.convert("RGB")
    return image_obj.copy()

def _run_yolo(model, image_obj):
    return model(_prepare_image_for_yolo(image_obj), device="cpu", verbose=False)

def process_discard_pool(image_obj, source_type, current_model_name, model):
    """辨識桌面明牌區（所有打出去的牌及吃碰槓），僅用於麻將助手模式"""
    img_id_base = getattr(image_obj, 'name', 'camera') if source_type == 'upload' else 'pool_camera_shot'
    cache_key = (img_id_base, current_model_name, 'discard_pool')
    if st.session_state.get('pool_cache_key') == cache_key:
        return  # 已辨識過，跳過
    st.session_state.pool_cache_key = cache_key
    results = _run_yolo(model, image_obj)
    st.session_state.pool_plot = results[0].plot()
    tile_data = []
    for r in results:
        if hasattr(r, 'obb') and r.obb is not None:
            classes = r.obb.cls.cpu().numpy()
            xywh = r.obb.xywhr.cpu().numpy()
            for i, c in enumerate(classes):
                tile_data.append({'code': model.names[int(c)], 'x': float(xywh[i][0]), 'y': float(xywh[i][1])})
        elif hasattr(r, 'boxes') and r.boxes is not None:
            classes = r.boxes.cls.cpu().numpy()
            xywh = r.boxes.xywh.cpu().numpy()
            for i, c in enumerate(classes):
                tile_data.append({'code': model.names[int(c)], 'x': float(xywh[i][0]), 'y': float(xywh[i][1])})
    # 桌面明牌不分上下排，全部視為已知牌
    st.session_state.discard_pool = [d['code'] for d in tile_data if d['code'] in TILE_INFO and TILE_INFO[d['code']]['type'] != 'h']
    st.toast(f"✅ 桌面明牌辨識完成！共偵測了 {len(st.session_state.discard_pool)} 張。", icon="🀄️")

def process_detection(image_obj, source_type, current_model_name, mode, model):
    img_id_base = getattr(image_obj, 'name', 'camera') if source_type == 'upload' else 'camera_shot'
    cache_key = (img_id_base, current_model_name, mode)
    if 'current_cache_key' not in st.session_state or st.session_state.current_cache_key != cache_key:
        st.session_state.current_cache_key = cache_key
        st.session_state.current_image = _prepare_image_for_yolo(image_obj)
        results = _run_yolo(model, image_obj)
        st.session_state.current_plot = results[0].plot()
        tile_data = []
        for r in results:
            if hasattr(r, 'obb') and r.obb is not None:
                classes = r.obb.cls.cpu().numpy()
                xywh = r.obb.xywhr.cpu().numpy()
                for i, c in enumerate(classes):
                    tile_data.append({'code': model.names[int(c)], 'x': float(xywh[i][0]), 'y': float(xywh[i][1])})
            elif hasattr(r, 'boxes') and r.boxes is not None:
                classes = r.boxes.cls.cpu().numpy()
                xywh = r.boxes.xywh.cpu().numpy()
                for i, c in enumerate(classes):
                    tile_data.append({'code': model.names[int(c)], 'x': float(xywh[i][0]), 'y': float(xywh[i][1])})

        if not tile_data:
            st.warning("未偵測到任何麻將牌")
            st.session_state.con_manual, st.session_state.exp_manual = [], []
            return
            
        # 🌟 修改 2：算台與助手模式都要區分上下排
        if mode in ["台數計算", "麻將助手"]:
            sorted_y = sorted(tile_data, key=lambda x: x['y'])
            gaps = np.diff([d['y'] for d in sorted_y])
            max_idx = np.argmax(gaps) if len(gaps) > 0 else -1
            threshold = (sorted_y[max_idx]['y'] + sorted_y[max_idx+1]['y'])/2 if (max_idx != -1 and gaps[max_idx] > 40) else -1

            st.session_state.con_manual = [d['code'] for d in tile_data if d['y'] >= threshold]
            st.session_state.exp_manual = [d['code'] for d in tile_data if d['y'] < threshold]
            
            # 只有算台模式才需要抓取胡牌張
            if mode == "台數計算":
                hand_objs = [d for d in tile_data if d['y'] >= threshold]
                if hand_objs: st.session_state.win_tile = max(hand_objs, key=lambda d: d['x'])['code']
                elif tile_data: st.session_state.win_tile = tile_data[0]['code']
                
        else: # 聽牌分析模式
            st.session_state.con_manual = [d['code'] for d in tile_data]
            st.session_state.exp_manual = []
            
        # 增加 Toast 狀態回饋
        st.toast(f"✅ 照片辨識完成！共偵測了 {len(tile_data)} 張牌。", icon="🎉")
