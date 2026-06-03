# Streamlit APP DEMO 檔案清單

整理日期：2026-06-03

用途：`app.py` 自動麻將系統展示。

## 啟動檔

- `run_app_streamlit.cmd`
- `app.py`

## 程式模組

畫面與流程：

- `ui/main_ui.py`
- `ui/scoreboard.py`

AI 與辨識：

- `ai/vision.py`
- `ai/agent.py`
- `ai/coach.py`

麻將邏輯：

- `core/config.py`
- `core/calculator.py`

決策與 EV / FlatMC 輔助：

- `algorithms/__init__.py`
- `algorithms/rl_env.py`
- `algorithms/flat_mc.py`
- `algorithms/lyl_expected_value.py`
- `algorithms/lyl_progress_ev_judgement.py`
- `algorithms/structured_discard_reward.py`
- `algorithms/reward_v2.py`

## 模型檔

RL 出牌建議模型：

- `models/cnn_structured_8x34/mahjong_cnn_structured_8x34_agent_v1.zip`

YOLO 影像辨識模型：

- `影像辨識模型/yolov8s.pt`
- `影像辨識模型/yolov8n.pt`
- `影像辨識模型/YOLOv8s_obb.pt`
- `影像辨識模型/YOLOv8n_obb.pt`

## LLM 教練設定

已放入範本：

- `.streamlit/secrets.example.toml`

若展示時要啟用 LLM 教練，請在同一層建立：

```text
.streamlit/secrets.toml
```

內容格式：

```toml
GEMINI_API_KEY = "你的 Gemini API Key"
```

實際 API Key 不應放進對外交付包。

## Python 套件

套件清單在：

- `requirements_app_demo.txt`

若要在新環境安裝，可執行：

```bat
python -m pip install -r requirements_app_demo.txt
```

## 啟動方式

在 `APP_STREAMLIT_DEMO_required_files_20260603` 資料夾中執行：

```bat
run_app_streamlit.cmd
```

預設網址：

```text
http://127.0.0.1:8501/
```

## 沒有放入的內容

- 模型對戰網頁 demo
- 模型對戰額外 checkpoint
- TensorBoard 與訓練紀錄
- `__pycache__`
- 報告、簡報、論文資料
- 實際 API Key
