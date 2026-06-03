---
title: Mahjong Auto System
emoji: 🀄
colorFrom: green
colorTo: blue
sdk: streamlit
app_file: app.py
---

# 自動麻將系統

這是 Streamlit 版本的自動麻將系統 DEMO。

功能包含：

- YOLO 麻將牌影像辨識
- 台灣麻將台數與聽牌分析
- Structured 8x34 RL 出牌建議
- EV 與 FlatMC 輔助排序
- 選配 Gemini LLM 教練

## Streamlit Community Cloud 部署

1. 將此資料夾內容上傳到 GitHub public repository。
2. 到 Streamlit Community Cloud 建立新 App：

```text
https://share.streamlit.io/
```

3. 選擇你的 GitHub repository。
4. Main file path 填：

```text
app.py
```

5. Python dependencies 會由 `requirements.txt` 安裝。

## Gemini LLM 教練

若要啟用 LLM 教練，請在 Streamlit Community Cloud 的 App settings -> Secrets 加入：

```toml
GEMINI_API_KEY = "你的 Gemini API Key"
```

不要把實際 `.streamlit/secrets.toml` 上傳到公開 GitHub repository。
