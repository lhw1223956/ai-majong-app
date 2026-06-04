---
title: Mahjong Auto System
emoji: 🀄
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 自動麻將系統

這是自動麻將系統的公開展示版本，使用 Streamlit 提供操作介面，並整合：

- YOLO 麻將牌影像辨識
- Structured 8x34 強化學習策略
- EV 與 FlatMC 推薦邏輯
- Gemini LLM 人工智慧教練

## Hugging Face Docker Space 部署

此版本已改為 Hugging Face Docker Space 格式。Docker 映像會自行安裝 OpenCV 需要的系統函式庫，避免一般 Streamlit Cloud 常見的 `libGL.so.1`、`libgthread-2.0.so.0` 與 apt 相依衝突。

Space 建議設定：

- Space SDK: `Docker`
- App port: `7860`
- Main app: `app.py`

## Gemini 金鑰

如需啟用 Gemini LLM 教練，請在 Hugging Face Space 的 `Settings -> Variables and secrets` 新增 secret：

```text
GEMINI_API_KEY=你的 Gemini API Key
```

請不要把真正的金鑰提交到 GitHub 或 Hugging Face repository。
