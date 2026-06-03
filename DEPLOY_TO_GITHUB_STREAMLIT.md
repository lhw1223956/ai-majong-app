# 上傳 GitHub 並部署 Streamlit 公開網頁

目標：讓自動麻將系統在雲端公開執行，不需要本機電腦開機。

## 第一步：建立 GitHub repository

到 GitHub 建立一個 public repository，例如：

```text
mahjong-auto-system
```

不要勾選初始化 README，因為此資料夾已經有 README。

## 第二步：推送此資料夾

在此資料夾開啟 PowerShell，執行：

```powershell
git remote add origin https://github.com/<你的帳號>/mahjong-auto-system.git
git branch -M main
git push -u origin main
```

如果第一次使用 GitHub，推送時會要求登入。

## 第三步：部署到 Streamlit Community Cloud

開啟：

```text
https://share.streamlit.io/
```

建立 New app，設定：

```text
Repository: <你的帳號>/mahjong-auto-system
Branch: main
Main file path: app.py
```

部署成功後會得到公開網址，例如：

```text
https://mahjong-auto-system.streamlit.app/
```

## Gemini LLM 教練設定

若要啟用 LLM 教練，請在 Streamlit Community Cloud：

```text
App settings -> Secrets
```

加入：

```toml
GEMINI_API_KEY = "你的 Gemini API Key"
```

不要把實際 `.streamlit/secrets.toml` 放進公開 GitHub。

## 注意

- 此資料夾已排除訓練紀錄、TensorBoard、報告、簡報與實際 API Key。
- 模型檔都低於 GitHub 單檔 100MB 限制，因此可以直接放在 GitHub，不需要 Git LFS。
- 如果 Streamlit Cloud 記憶體不足，建議改用 Hugging Face Spaces。
