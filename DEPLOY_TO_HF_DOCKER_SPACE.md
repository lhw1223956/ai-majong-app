# Hugging Face Docker Space 部署步驟

## 方式一：網頁建立 Space

1. 到 Hugging Face 建立新的 Space。
2. Space SDK 選擇 `Docker`。
3. Repository 建議名稱可用 `ai-majong-app`。
4. 上傳或同步此資料夾內的所有檔案。
5. 到 `Settings -> Variables and secrets` 新增：

```text
GEMINI_API_KEY=你的 Gemini API Key
```

6. 等待 Space 自動 Build 完成後，即可取得公開網址。

## 方式二：使用 Hugging Face CLI

如果本機已登入 Hugging Face CLI，可以使用：

```powershell
hf auth login
hf repos create xucinnn/ai-majong-app --type space --space-sdk docker --exist-ok
hf upload xucinnn/ai-majong-app "C:\Users\liuwi\Desktop\majong anti\APP_STREAMLIT_GITHUB_DEPLOY_20260603" --type space --commit-message "Deploy Mahjong Streamlit Docker Space"
```

若 Space 名稱或帳號不同，請把 `xucinnn/ai-majong-app` 改成你的實際 Space ID。

## 為什麼改用 Docker Space

這個專案會載入 `ultralytics`、`torch`、`opencv` 等套件。一般 Streamlit Cloud 有時會因為底層 apt 套件版本不同，出現：

- `ImportError: libGL.so.1`
- `ImportError: libgthread-2.0.so.0`
- `Error installing requirements`

Docker Space 可以固定 Python 版本、系統套件與啟動指令，因此比單純 Streamlit Cloud 更適合這個影像辨識 Demo。
