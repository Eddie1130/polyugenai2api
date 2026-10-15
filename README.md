# polyugenai2api

將自己的 [PolyU GenAI](https://genai.polyu.edu.hk/) 登入工作階段，轉成可供 OpenAI SDK、支援自訂 OpenAI URL 的聊天客戶端使用的本機 API。

## 快速開始

需要 Python 3.11+ 及 Google Chrome。

```bash
cd /Users/eddie/Desktop/polyugenai2api
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[login]'
polyugenai2api login
polyugenai2api doctor
polyugenai2api serve
```

`login` 會開啟獨立 Chrome 視窗。完成學校 SSO／MFA、進入聊天頁後，回到終端按 Enter。登入 Cookie 儲存在 `.auth/storage-state.json`，權限為 `600`。不需要 OpenAI 平台 API key。

沒有 Chrome 時，可執行 `playwright install chromium`，再用 `polyugenai2api login --browser chromium`。

客戶端設定：

| 欄位 | 值 |
|---|---|
| Base URL | `http://127.0.0.1:8000/v1` |
| API key | `.auth/api-key` 的內容（首次 `serve` 時產生） |
| Model | `GET /v1/models` 回傳的模型 ID，例如 HAR 中的 `GPT56Sol`、`Qwen` |

```bash
export OPENAI_API_KEY="$(cat .auth/api-key)"
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"GPT56Sol","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

使用 Python SDK（另安裝 `pip install openai`）：

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    api_key=Path(".auth/api-key").read_text().strip(),
    base_url="http://127.0.0.1:8000/v1",
    max_retries=0,  # 避免生成請求失敗後自動重試而重複扣除學校額度
)
answer = client.chat.completions.create(
    model="GPT56Sol",
    messages=[{"role": "user", "content": "請解釋 queue 與 stack"}],
)
print(answer.choices[0].message.content)
```

## 支援範圍

- `GET /health`：僅檢查本機服務；`polyugenai2api doctor` 才會檢查學校登入與模型。
- `GET /v1/models`：即時讀取學校 `/api/llm/list`，只列出帳戶獲授權且支援串流的模型。
- `POST /v1/chat/completions`：一般 JSON 或 SSE 串流；串流成功以 `[DONE]` 結束。
- `system`、`developer`、`user`、`assistant` 文字訊息。多輪歷史序列化為單一文字提示，角色優先序與原生 OpenAI API 不完全相同；最後一則須為 `user`。
- `user` 的 `image_url` 內容支援 PNG/JPEG/WebP/GIF base64 data URL。遠端圖片 URL 不支援。
- `reasoning_effort` 會依學校模型設定驗證；未指定時沿用模型預設值。
- 每次請求建立新學校 topic，避免不同客戶端或並行請求共用歷史。新 topic 和訊息會保留在學校帳戶；程式不會刪除它們。

這是 **Chat Completions 的有限相容介面**，目前不支援 Responses API、工具呼叫、JSON schema 輸出、檔案、音訊、圖片生成、temperature、max_tokens／max_completion_tokens、stop 或多個候選答案。未支援參數會回傳 400，不會默默忽略。需要 Responses API／工具呼叫的程式（例如部分程式碼代理）不能直接使用。

學校串流只提供文字，沒有每次請求的 token 統計，因此 `usage` 為 `null`，不支援 `stream_options.include_usage`。HAR 中的歷史 token 數字包含 topic 的累計用量，不能當成單次用量。

## 登入與設定

| 環境變數 | 用途 |
|---|---|
| `POLYUGENAI2API_STATE_PATH` | 登入狀態檔路徑，預設 `.auth/storage-state.json` |
| `POLYUGENAI2API_COOKIE` | 可選：手動提供自己的完整 Cookie 標頭，優先於登入狀態檔 |
| `POLYUGENAI2API_API_KEY` | 可選：自行設定本機轉接器金鑰，優先於 `.auth/api-key` |

`polyugenai2api serve --port 8001` 可修改埠號。服務只綁定 `127.0.0.1`，需要本機 Bearer key；不啟用 CORS。直接以 Uvicorn 啟動時，必須設定 `POLYUGENAI2API_API_KEY`。

學校登入失效會回傳 401，重新執行 `polyugenai2api login` 即可；服務每次請求重新載入 Cookie，毋須重啟。Cookie 模式只適用於以 Cookie 驗證的工作階段，尚未確認是否另有裝置綁定或額外驗證要求。不要分享 `.auth/`、Cookie 或 HAR；均已加入 `.gitignore`。

## 分析依據與驗證

以 REA MCP 的 `inspect_web_network_capture` 在本機讀取使用者提供的 HAR（18 筆紀錄），並對照學校公開前端 `index-B2CEO3cf.js`：

1. `POST /api/topics/create`，JSON `{"title":"..."}` → `topic_id`（公開前端程式）。
2. `POST /api/chat/create`，multipart 欄位 `genai_model`、`input`、`topic_id`，及可重複的 `images` → `chat_id`（HAR）。
3. `POST /api/chat/send/{endpoint_key}`，JSON 模型設定 → `data: {"content":"..."}` SSE；HAR 使用 EOF 結束，沒有 `[DONE]`。
4. `/api/llm/list` 的 `details` 包含 `id`、`endpoint_key` 及模型設定（公開前端程式）。

OpenAI 輸出格式對照 [官方 Chat Completions API 文件](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)。這是非官方轉接器，學校內部 API 可能變更。

驗證結果（2026-10-16）：

- 41 項自動測試通過，涵蓋 OpenAI Python SDK、串流、Unicode 分段、錯誤、對話隔離、模型授權篩選、取消時清理和 Cookie 範圍。
- HAR 三段 SSE 的離線重播與原文一致。
- Ruff、格式檢查及 mypy 通過；CLI 本機伺服器啟動與健康檢查正常。
- 使用專案現有登入狀態，成功讀取學校即時模型清單，並透過 OpenAI Python SDK 完成 Qwen 一般生成與串流生成（兩個極短請求，會使用少量學校額度，測試 topic 保留在帳戶）。

已完成實際驗證的模型為 Qwen；其他模型及圖片輸入尚未逐一實測。HAR 本身沒有登入憑證，實際驗證使用的是獨立的本機登入狀態檔。

```bash
pip install -e '.[dev]'
pytest
ruff check polyugenai2api tests
ruff format --check polyugenai2api tests
```
