"""Login and start the local adapter without exposing session cookies."""

import argparse
import asyncio
import json
import os
import secrets
from pathlib import Path

import httpx

from .adapter import ORIGIN, STATE_PATH, APIError, PolyUClient
from .server import create_client


def private_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + ".tmp-" + secrets.token_hex(6))
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as output:
            output.write(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


async def login(browser: str, state_path: Path) -> None:
    from playwright.async_api import async_playwright

    async with async_playwright() as playwright:
        instance = await playwright.chromium.launch(
            headless=False, channel="chrome" if browser == "chrome" else None
        )
        try:
            context = await instance.new_context()
            page = await context.new_page()
            await page.goto(ORIGIN)
            print("請在開啟的瀏覽器完成學校登入，進入聊天頁面後回到此處按 Enter。")
            await asyncio.to_thread(input)
            check = await context.request.post(ORIGIN + "/api/iam/me", max_redirects=0)
            if check.status != 200 or "application/json" not in check.headers.get(
                "content-type", ""
            ):
                raise APIError(401, "登入尚未完成，請重新執行 polyugenai login。")
            data = await check.json()
            if not isinstance(data, dict) or not data or data.get("error"):
                raise APIError(401, "學校未確認登入身分，請重新登入。")
            # Save cookies applicable to the school host only; omit SSO local storage.
            cookies = await context.cookies(ORIGIN)
            private_write(state_path, json.dumps({"cookies": cookies, "origins": []}))
            print(f"登入狀態已儲存於 {state_path}（檔案權限 600）。")
        finally:
            await instance.close()


async def doctor() -> None:
    async with create_client() as client:
        upstream = PolyUClient(client)
        await upstream.post("/api/iam/me")
        models = await upstream.models()
        print("學校登入有效；可用模型：" + ", ".join(m["id"] for m in models))


def main() -> None:
    parser = argparse.ArgumentParser(description="PolyU GenAI → OpenAI 本機轉接器")
    sub = parser.add_subparsers(dest="command", required=True)
    auth = sub.add_parser("login", help="開啟瀏覽器登入學校帳戶")
    auth.add_argument("--browser", choices=["chrome", "chromium"], default="chrome")
    sub.add_parser("doctor", help="檢查登入及模型，不產生付費訊息")
    serve = sub.add_parser("serve", help="啟動本機 OpenAI API")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    state_path = Path(os.getenv("POLYU_STATE_PATH", str(STATE_PATH)))
    try:
        if args.command == "login":
            asyncio.run(login(args.browser, state_path))
        elif args.command == "doctor":
            asyncio.run(doctor())
        else:
            import uvicorn

            key = os.getenv("POLYUGENAI_API_KEY")
            if not key:
                key_path = Path(".auth/api-key")
                if not key_path.exists():
                    private_write(key_path, secrets.token_urlsafe(32))
                key = key_path.read_text().strip()
                if not key:
                    raise APIError(503, ".auth/api-key 是空的，請刪除後重新啟動。")
                os.environ["POLYUGENAI_API_KEY"] = key
                print("本機 API key 位於 .auth/api-key；請將其填入客戶端。")
            uvicorn.run(
                "polyugenai.server:app",
                host="127.0.0.1",
                port=args.port,
                limit_concurrency=16,
                access_log=False,
            )
    except (APIError, httpx.HTTPError) as exc:
        parser.exit(1, f"錯誤：{exc if isinstance(exc, APIError) else '學校連線失敗。'}\n")


if __name__ == "__main__":
    main()
