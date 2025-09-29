from __future__ import annotations
from ..config import settings
from typing import Optional, Dict, Any

def render_screenshot(url: str) -> Dict[str, Any]:
    if not settings.allow_screenshot:
        return {"screenshot_path": None}
    try:
        from playwright.sync_api import sync_playwright
        import time, os, hashlib
        os.makedirs(settings.data_dir, exist_ok=True)
        fn = os.path.join(settings.data_dir, f"snap_{hashlib.md5(url.encode()).hexdigest()}.png")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(url, wait_until="load", timeout=int(settings.http_timeout*1000))
            time.sleep(0.5)
            page.screenshot(path=fn, full_page=True)
            browser.close()
        return {"screenshot_path": fn}
    except Exception as e:
        return {"screenshot_path": None, "screenshot_error": str(e)}
