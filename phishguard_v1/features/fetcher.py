from __future__ import annotations
import asyncio, httpx, re
from loguru import logger
from typing import Optional, Dict, Any
from .url_features import url_stats
from ..config import settings

async def fetch_one(url: str, client: Optional[httpx.AsyncClient] = None) -> Dict[str, Any]:
    meta = {"ok": False, "error": None, "redirects": [], "tls_retry": False, "vercel_www_fix": False}
    # 规范化
    if not re.match(r"^https?://", url):
        url = "http://" + url

    async def _do_request(c: httpx.AsyncClient, u: str):
        # 有些站点不支持 HEAD，不要让它阻断
        try:
            r_head = await c.head(u, follow_redirects=True)
            redirects = [str(h.headers.get("location", "")) for h in r_head.history]
        except Exception:
            redirects = []
        r = await c.get(u, follow_redirects=True)
        redirects = [str(h.headers.get("location", "")) for h in r.history] or redirects
        return r, redirects

    # 首次尝试：按配置的 tls_verify/http2/trust_env
    close_after = False
    if client is None:
        client = httpx.AsyncClient(
            timeout=settings.http_timeout,
            headers={"User-Agent": settings.user_agent},
            verify=settings.tls_verify,
            http2=settings.http2,
            trust_env=True,   # 让 httpx 读取系统/环境代理
        )
        close_after = True

    try:
        try:
            r, redirects = await _do_request(client, url)
        except Exception as e:
            # TLS 类错误：尝试一次回退
            err = str(e)
            if settings.retry_on_tls_error and any(k in err for k in ["CERTIFICATE_VERIFY_FAILED", "self-signed", "Hostname mismatch"]):
                meta["tls_retry"] = True
                url_try = url
                # 处理 vercel 的 www 二级导致 Hostname mismatch 的情况
                if ".vercel.app" in url_try and "://www." in url_try:
                    url_try = url_try.replace("://www.", "://", 1)
                    meta["vercel_www_fix"] = True
                async with httpx.AsyncClient(
                    timeout=settings.http_timeout,
                    headers={"User-Agent": settings.user_agent},
                    verify=False,            # 回退为不校验证书
                    http2=settings.http2,
                    trust_env=True,
                ) as c2:
                    r, redirects = await _do_request(c2, url_try)
                    url = url_try
            else:
                raise

        content = r.content[: settings.http_max_bytes]
        try:
            html = content.decode(r.encoding or "utf-8", errors="ignore")
        except Exception:
            html = content.decode("utf-8", errors="ignore")

        headers = dict(r.headers)
        cookies = {k: v for k, v in r.cookies.items()}

        out = {
            "request_url": url,
            "final_url": str(r.url),
            "status_code": r.status_code,
            "content_type": headers.get("content-type", ""),
            "headers": headers,
            "set_cookie": headers.get("set-cookie", ""),
            "cookies": cookies,
            "html": html,
            "bytes": len(content),
            "url_feats": url_stats(url),
            "ok": True,
            "meta": {**meta, "redirects": redirects},
        }
        return out
    except Exception as e:
        meta["error"] = str(e)
        logger.warning(f"fetch error {url}: {e}")
        return {
            "request_url": url,
            "final_url": None,
            "status_code": None,
            "content_type": None,
            "headers": {},
            "set_cookie": "",
            "cookies": {},
            "html": "",
            "bytes": 0,
            "url_feats": url_stats(url),
            "ok": False,
            "meta": meta,
        }
    finally:
        if close_after:
            await client.aclose()

async def fetch_many(urls, concurrency: int = None, timeout: float = None):
    concurrency = concurrency or settings.concurrency
    timeout = timeout or settings.http_timeout
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(
        timeout=timeout,
        headers={"User-Agent": settings.user_agent},
        verify=settings.tls_verify,
        http2=settings.http2,
        trust_env=True,             # 读取环境代理（HTTP[S]_PROXY/NO_PROXY）
    ) as client:
        async def _one(u):
            async with sem:
                return await fetch_one(u, client)
        return await asyncio.gather(*[_one(u) for u in urls])
