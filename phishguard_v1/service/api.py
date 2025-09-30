from __future__ import annotations
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from ..config import settings
from ..features.fetcher import fetch_one
from ..features.parser import extract_from_html
from ..features.render import render_screenshot
from ..models.inference import InferencePipeline

app = FastAPI(title="PhishGuard v1", version="1.0.0")
pipe = InferencePipeline(fusion_ckpt_path="artifacts/fusion_dalwfr_v5.pt", enable_fusion=True)

class PredictInput(BaseModel):
    url: str = Field(..., description="待检测URL")
    screenshot: bool = Field(False, description="是否尝试截图（需安装playwright）")

@app.post("/predict")
async def predict(inp: PredictInput):
    from httpx import AsyncClient
    async with AsyncClient(timeout=settings.http_timeout, headers={"User-Agent": settings.user_agent}) as client:
        item = await fetch_one(inp.url, client)

    html_feats = extract_from_html(item.get("html", ""), item.get("final_url") or item.get("request_url"))
    item["html_feats"] = html_feats

    snap = {}
    if inp.screenshot:
        snap = render_screenshot(item.get("final_url") or item.get("request_url"))
        item.update(snap)

    pred = pipe.predict(item)
    return {"input": inp.model_dump(), "prediction": pred, "features": item}

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": pipe.url_model.model_id}
