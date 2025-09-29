from __future__ import annotations
import argparse, asyncio, pandas as pd
from ..features.fetcher import fetch_many
from ..features.parser import extract_from_html
from loguru import logger

async def run(inp, out, screenshot=False):
    urls = [u.strip() for u in open(inp, "r", encoding="utf-8").read().splitlines() if u.strip()]
    items = await fetch_many(urls)
    rows = []
    for it in items:
        hf = extract_from_html(it.get("html",""), it.get("final_url") or it.get("request_url"))
        it["html_feats"] = hf
        row = {}
        # 扁平化关键信息方便训练
        row.update({k: it["url_feats"].get(k) for k in it["url_feats"].keys() if k not in ["domain","hostname"]})
        for k in ["status_code","content_type","bytes"]:
            row[k] = it.get(k)
        for k in ["title_len","num_meta","num_links","num_stylesheets","num_scripts","num_script_ext","num_script_inline","num_iframes","num_forms","has_password_input","has_email_input","suspicious_js_inline","has_html"]:
            row[k] = hf.get(k, 0)
        row["label"] = None  # 稍后由 build_dataset.py 或人工/规则填充
        row["url"] = it.get("final_url") or it.get("request_url")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    logger.info(f"saved {out}, rows={len(df)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--screenshot", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(run(args.inp, args.out, screenshot=bool(args.screenshot)))
