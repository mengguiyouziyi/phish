from __future__ import annotations
import argparse, asyncio, pandas as pd, requests, time
from loguru import logger
from ..features.fetcher import fetch_many
from ..features.parser import extract_from_html
from ..config import settings

# ====== 数据源：OpenPhish / PhishTank / URLHaus / Tranco ======

def get_openphish(limit=5000):
    # 免费公开feed（近实时）
    url = "https://openphish.com/feed.txt"
    try:
        txt = requests.get(url, timeout=15, verify=settings.tls_verify).text
        urls = [l.strip() for l in txt.splitlines() if l.startswith("http")]
        return urls[:limit]
    except Exception:
        return []

def get_phishtank(limit=5000):
    # 简易公共API（每日频次有限，更多请注册key）
    # 这里我们直接拉开放数据CSV（若不可用请切换到API）
    try:
        csv_url = "https://data.phishtank.com/data/online-valid.csv"
        df = pd.read_csv(csv_url)
        urls = df["url"].dropna().astype(str).tolist()
        return urls[:limit]
    except Exception:
        return []

def get_urlhaus(limit=5000):
    api = "https://urlhaus.abuse.ch/downloads/text/"
    try:
        txt = requests.get(api, timeout=15, verify=settings.tls_verify).text
        urls = [l.strip() for l in txt.splitlines() if l and not l.startswith("#")]
        return urls[:limit]
    except Exception:
        return []

def get_tranco(limit=5000):
    # Tranco 最新榜单（CSV下载链接需在网站获取；此处提供兜底的公共镜像接口，如不可用需手工下载）
    try:
        csv = "https://tranco-list.eu/top-1m.csv.zip"
        df = pd.read_csv(csv, compression="zip", header=None, names=["rank","domain"])
        urls = ["http://" + d for d in df["domain"].head(limit).astype(str).tolist()]
        return urls
    except Exception:
        # 回退：使用常见良性站点样例
        seeds = ["example.com", "wikipedia.org", "github.com", "google.com", "microsoft.com", "apple.com", "cloudflare.com"]
        return ["http://" + s for s in seeds][:limit]

async def run(limit_pos=5000, limit_neg=5000, out="data/dataset.parquet"):
    pos = list(dict.fromkeys(get_openphish(limit_pos) + get_phishtank(limit_pos) + get_urlhaus(limit_pos)))
    neg = list(dict.fromkeys(get_tranco(limit_neg)))
    logger.info(f"positives={len(pos)}, negatives={len(neg)}")

    # 抓取两侧特征
    pos_items = await fetch_many(pos)
    neg_items = await fetch_many(neg)
    def to_rows(items, label):
        rows = []
        for it in items:
            hf = extract_from_html(it.get("html",""), it.get("final_url") or it.get("request_url"))
            it["html_feats"] = hf
            row = {}
            row.update({k: it["url_feats"].get(k) for k in it["url_feats"].keys() if k not in ["domain","hostname"]})
            for k in ["status_code","content_type","bytes"]:
                row[k] = it.get(k)
            for k in ["title_len","num_meta","num_links","num_stylesheets","num_scripts","num_script_ext","num_script_inline","num_iframes","num_forms","has_password_input","has_email_input","suspicious_js_inline","has_html"]:
                row[k] = hf.get(k, 0)
            row["label"] = int(label)
            row["url"] = it.get("final_url") or it.get("request_url")
            rows.append(row)
        return rows

    rows = to_rows(pos_items, 1) + to_rows(neg_items, 0)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    logger.info(f"saved dataset: {out}, rows={len(df)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_pos", type=int, default=5000)
    ap.add_argument("--limit_neg", type=int, default=5000)
    ap.add_argument("--out", type=str, default="data/dataset.parquet")
    args = ap.parse_args()
    import asyncio
    asyncio.run(run(args.limit_pos, args.limit_neg, args.out))
