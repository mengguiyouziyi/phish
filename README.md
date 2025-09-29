# PhishGuard v1 —— 钓鱼网站监测与网页指纹识别（可抓取、可预测、带Web服务）

> 目标：启动即可使用的 **抓取 + 特征抽取 + 预测** 一体化工程，支持后续训练与主动学习（DALWFR思路）。

## 快速开始

### 1) 环境准备（Python 3.10+）
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip

# GPU（可选）：按需安装匹配CUDA的PyTorch（示例是CUDA 12.1）
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 必要依赖
pip install -r requirements.txt

# 可选：页面渲染 + 截图（Playwright）
# pip install -r extras/vision.txt
# playwright install chromium
```

### 2) 启动Web API（FastAPI）
```bash
uvicorn phishguard_v1.service.api:app --host 0.0.0.0 --port 8000
# 访问文档: http://localhost:8000/docs
```

### 3) 启动可视化界面（Gradio）
```bash
python -m phishguard_v1.service.ui
```

### 4) 直接预测一个URL
API方式：
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"url":"https://example.com"}'
```
或在Gradio界面输入URL点击“扫描”。

### 5) 快速批量抓取与特征抽取
```bash
# 准备一个URL列表（每行一个URL）
cat > data/urls.txt <<EOF
https://example.com
https://openphish.com/
EOF

# 并发抓取 + 特征抽取，保存到 data/features.parquet
python -m phishguard_v1.scripts.crawl_urls --in data/urls.txt --out data/features.parquet --screenshot 0
```

### 6) （可选）自动构建训练集（正负样本）
> 正样本：OpenPhish / PhishTank / URLHaus 最新URL；负样本：Tranco Top列表 & 常见官网；随后自动抓取特征并划分数据集。
```bash
python -m phishguard_v1.scripts.build_dataset --limit_pos 5000 --limit_neg 5000
python -m phishguard_v1.scripts.split_data --in data/dataset.parquet --train data/train.parquet --val data/val.parquet --test data/test.parquet
```

### 7) 训练融合模型（DNN）
> v1默认直接使用**预训练URL Transformer**（Hugging Face）即可预测；下述命令用于再训练**特征融合DNN**，在你具备更多抓取样本后进一步提升精度。
```bash
python -m phishguard_v1.models.train_fusion --train data/train.parquet --val data/val.parquet --epochs 5 --gpus 2
```

---

## 架构说明（v1）

- **采集层**：`httpx` 并发抓取（HTTP状态码、响应头、Cookie、HTML、外链JS/CSS、Meta等）；可选 `playwright` 渲染截图。
- **特征层**：
  - URL与域名统计特征（长度、子域层级、数字/连字符比例、TLD等）；
  - HTML结构与表单特征（密码框、表单 action 域名、iframe/脚本/样式统计、关键字等）；
  - 外部资源与“网页指纹”（库名称/域名、指纹脚本特征、Set-Cookie属性等）；
  - 可选 **截图视觉嵌入**（后续v2计划）。
- **模型层**：
  - **预训练URL Transformer**（Hugging Face，默认启用，零训练即用）；
  - **融合DNN**（URL Transformer 向量 + 结构/指纹统计特征 + 可选视觉嵌入）。
  - **DALWFR主动学习**：基于不确定性采样 + OOD筛选，便于高性价比标注迭代。
- **服务层**：FastAPI REST + Gradio UI，一键启动可抓取并预测。

## 预训练模型（默认）
- URL分支默认加载 Hugging Face 上的已训模型（可在 `config.py` 修改）：
  - `imanoop7/bert-phishing-detector` 或 `ealvaradob/bert-phishing-url` / `bert-finetuned-phishing`。
- 优点：即刻可用、无需本地训练；缺点：仅URL维度，建议配合抓取特征与后续融合DNN微调。

## 数据与标注
- **正例（活跃钓鱼）**：OpenPhish、PhishTank、URLHaus（脚本见 `scripts/build_dataset.py`）。
- **反例（良性）**：Tranco Top 列表域名、常见品牌官网、Common Crawl 采样。
- **已有开源数据**：UCI PhiUSIIL（带特征）、VisualPhishNet / Phish-IRIS（截图为主，按需手工申请/下载）。
- **切分**：`split_data.py` 按域名去重后再按URL分层抽样，避免“域名泄漏”。

## 目录结构
```
phishguard_v1/
  assets/
    rules/keywords.txt
    rules/libraries.json
  data/                 # 数据输出目录（空）
  models/
  features/
  service/
  scripts/
  requirements.txt
  extras/vision.txt
```

## 许可证
本工程以 **MIT** 开源，供研究与内部评估使用。生产部署请遵循目标站点的 `robots.txt` 与相关法律法规。

