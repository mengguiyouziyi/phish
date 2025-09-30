# PhishGuard v1 —— 钓鱼网站监测与网页指纹识别（可抓取、可预测、带Web服务）

> 目标：启动即可使用的 **抓取 + 特征抽取 + 预测** 一体化工程，支持后续训练与主动学习（DALWFR思路）。

## 快速开始

### 1) 环境准备（Python 3.10+）
```bash
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate

# 如需走本地代理，先执行一次 scripts/proxy_on.sh（默认指向 127.0.0.1:7890）
# source scripts/proxy_on.sh

# 使用清华源安装依赖（Mac/CPU 会自动落到 CPU 版 PyTorch）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 完成后可执行 scripts/proxy_off.sh 取消代理
# source scripts/proxy_off.sh

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

### 5) 拉取最新 URL 列表（可选）
```bash
# 自动从 OpenPhish / PhishTank / URLHaus 与 Tranco / Majestic 拉取样本
python -m phishguard_v1.pipeline.fetch_feeds \
  --phish-out data/phish_auto.txt \
  --benign-out data/benign_auto.txt \
  --phish-limit 4000 --benign-limit 4000
```

### 6) 快速批量抓取与特征抽取
```bash
# 准备一个URL列表（每行一个URL）
cat > data/urls.txt <<EOF
https://example.com
https://openphish.com/
EOF

# 并发抓取 + 特征抽取，保存到 data/features.parquet
python -m phishguard_v1.scripts.crawl_urls --in data/urls.txt --out data/features.parquet --screenshot 0
```

### 7) 自动采集并构建特征齐全的数据集
> 在 `txt` 中维护良性/钓鱼 URL 列表，脚本将并发抓取、抽取特征并切分数据集。
```bash
# 例：data/benign_urls.txt、data/phish_urls.txt
python -m phishguard_v1.pipeline.build_dataset \
    --benign data/benign_urls.txt \
    --phish data/phish_urls.txt \
    --output data_custom \
    --name custom_seed

# 项目当前默认训练分布可复用 data_hyper/train|val|test.parquet
# 该目录在 data_super 基础上继续融合 data_incremental（600良性/600钓鱼）等增量样本
```

### 8) 训练 DALWFR 融合模型（DNN）
> `phishguard_v1/pipeline/train_dnn.py` 会自动完成标准化、类别加权、指标评估，默认输出到 `artifacts/fusion_dalwfr_v5.pt`。
```bash
python -m phishguard_v1.pipeline.train_dnn \
    --train data_ultra/train.parquet \
    --val data_ultra/val.parquet \
    --epochs 50 \
    --ckpt artifacts/fusion_dalwfr_v5.pt
```

**当前模型表现（`artifacts/fusion_dalwfr_v5.pt`，训练集：data_ultra）：**

- 验证集：Accuracy 0.973，AUC 0.991，Phish F1 0.954
- 测试集：Accuracy 0.975，AUC 0.993，Phish F1 0.957
- 真实 200/200 样本：Accuracy 0.963，AUC 0.98，Phish F1 0.962
- 训练脚本会将完整 `classification_report`、标准化参数等一并写入 checkpoint
- 真实评测报告：`artifacts/eval_sample_report_v5.csv`
- 100 条人工校验清单：`artifacts/manual_eval_v5.csv`

### 9) DALWFR 主动学习闭环（可选）
> 使用新的 `phishguard_v1/pipeline/dalwfr.py` 基于不确定性采样迭代扩充标注集。
```bash
python -m phishguard_v1.pipeline.dalwfr \
    --labeled data_hyper/train.parquet \
    --unlabeled data_hyper/test.parquet \
    --output artifacts/dalwfr_runs/run1 \
    --rounds 5 --query-size 256 --initial 600
```

---

## 架构说明（v1）

- **采集层**：`httpx` 并发抓取（HTTP状态码、完整响应头、Cookie、HTML、外链JS/CSS、Meta等）；可选 `playwright` 渲染截图。
- **特征层**：
  - `feature_engineering.py` 统一构建 90+ 维指纹特征（URL结构、HTTP安全头、Cookie策略、HTML指纹、脚本/样式库、指纹哈希熵等）；
  - 所有数值特征通过 `prepare_features_dataframe` 自动补齐，便于离线/在线复用；
  - 原始 headers/cookies/meta/script 列以 JSON 形式保存在数据集用于审计。
- **模型层**：
  - **预训练 URL Transformer**（默认启用）；
  - **FusionDNN (DALWFR)** —— 最新 `fusion_dalwfr_v5.pt` 在验证集/实测集均维持 0.97 以上准确率，主动学习可持续迭代；
  - **主动学习管线**：`pipeline/dalwfr.py` 提供基于熵的不确定性采样、自动增量标注、轮次检查点。
- **服务层**：FastAPI REST + Gradio UI，一键启动。UI 新增关键指纹特征摘要、HTTP/Cookie/Meta 三大解析面板，API 返回 `feature_snapshot` 并内置 URL 降权策略，显著降低权威站点误报。

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
data_processed/            # 示例最新抓取数据
data_final/                # 原始融合数据留存，可做对比实验
data_full/                 # 历史融合数据（保留对比）
data_super/                # 历史增量融合数据（保留对比）
data_incremental/          # 新增 600/600 增量抓取生成的特征数据
data_hyper/                # 历史交付版本使用的数据集（融合 data_super + data_incremental）
data_ultra/                # 当前训练/验证/测试数据（融合 data_hyper + data_new/external_features）
legacy/                    # 历史训练脚本与旧版数据处理
tools/                     # 调试与外部采集辅助脚本
tests/                     # 测试与验证脚本
```

## 许可证
本工程以 **MIT** 开源，供研究与内部评估使用。生产部署请遵循目标站点的 `robots.txt` 与相关法律法规。
