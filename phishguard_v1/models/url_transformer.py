from __future__ import annotations
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from ..config import settings

class URLTransformer:
    def __init__(self, model_id: str = None, device: str = None):
        self.model_id = model_id or settings.url_model_id
        self.device = device or settings.device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, trust_remote_code=True)
        self.model.eval()
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            self.model.to("cuda")

    @torch.inference_mode()
    def score(self, url: str) -> float:
        # 简单预处理：保证以http开头
        if not url.startswith("http"):
            url = "http://" + url
        inputs = self.tokenizer(url, truncation=True, max_length=256, return_tensors="pt")
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        prob = torch.softmax(logits, dim=-1)[0, -1].item()  # 假设 label 1 为“phishing”
        return float(prob)
