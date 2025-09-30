from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # 模型 & 抓取配置
    url_model_id: str = Field(default="imanoop7/bert-phishing-detector", description="URL模型ID")
    device: str = Field(default="cpu", description="cuda 或 cpu")
    url_phish_threshold: float = Field(default=0.8, description="URL模型判定钓鱼的概率阈值")
    url_phish_min_threshold: float = Field(default=0.65, description="URL模型触发融合确认的最低钓鱼概率")
    fusion_phish_threshold: float = Field(default=0.6, description="FusionDNN判定钓鱼的概率阈值")
    final_phish_threshold: float = Field(default=0.7, description="最终综合钓鱼概率阈值")
    http_timeout: float = 12.0
    http_max_bytes: int = 2_500_000  # 限制响应体大小
    concurrency: int = 16
    user_agent: str = "PhishGuard/1.0 (+security research; respectful crawler)"
    allow_screenshot: bool = False

    # TLS 与 HTTP2 控制
    tls_verify: bool = Field(default=True, description="是否校验证书（可用环境变量 TLS_VERIFY=0 关闭）")
    http2: bool = Field(default=False, description="某些代理对 HTTP/2 兼容不好")
    retry_on_tls_error: bool = Field(default=True, description="TLS 失败时回退一次")

    # 服务
    host: str = "0.0.0.0"
    port: int = 8000

    # 数据
    data_dir: str = "data"

settings = Settings()
