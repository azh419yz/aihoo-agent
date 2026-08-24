"""应用配置管理"""

from pathlib import Path
from urllib.parse import quote_plus

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置，优先从环境变量读取，支持 .env 文件"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- 应用基础配置 ----------
    APP_NAME: str = "AI 中医助手 Agent"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ---------- 阿里云百炼 (DashScope) ----------
    DASHSCOPE_API_KEY: str = ""
    DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 统一模型名称（qwen3.8-max 原生多模态，文本/图片/结构化输出共用）
    LLM_MODEL_NAME: str = "qwen3.8-max"

    # ---------- 百炼知识库 ----------
    # 通用中医行业知识库 ID
    KNOWLEDGE_BASE_GENERAL: str = ""
    # 名医经验知识库 ID
    KNOWLEDGE_BASE_EXPERT: str = ""
    # 百炼业务空间 ID（新 API 必需，从百炼控制台获取）
    BAILIAN_WORKSPACE_ID: str = ""
    # 知识库 Agent ID（用于 /api/v1/indices/knowledge/search 接口）
    KNOWLEDGE_BASE_GENERAL_AGENT_ID: str = ""
    KNOWLEDGE_BASE_EXPERT_AGENT_ID: str = ""

    # ---------- Redis ----------
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_TTL: int = 86400  # 24 小时

    # ---------- MySQL ----------
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = "aihoo_agent"

    # TCM 辨病辨证表已迁移到当前数据库（MYSQL_DATABASE）

    @property
    def MYSQL_DSN(self) -> str:
        """获取异步 MySQL DSN（自动 URL 编码特殊字符）"""
        return (
            f"mysql+aiomysql://{quote_plus(self.MYSQL_USER)}:{quote_plus(self.MYSQL_PASSWORD)}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
            "?charset=utf8mb4"
        )

    @property
    def REDIS_URL(self) -> str:
        """获取 Redis URL"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"


# 全局单例
settings = Settings()

# 项目根目录
ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent
