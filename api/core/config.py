from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """应用配置"""
    # API 配置
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "多模态数据质量检测系统"
    
    # 数据目录配置
    DATA_DIR: str = "data"
    UPLOAD_DIR: str = "data/uploads"
    SAMPLE_DIR: str = "data/samples"
    
    # 环境配置
    CONDA_ENV: str = "multimodal_qa"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


settings = Settings()
