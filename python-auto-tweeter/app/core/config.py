from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """アプリケーション設定"""
    
    # アプリケーション設定
    app_name: str = "Python Auto Tweeter"
    debug: bool = False
    
    # データベース
    database_url: str = "sqlite:///./data/database.db"
    
    # Twitter API
    twitter_api_key: str
    twitter_api_secret: str
    twitter_access_token: Optional[str] = None
    twitter_access_token_secret: Optional[str] = None
    
    # セキュリティ
    secret_key: str
    access_token_expire_minutes: int = 30
    
    # スケジューラー
    scheduler_timezone: str = "Asia/Tokyo"
    
    # ログ
    log_level: str = "INFO"
    log_file: str = "data/logs/app.log"
    
    class Config:
        env_file = ".env"


settings = Settings()