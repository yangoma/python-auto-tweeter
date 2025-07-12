from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class BotBase(BaseModel):
    """ボットベーススキーマ"""
    name: str
    target_accounts: List[str] = []
    post_schedule: Dict[str, Any] = {}
    tweet_templates: List[str] = []
    ng_words: List[str] = []
    filters: Dict[str, Any] = {}


class BotCreate(BotBase):
    """ボット作成スキーマ"""
    twitter_account_id: str


class BotUpdate(BaseModel):
    """ボット更新スキーマ"""
    name: Optional[str] = None
    target_accounts: Optional[List[str]] = None
    post_schedule: Optional[Dict[str, Any]] = None
    tweet_templates: Optional[List[str]] = None
    ng_words: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class BotResponse(BotBase):
    """ボットレスポンススキーマ"""
    id: str
    twitter_account_id: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True