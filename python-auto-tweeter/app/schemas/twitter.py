from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class TwitterAccountBase(BaseModel):
    """Twitterアカウントベーススキーマ"""
    twitter_user_id: str
    username: str
    display_name: Optional[str] = None


class TwitterAccountCreate(TwitterAccountBase):
    """Twitterアカウント作成スキーマ"""
    access_token: str
    access_token_secret: str


class TwitterAccountUpdate(BaseModel):
    """Twitterアカウント更新スキーマ"""
    display_name: Optional[str] = None
    is_active: Optional[bool] = None


class TwitterAccountResponse(TwitterAccountBase):
    """Twitterアカウントレスポンススキーマ"""
    id: str
    user_id: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True