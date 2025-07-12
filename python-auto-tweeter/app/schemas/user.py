from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class UserBase(BaseModel):
    """ユーザーベーススキーマ"""
    email: str
    name: Optional[str] = None
    plan_type: str = "personal"


class UserCreate(UserBase):
    """ユーザー作成スキーマ"""
    pass


class UserUpdate(BaseModel):
    """ユーザー更新スキーマ"""
    name: Optional[str] = None
    plan_type: Optional[str] = None


class UserResponse(UserBase):
    """ユーザーレスポンススキーマ"""
    id: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True