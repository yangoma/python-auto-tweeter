from sqlalchemy import Column, String, Boolean, DateTime, func
from sqlalchemy.orm import relationship
from ..core.database import Base
import uuid


class User(Base):
    """ユーザーモデル"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True)
    name = Column(String)
    plan_type = Column(String, default="personal")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relations
    twitter_accounts = relationship("TwitterAccount", back_populates="user", cascade="all, delete-orphan")
    affiliate_settings = relationship("AffiliateSettings", back_populates="user", cascade="all, delete-orphan")