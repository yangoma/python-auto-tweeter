from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from ..core.database import Base
import uuid


class TwitterAccount(Base):
    """Twitterアカウントモデル"""
    __tablename__ = "twitter_accounts"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    twitter_user_id = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    display_name = Column(String)
    profile_image_url = Column(String)
    api_key = Column(String)  # Consumer Key
    api_secret = Column(String)  # Consumer Secret
    access_token = Column(String)
    access_token_secret = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relations
    user = relationship("User", back_populates="twitter_accounts")
    bots = relationship("Bot", back_populates="twitter_account", cascade="all, delete-orphan")