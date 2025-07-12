from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, JSON, func
from sqlalchemy.orm import relationship
from ..core.database import Base
import uuid


class Bot(Base):
    """ボットモデル"""
    __tablename__ = "bots"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    twitter_account_id = Column(String, ForeignKey("twitter_accounts.id"))
    name = Column(String)
    description = Column(String)
    content_template = Column(String)
    posting_interval = Column(String, default="60")  # 分単位
    max_posts_per_day = Column(String, default="10")
    target_accounts = Column(JSON, default=list)
    post_schedule = Column(JSON, default=dict)
    tweet_templates = Column(JSON, default=list)
    ng_words = Column(JSON, default=list)
    filters = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relations
    twitter_account = relationship("TwitterAccount", back_populates="bots")
    posts = relationship("Post", back_populates="bot", cascade="all, delete-orphan")
    scheduled_posts = relationship("ScheduledPost", back_populates="bot", cascade="all, delete-orphan")