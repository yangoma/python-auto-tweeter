from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from ..core.database import Base
import uuid


class AffiliateSettings(Base):
    """アフィリエイト設定モデル"""
    __tablename__ = "affiliate_settings"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    name = Column(String)
    base_url = Column(String)
    affiliate_id = Column(String)
    conversion_pattern = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relations
    user = relationship("User", back_populates="affiliate_settings")