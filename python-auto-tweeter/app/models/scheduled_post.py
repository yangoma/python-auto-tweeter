"""予約投稿モデル"""

from sqlalchemy import Column, String, DateTime, Text, Integer, Boolean, ForeignKey, func
from sqlalchemy.orm import relationship
from ..core.database import Base
import uuid


class ScheduledPost(Base):
    """予約投稿モデル"""
    __tablename__ = "scheduled_posts"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # 投稿内容
    content = Column(Text, nullable=False)
    image_urls = Column(Text)  # JSON形式で複数画像を保存
    
    # スケジュール情報
    scheduled_time = Column(DateTime, nullable=False)
    status = Column(String, default="pending")  # pending, posted, failed, cancelled, processing
    
    # 関連情報
    bot_id = Column(String, ForeignKey("bots.id"), nullable=False)
    
    # Google Sheets連携
    sheet_row_index = Column(Integer)  # スプレッドシートの行番号
    sheet_id = Column(String)  # スプレッドシートID
    
    # 投稿結果
    twitter_post_id = Column(String)  # 投稿成功時のTwitter ID
    error_message = Column(Text)  # エラー時のメッセージ
    posted_at = Column(DateTime)  # 実際の投稿日時
    
    # LLM生成関連
    is_ai_generated = Column(Boolean, default=False)  # AI生成投稿かどうか
    ai_prompt = Column(Text)  # AI生成時のプロンプト
    source_data_id = Column(String)  # 元データ（アフィリエイト商品など）のID
    
    # システム情報
    retry_count = Column(Integer, default=0)  # リトライ回数
    max_retries = Column(Integer, default=3)  # 最大リトライ回数
    
    # タイムスタンプ
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relations
    bot = relationship("Bot", back_populates="scheduled_posts")
    
    def __repr__(self):
        return f"<ScheduledPost(id={self.id}, content={self.content[:50]}..., scheduled_time={self.scheduled_time})>"
    
    @property
    def is_due(self) -> bool:
        """投稿時刻が来ているかチェック"""
        from datetime import datetime, timezone
        return self.scheduled_time <= datetime.now(timezone.utc) and self.status == "pending"
    
    @property
    def can_retry(self) -> bool:
        """リトライ可能かチェック"""
        return self.retry_count < self.max_retries and self.status == "failed"
    
    def to_dict(self) -> dict:
        """辞書形式に変換"""
        import json
        
        image_urls = []
        if self.image_urls:
            try:
                image_urls = json.loads(self.image_urls)
            except:
                image_urls = []
        
        return {
            "id": self.id,
            "content": self.content,
            "image_urls": image_urls,
            "scheduled_time": self.scheduled_time.isoformat() if self.scheduled_time else None,
            "status": self.status,
            "bot_id": self.bot_id,
            "sheet_row_index": self.sheet_row_index,
            "sheet_id": self.sheet_id,
            "twitter_post_id": self.twitter_post_id,
            "error_message": self.error_message,
            "posted_at": self.posted_at.isoformat() if self.posted_at else None,
            "is_ai_generated": self.is_ai_generated,
            "ai_prompt": self.ai_prompt,
            "source_data_id": self.source_data_id,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }