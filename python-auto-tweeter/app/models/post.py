from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.orm import relationship
from ..core.database import Base
import uuid


class Post(Base):
    """投稿モデル"""
    __tablename__ = "posts"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    bot_id = Column(String, ForeignKey("bots.id"))
    twitter_id = Column(String, unique=True, index=True)
    twitter_post_id = Column(String)  # Twitter API v2で取得するID
    content = Column(Text)
    original_content = Column(Text)
    status = Column(String, default="pending")  # pending, posted, failed, simulated
    error_message = Column(Text)  # エラー時のメッセージ
    # ツリー投稿関連
    parent_post_id = Column(String, ForeignKey("posts.id"))  # 親投稿のID
    thread_root_id = Column(String, ForeignKey("posts.id"))  # スレッドのルート投稿ID
    thread_order = Column(Integer, default=0)  # スレッド内での順序
    in_reply_to_tweet_id = Column(String)  # 返信先のTwitter投稿ID
    # 画像関連
    image_urls = Column(Text)  # 投稿に添付された画像のURL（JSON形式で複数保存）
    media_ids = Column(Text)  # Twitter上のメディアID（JSON形式）
    # 統計情報
    retweet_count = Column(Integer, default=0)
    favorite_count = Column(Integer, default=0)
    reply_count = Column(Integer, default=0)
    is_retweet = Column(Boolean, default=False)
    posted_at = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relations
    bot = relationship("Bot", back_populates="posts")
    # ツリー関係
    parent_post = relationship("Post", remote_side=[id], foreign_keys=[parent_post_id])
    thread_root = relationship("Post", remote_side=[id], foreign_keys=[thread_root_id])