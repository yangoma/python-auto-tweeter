"""分析関連のモデル"""

from sqlalchemy import Column, String, DateTime, Integer, Float, Text, Boolean, ForeignKey, func
from sqlalchemy.orm import relationship
from ..core.database import Base
import uuid


class PostMetrics(Base):
    """投稿メトリクスモデル"""
    __tablename__ = "post_metrics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    post_id = Column(String, ForeignKey("posts.id"), unique=True, index=True)
    
    # エンゲージメント指標
    likes_count = Column(Integer, default=0)
    retweets_count = Column(Integer, default=0)
    replies_count = Column(Integer, default=0)
    quotes_count = Column(Integer, default=0)
    views_count = Column(Integer, default=0)
    
    # 計算指標
    engagement_rate = Column(Float, default=0.0)  # (likes + retweets + replies) / views
    click_rate = Column(Float, default=0.0)  # リンククリック率
    
    # メトリクス更新情報
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now())
    created_at = Column(DateTime, default=func.now())
    
    # Relations
    post = relationship("Post", backref="metrics")


class BotAnalytics(Base):
    """ボット分析データモデル"""
    __tablename__ = "bot_analytics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    bot_id = Column(String, ForeignKey("bots.id"), index=True)
    
    # 期間指定
    date = Column(DateTime, index=True)  # 分析対象日
    period_type = Column(String, default="daily")  # daily, weekly, monthly
    
    # 投稿統計
    total_posts = Column(Integer, default=0)
    successful_posts = Column(Integer, default=0)
    failed_posts = Column(Integer, default=0)
    ai_generated_posts = Column(Integer, default=0)
    manual_posts = Column(Integer, default=0)
    
    # エンゲージメント統計
    total_likes = Column(Integer, default=0)
    total_retweets = Column(Integer, default=0)
    total_replies = Column(Integer, default=0)
    total_views = Column(Integer, default=0)
    
    # 計算指標
    success_rate = Column(Float, default=0.0)  # 成功率
    avg_engagement_rate = Column(Float, default=0.0)  # 平均エンゲージメント率
    best_posting_hour = Column(Integer)  # 最適投稿時間
    
    # システム情報
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relations
    bot = relationship("Bot", backref="analytics")


class ContentAnalysis(Base):
    """コンテンツ分析モデル"""
    __tablename__ = "content_analysis"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    post_id = Column(String, ForeignKey("posts.id"), unique=True, index=True)
    
    # コンテンツ分類
    content_type = Column(String)  # promotional, informational, engaging, question
    has_hashtags = Column(Boolean, default=False)
    hashtag_count = Column(Integer, default=0)
    has_mentions = Column(Boolean, default=False)
    mention_count = Column(Integer, default=0)
    has_links = Column(Boolean, default=False)
    link_count = Column(Integer, default=0)
    has_media = Column(Boolean, default=False)
    media_count = Column(Integer, default=0)
    
    # テキスト分析
    character_count = Column(Integer, default=0)
    word_count = Column(Integer, default=0)
    sentiment_score = Column(Float)  # 感情分析スコア (-1 to 1)
    sentiment_label = Column(String)  # positive, negative, neutral
    
    # キーワード分析
    top_keywords = Column(Text)  # JSON形式でキーワードと出現頻度を保存
    
    # パフォーマンス予測
    predicted_engagement = Column(Float)  # 予測エンゲージメント
    actual_engagement = Column(Float)  # 実際のエンゲージメント
    prediction_accuracy = Column(Float)  # 予測精度
    
    # システム情報
    analyzed_at = Column(DateTime, default=func.now())
    created_at = Column(DateTime, default=func.now())
    
    # Relations
    post = relationship("Post", backref="content_analysis")


class TrendAnalysis(Base):
    """トレンド分析モデル"""
    __tablename__ = "trend_analysis"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # トレンド情報
    trend_date = Column(DateTime, index=True)
    trend_type = Column(String, index=True)  # hashtag, keyword, topic
    trend_value = Column(String)  # ハッシュタグ名、キーワードなど
    
    # 統計情報
    usage_count = Column(Integer, default=0)  # 使用回数
    engagement_score = Column(Float, default=0.0)  # エンゲージメントスコア
    growth_rate = Column(Float, default=0.0)  # 成長率
    
    # トレンド分析
    is_trending = Column(Boolean, default=False)
    trend_strength = Column(Float, default=0.0)  # トレンド強度 (0-1)
    recommended_usage = Column(Boolean, default=False)  # 使用推奨
    
    # システム情報
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class RecommendationLog(Base):
    """分析結果に基づく推奨ログ"""
    __tablename__ = "recommendation_log"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    bot_id = Column(String, ForeignKey("bots.id"), index=True)
    
    # 推奨内容
    recommendation_type = Column(String)  # posting_time, content_type, hashtags
    recommendation_text = Column(Text)
    recommendation_data = Column(Text)  # JSON形式で詳細データを保存
    
    # 推奨根拠
    analysis_basis = Column(Text)  # 分析根拠
    confidence_score = Column(Float, default=0.0)  # 信頼度スコア
    
    # 適用結果
    applied = Column(Boolean, default=False)
    application_result = Column(Text)  # 適用結果
    effectiveness_score = Column(Float)  # 効果スコア
    
    # システム情報
    created_at = Column(DateTime, default=func.now())
    applied_at = Column(DateTime)
    
    # Relations
    bot = relationship("Bot", backref="recommendations")