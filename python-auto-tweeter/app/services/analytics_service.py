"""分析サービス"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc, asc
import json
import re
from collections import Counter

from ..models.post import Post
from ..models.scheduled_post import ScheduledPost
from ..models.bot import Bot
from ..models.analytics import PostMetrics, BotAnalytics, ContentAnalysis, TrendAnalysis, RecommendationLog

logger = logging.getLogger(__name__)


class AnalyticsService:
    """分析サービス"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_overview_statistics(self, bot_id: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """
        概要統計を取得
        
        Args:
            bot_id: ボットID（指定しない場合は全ボット）
            days: 分析期間（日数）
            
        Returns:
            Dict: 概要統計データ
        """
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        # 基本クエリ
        posts_query = self.db.query(Post)
        scheduled_query = self.db.query(ScheduledPost)
        
        if bot_id:
            posts_query = posts_query.filter(Post.bot_id == bot_id)
            scheduled_query = scheduled_query.filter(ScheduledPost.bot_id == bot_id)
        
        # 期間フィルタ
        posts_query = posts_query.filter(Post.created_at.between(start_date, end_date))
        scheduled_query = scheduled_query.filter(ScheduledPost.created_at.between(start_date, end_date))
        
        # 統計計算
        total_posts = posts_query.count()
        successful_posts = posts_query.filter(Post.status == "posted").count()
        failed_posts = posts_query.filter(Post.status == "failed").count()
        
        total_scheduled = scheduled_query.count()
        scheduled_posted = scheduled_query.filter(ScheduledPost.status == "posted").count()
        scheduled_pending = scheduled_query.filter(ScheduledPost.status == "pending").count()
        scheduled_failed = scheduled_query.filter(ScheduledPost.status == "failed").count()
        
        # エンゲージメント統計
        engagement_stats = self.db.query(
            func.sum(Post.favorite_count).label('total_likes'),
            func.sum(Post.retweet_count).label('total_retweets'),
            func.sum(Post.reply_count).label('total_replies'),
            func.avg(Post.favorite_count).label('avg_likes'),
            func.avg(Post.retweet_count).label('avg_retweets'),
            func.avg(Post.reply_count).label('avg_replies')
        ).filter(
            Post.status == "posted",
            Post.created_at.between(start_date, end_date)
        )
        
        if bot_id:
            engagement_stats = engagement_stats.filter(Post.bot_id == bot_id)
        
        engagement_result = engagement_stats.first()
        
        # AI生成 vs 手動投稿
        ai_posts = scheduled_query.filter(ScheduledPost.is_ai_generated == True).count()
        manual_posts = scheduled_query.filter(ScheduledPost.is_ai_generated == False).count()
        
        return {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days
            },
            "posts": {
                "total": total_posts,
                "successful": successful_posts,
                "failed": failed_posts,
                "success_rate": (successful_posts / total_posts * 100) if total_posts > 0 else 0
            },
            "scheduled_posts": {
                "total": total_scheduled,
                "posted": scheduled_posted,
                "pending": scheduled_pending,
                "failed": scheduled_failed,
                "success_rate": (scheduled_posted / total_scheduled * 100) if total_scheduled > 0 else 0
            },
            "engagement": {
                "total_likes": int(engagement_result.total_likes or 0),
                "total_retweets": int(engagement_result.total_retweets or 0),
                "total_replies": int(engagement_result.total_replies or 0),
                "avg_likes": float(engagement_result.avg_likes or 0),
                "avg_retweets": float(engagement_result.avg_retweets or 0),
                "avg_replies": float(engagement_result.avg_replies or 0)
            },
            "content_types": {
                "ai_generated": ai_posts,
                "manual": manual_posts,
                "ai_ratio": (ai_posts / (ai_posts + manual_posts) * 100) if (ai_posts + manual_posts) > 0 else 0
            }
        }
    
    def get_posting_time_analysis(self, bot_id: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """
        投稿時間分析
        
        Args:
            bot_id: ボットID
            days: 分析期間
            
        Returns:
            Dict: 時間別投稿分析データ
        """
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        query = self.db.query(
            func.extract('hour', Post.posted_at).label('hour'),
            func.count(Post.id).label('post_count'),
            func.avg(Post.favorite_count + Post.retweet_count + Post.reply_count).label('avg_engagement'),
            func.sum(Post.favorite_count).label('total_likes'),
            func.sum(Post.retweet_count).label('total_retweets'),
            func.sum(Post.reply_count).label('total_replies')
        ).filter(
            Post.status == "posted",
            Post.posted_at.between(start_date, end_date)
        ).group_by(func.extract('hour', Post.posted_at))
        
        if bot_id:
            query = query.filter(Post.bot_id == bot_id)
        
        results = query.all()
        
        # 時間別データを整理
        hourly_data = {}
        for result in results:
            hour = int(result.hour)
            hourly_data[hour] = {
                "hour": hour,
                "post_count": result.post_count,
                "avg_engagement": float(result.avg_engagement or 0),
                "total_likes": int(result.total_likes or 0),
                "total_retweets": int(result.total_retweets or 0),
                "total_replies": int(result.total_replies or 0)
            }
        
        # 全24時間分のデータを作成（データがない時間は0で埋める）
        complete_hourly_data = []
        for hour in range(24):
            if hour in hourly_data:
                complete_hourly_data.append(hourly_data[hour])
            else:
                complete_hourly_data.append({
                    "hour": hour,
                    "post_count": 0,
                    "avg_engagement": 0.0,
                    "total_likes": 0,
                    "total_retweets": 0,
                    "total_replies": 0
                })
        
        # 最適投稿時間を算出
        best_hours = sorted(
            [data for data in complete_hourly_data if data["post_count"] > 0],
            key=lambda x: x["avg_engagement"],
            reverse=True
        )[:3]
        
        return {
            "hourly_data": complete_hourly_data,
            "best_posting_hours": [data["hour"] for data in best_hours],
            "recommendations": [
                f"{data['hour']}時: 平均エンゲージメント {data['avg_engagement']:.1f}"
                for data in best_hours
            ]
        }
    
    def get_content_performance_analysis(self, bot_id: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """
        コンテンツパフォーマンス分析
        
        Args:
            bot_id: ボットID
            days: 分析期間
            
        Returns:
            Dict: コンテンツ分析データ
        """
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        query = self.db.query(Post).filter(
            Post.status == "posted",
            Post.posted_at.between(start_date, end_date)
        )
        
        if bot_id:
            query = query.filter(Post.bot_id == bot_id)
        
        posts = query.all()
        
        # コンテンツ分析
        analysis_results = {
            "total_posts": len(posts),
            "content_types": {},
            "hashtag_analysis": {},
            "link_analysis": {},
            "media_analysis": {},
            "length_analysis": {},
            "top_performing_posts": []
        }
        
        hashtag_performance = {}
        link_posts = []
        media_posts = []
        lengths = []
        
        for post in posts:
            content = post.content or ""
            engagement = (post.favorite_count or 0) + (post.retweet_count or 0) + (post.reply_count or 0)
            
            # 長さ分析
            lengths.append({
                "length": len(content),
                "engagement": engagement
            })
            
            # ハッシュタグ分析
            hashtags = re.findall(r'#\w+', content)
            for hashtag in hashtags:
                if hashtag not in hashtag_performance:
                    hashtag_performance[hashtag] = {"count": 0, "total_engagement": 0}
                hashtag_performance[hashtag]["count"] += 1
                hashtag_performance[hashtag]["total_engagement"] += engagement
            
            # リンク分析
            if re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content):
                link_posts.append(engagement)
            
            # メディア分析
            if post.image_urls:
                try:
                    image_urls = json.loads(post.image_urls) if isinstance(post.image_urls, str) else post.image_urls
                    if image_urls:
                        media_posts.append(engagement)
                except:
                    pass
        
        # ハッシュタグパフォーマンス計算
        hashtag_stats = {}
        for hashtag, data in hashtag_performance.items():
            hashtag_stats[hashtag] = {
                "count": data["count"],
                "avg_engagement": data["total_engagement"] / data["count"] if data["count"] > 0 else 0
            }
        
        # 長さ別パフォーマンス
        short_posts = [p for p in lengths if p["length"] <= 50]
        medium_posts = [p for p in lengths if 50 < p["length"] <= 150]
        long_posts = [p for p in lengths if p["length"] > 150]
        
        analysis_results.update({
            "hashtag_analysis": {
                "total_hashtags": len(hashtag_stats),
                "top_hashtags": sorted(
                    hashtag_stats.items(),
                    key=lambda x: x[1]["avg_engagement"],
                    reverse=True
                )[:10]
            },
            "link_analysis": {
                "posts_with_links": len(link_posts),
                "avg_engagement_with_links": sum(link_posts) / len(link_posts) if link_posts else 0,
                "avg_engagement_without_links": sum([p["engagement"] for p in lengths if p not in link_posts]) / max(1, len(lengths) - len(link_posts))
            },
            "media_analysis": {
                "posts_with_media": len(media_posts),
                "avg_engagement_with_media": sum(media_posts) / len(media_posts) if media_posts else 0,
                "avg_engagement_without_media": sum([p["engagement"] for p in lengths if p not in media_posts]) / max(1, len(lengths) - len(media_posts))
            },
            "length_analysis": {
                "short_posts": {
                    "count": len(short_posts),
                    "avg_engagement": sum([p["engagement"] for p in short_posts]) / len(short_posts) if short_posts else 0
                },
                "medium_posts": {
                    "count": len(medium_posts),
                    "avg_engagement": sum([p["engagement"] for p in medium_posts]) / len(medium_posts) if medium_posts else 0
                },
                "long_posts": {
                    "count": len(long_posts),
                    "avg_engagement": sum([p["engagement"] for p in long_posts]) / len(long_posts) if long_posts else 0
                }
            }
        })
        
        # トップパフォーマンス投稿
        top_posts = sorted(posts, key=lambda p: (p.favorite_count or 0) + (p.retweet_count or 0) + (p.reply_count or 0), reverse=True)[:5]
        analysis_results["top_performing_posts"] = [
            {
                "id": post.id,
                "content": post.content[:100] + "..." if len(post.content) > 100 else post.content,
                "engagement": (post.favorite_count or 0) + (post.retweet_count or 0) + (post.reply_count or 0),
                "likes": post.favorite_count or 0,
                "retweets": post.retweet_count or 0,
                "replies": post.reply_count or 0,
                "posted_at": post.posted_at.isoformat() if post.posted_at else None
            }
            for post in top_posts
        ]
        
        return analysis_results
    
    def get_trend_analysis(self, days: int = 7) -> Dict[str, Any]:
        """
        トレンド分析
        
        Args:
            days: 分析期間
            
        Returns:
            Dict: トレンド分析データ
        """
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        # 期間別投稿数推移
        daily_posts = self.db.query(
            func.date(Post.posted_at).label('date'),
            func.count(Post.id).label('post_count'),
            func.sum(Post.favorite_count + Post.retweet_count + Post.reply_count).label('total_engagement')
        ).filter(
            Post.status == "posted",
            Post.posted_at.between(start_date, end_date)
        ).group_by(func.date(Post.posted_at)).order_by(func.date(Post.posted_at)).all()
        
        # 投稿推移データ
        trend_data = []
        for result in daily_posts:
            trend_data.append({
                "date": result.date.isoformat(),
                "post_count": result.post_count,
                "total_engagement": int(result.total_engagement or 0),
                "avg_engagement": float(result.total_engagement / result.post_count) if result.post_count > 0 else 0
            })
        
        # ハッシュタグトレンド
        hashtag_trends = self._analyze_hashtag_trends(start_date, end_date)
        
        return {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days
            },
            "posting_trends": trend_data,
            "hashtag_trends": hashtag_trends,
            "growth_metrics": self._calculate_growth_metrics(trend_data)
        }
    
    def _analyze_hashtag_trends(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """ハッシュタグトレンド分析"""
        posts = self.db.query(Post).filter(
            Post.status == "posted",
            Post.posted_at.between(start_date, end_date)
        ).all()
        
        hashtag_data = {}
        for post in posts:
            content = post.content or ""
            hashtags = re.findall(r'#\w+', content)
            engagement = (post.favorite_count or 0) + (post.retweet_count or 0) + (post.reply_count or 0)
            
            for hashtag in hashtags:
                if hashtag not in hashtag_data:
                    hashtag_data[hashtag] = {"count": 0, "total_engagement": 0, "posts": []}
                hashtag_data[hashtag]["count"] += 1
                hashtag_data[hashtag]["total_engagement"] += engagement
                hashtag_data[hashtag]["posts"].append(post.posted_at)
        
        # トレンド強度計算
        trending_hashtags = []
        for hashtag, data in hashtag_data.items():
            if data["count"] >= 2:  # 最低2回使用されたハッシュタグのみ
                avg_engagement = data["total_engagement"] / data["count"]
                # 時間的分散も考慮（最近使用されているかどうか）
                recent_usage = sum(1 for post_time in data["posts"] 
                                 if post_time > end_date - timedelta(days=2))
                trend_strength = avg_engagement * (1 + recent_usage / data["count"])
                
                trending_hashtags.append({
                    "hashtag": hashtag,
                    "usage_count": data["count"],
                    "avg_engagement": avg_engagement,
                    "trend_strength": trend_strength,
                    "recent_usage": recent_usage
                })
        
        return sorted(trending_hashtags, key=lambda x: x["trend_strength"], reverse=True)[:10]
    
    def _calculate_growth_metrics(self, trend_data: List[Dict]) -> Dict:
        """成長指標計算"""
        if len(trend_data) < 2:
            return {"post_growth": 0, "engagement_growth": 0}
        
        # 前半と後半で比較
        mid_point = len(trend_data) // 2
        first_half = trend_data[:mid_point]
        second_half = trend_data[mid_point:]
        
        first_half_avg_posts = sum(d["post_count"] for d in first_half) / len(first_half)
        second_half_avg_posts = sum(d["post_count"] for d in second_half) / len(second_half)
        
        first_half_avg_engagement = sum(d["avg_engagement"] for d in first_half) / len(first_half)
        second_half_avg_engagement = sum(d["avg_engagement"] for d in second_half) / len(second_half)
        
        post_growth = ((second_half_avg_posts - first_half_avg_posts) / first_half_avg_posts * 100) if first_half_avg_posts > 0 else 0
        engagement_growth = ((second_half_avg_engagement - first_half_avg_engagement) / first_half_avg_engagement * 100) if first_half_avg_engagement > 0 else 0
        
        return {
            "post_growth": round(post_growth, 2),
            "engagement_growth": round(engagement_growth, 2)
        }
    
    def generate_recommendations(self, bot_id: str) -> List[Dict[str, Any]]:
        """
        分析結果に基づく推奨事項を生成
        
        Args:
            bot_id: ボットID
            
        Returns:
            List[Dict]: 推奨事項リスト
        """
        recommendations = []
        
        # 投稿時間分析による推奨
        time_analysis = self.get_posting_time_analysis(bot_id, days=30)
        if time_analysis["best_posting_hours"]:
            recommendations.append({
                "type": "posting_time",
                "title": "最適投稿時間の提案",
                "description": f"エンゲージメントが高い時間帯: {', '.join(map(str, time_analysis['best_posting_hours']))}時",
                "priority": "high",
                "data": time_analysis["best_posting_hours"]
            })
        
        # コンテンツ分析による推奨
        content_analysis = self.get_content_performance_analysis(bot_id, days=30)
        
        # メディア使用推奨
        media_with = content_analysis["media_analysis"]["avg_engagement_with_media"]
        media_without = content_analysis["media_analysis"]["avg_engagement_without_media"]
        if media_with > media_without * 1.2:
            recommendations.append({
                "type": "media_usage",
                "title": "画像・動画の使用推奨",
                "description": f"メディア付き投稿のエンゲージメントが{((media_with / media_without - 1) * 100):.1f}%高くなっています",
                "priority": "medium",
                "data": {"improvement": (media_with / media_without - 1) * 100}
            })
        
        # ハッシュタグ推奨
        if content_analysis["hashtag_analysis"]["top_hashtags"]:
            top_hashtag = content_analysis["hashtag_analysis"]["top_hashtags"][0]
            recommendations.append({
                "type": "hashtag_usage",
                "title": "効果的なハッシュタグの活用",
                "description": f"「{top_hashtag[0]}」の使用を推奨（平均エンゲージメント: {top_hashtag[1]['avg_engagement']:.1f}）",
                "priority": "low",
                "data": {"hashtag": top_hashtag[0], "avg_engagement": top_hashtag[1]["avg_engagement"]}
            })
        
        return recommendations


# シングルトンインスタンス用のファクトリー
def get_analytics_service(db: Session) -> AnalyticsService:
    """AnalyticsServiceのインスタンスを取得"""
    return AnalyticsService(db)