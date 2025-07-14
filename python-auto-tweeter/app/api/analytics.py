"""分析API エンドポイント"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from datetime import datetime, timezone
import logging

from ..core.database import get_db
from ..services.analytics_service import get_analytics_service
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter()


class AnalyticsRequest(BaseModel):
    """分析リクエスト"""
    bot_id: Optional[str] = None
    days: int = 30
    include_details: bool = True


class TimeRangeRequest(BaseModel):
    """時間範囲指定リクエスト"""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    bot_id: Optional[str] = None


@router.get("/overview")
async def get_analytics_overview(
    db: Session = Depends(get_db),
    bot_id: Optional[str] = Query(None, description="ボットID（指定しない場合は全ボット）"),
    days: int = Query(30, description="分析期間（日数）", ge=1, le=365)
):
    """
    分析概要を取得
    """
    try:
        analytics_service = get_analytics_service(db)
        overview = analytics_service.get_overview_statistics(bot_id=bot_id, days=days)
        
        return {
            "success": True,
            "data": overview,
            "message": f"過去{days}日間の分析データを取得しました"
        }
        
    except Exception as e:
        logger.error(f"分析概要取得エラー: {e}")
        raise HTTPException(status_code=500, detail=f"分析データの取得に失敗しました: {str(e)}")


@router.get("/posting-time")
async def get_posting_time_analysis(
    db: Session = Depends(get_db),
    bot_id: Optional[str] = Query(None, description="ボットID"),
    days: int = Query(30, description="分析期間（日数）", ge=1, le=365)
):
    """
    投稿時間分析を取得
    """
    try:
        analytics_service = get_analytics_service(db)
        time_analysis = analytics_service.get_posting_time_analysis(bot_id=bot_id, days=days)
        
        return {
            "success": True,
            "data": time_analysis,
            "message": "投稿時間分析データを取得しました"
        }
        
    except Exception as e:
        logger.error(f"投稿時間分析エラー: {e}")
        raise HTTPException(status_code=500, detail=f"投稿時間分析に失敗しました: {str(e)}")


@router.get("/content-performance")
async def get_content_performance_analysis(
    db: Session = Depends(get_db),
    bot_id: Optional[str] = Query(None, description="ボットID"),
    days: int = Query(30, description="分析期間（日数）", ge=1, le=365)
):
    """
    コンテンツパフォーマンス分析を取得
    """
    try:
        analytics_service = get_analytics_service(db)
        content_analysis = analytics_service.get_content_performance_analysis(bot_id=bot_id, days=days)
        
        return {
            "success": True,
            "data": content_analysis,
            "message": "コンテンツパフォーマンス分析データを取得しました"
        }
        
    except Exception as e:
        logger.error(f"コンテンツパフォーマンス分析エラー: {e}")
        raise HTTPException(status_code=500, detail=f"コンテンツ分析に失敗しました: {str(e)}")


@router.get("/trends")
async def get_trend_analysis(
    db: Session = Depends(get_db),
    days: int = Query(7, description="分析期間（日数）", ge=1, le=90)
):
    """
    トレンド分析を取得
    """
    try:
        analytics_service = get_analytics_service(db)
        trend_analysis = analytics_service.get_trend_analysis(days=days)
        
        return {
            "success": True,
            "data": trend_analysis,
            "message": "トレンド分析データを取得しました"
        }
        
    except Exception as e:
        logger.error(f"トレンド分析エラー: {e}")
        raise HTTPException(status_code=500, detail=f"トレンド分析に失敗しました: {str(e)}")


@router.get("/recommendations/{bot_id}")
async def get_recommendations(
    bot_id: str,
    db: Session = Depends(get_db)
):
    """
    特定ボットに対する推奨事項を取得
    """
    try:
        analytics_service = get_analytics_service(db)
        recommendations = analytics_service.generate_recommendations(bot_id)
        
        return {
            "success": True,
            "data": {
                "bot_id": bot_id,
                "recommendations": recommendations,
                "total_recommendations": len(recommendations)
            },
            "message": f"ボット {bot_id} の推奨事項を生成しました"
        }
        
    except Exception as e:
        logger.error(f"推奨事項生成エラー: {e}")
        raise HTTPException(status_code=500, detail=f"推奨事項の生成に失敗しました: {str(e)}")


@router.get("/dashboard/{bot_id}")
async def get_bot_dashboard(
    bot_id: str,
    db: Session = Depends(get_db),
    days: int = Query(30, description="分析期間（日数）", ge=1, le=365)
):
    """
    ボット専用ダッシュボードデータを取得
    """
    try:
        analytics_service = get_analytics_service(db)
        
        # 各種分析データを並行して取得
        overview = analytics_service.get_overview_statistics(bot_id=bot_id, days=days)
        time_analysis = analytics_service.get_posting_time_analysis(bot_id=bot_id, days=days)
        content_analysis = analytics_service.get_content_performance_analysis(bot_id=bot_id, days=days)
        recommendations = analytics_service.generate_recommendations(bot_id)
        
        dashboard_data = {
            "bot_id": bot_id,
            "period": overview["period"],
            "overview": overview,
            "posting_time_analysis": time_analysis,
            "content_performance": content_analysis,
            "recommendations": recommendations,
            "summary": {
                "total_posts": overview["posts"]["total"],
                "success_rate": overview["posts"]["success_rate"],
                "avg_engagement": overview["engagement"]["avg_likes"] + overview["engagement"]["avg_retweets"] + overview["engagement"]["avg_replies"],
                "best_posting_hours": time_analysis["best_posting_hours"],
                "top_recommendation": recommendations[0] if recommendations else None
            }
        }
        
        return {
            "success": True,
            "data": dashboard_data,
            "message": f"ボット {bot_id} のダッシュボードデータを取得しました"
        }
        
    except Exception as e:
        logger.error(f"ダッシュボードデータ取得エラー: {e}")
        raise HTTPException(status_code=500, detail=f"ダッシュボードデータの取得に失敗しました: {str(e)}")


@router.get("/comparison")
async def get_bot_comparison(
    db: Session = Depends(get_db),
    bot_ids: str = Query(..., description="比較するボットID（カンマ区切り）"),
    days: int = Query(30, description="分析期間（日数）", ge=1, le=365)
):
    """
    複数ボットの比較分析
    """
    try:
        bot_id_list = [bot_id.strip() for bot_id in bot_ids.split(",")]
        if len(bot_id_list) < 2:
            raise HTTPException(status_code=400, detail="比較には最低2つのボットIDが必要です")
        
        analytics_service = get_analytics_service(db)
        comparison_data = {
            "period": {
                "days": days
            },
            "bots": {},
            "comparison_metrics": {}
        }
        
        bot_data = {}
        for bot_id in bot_id_list:
            overview = analytics_service.get_overview_statistics(bot_id=bot_id, days=days)
            bot_data[bot_id] = overview
            comparison_data["bots"][bot_id] = {
                "bot_id": bot_id,
                "overview": overview
            }
        
        # 比較指標計算
        if len(bot_data) >= 2:
            metrics = ["success_rate", "avg_engagement", "ai_ratio"]
            for metric in metrics:
                values = []
                for bot_id, data in bot_data.items():
                    if metric == "success_rate":
                        values.append(data["posts"]["success_rate"])
                    elif metric == "avg_engagement":
                        avg_eng = data["engagement"]["avg_likes"] + data["engagement"]["avg_retweets"] + data["engagement"]["avg_replies"]
                        values.append(avg_eng)
                    elif metric == "ai_ratio":
                        values.append(data["content_types"]["ai_ratio"])
                
                if values:
                    comparison_data["comparison_metrics"][metric] = {
                        "best_bot": bot_id_list[values.index(max(values))],
                        "best_value": max(values),
                        "worst_bot": bot_id_list[values.index(min(values))],
                        "worst_value": min(values),
                        "average": sum(values) / len(values)
                    }
        
        return {
            "success": True,
            "data": comparison_data,
            "message": f"{len(bot_id_list)}つのボットを比較しました"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ボット比較分析エラー: {e}")
        raise HTTPException(status_code=500, detail=f"ボット比較分析に失敗しました: {str(e)}")


@router.get("/export/{bot_id}")
async def export_analytics_data(
    bot_id: str,
    db: Session = Depends(get_db),
    days: int = Query(30, description="分析期間（日数）", ge=1, le=365),
    format: str = Query("json", description="エクスポート形式", regex="^(json|csv)$")
):
    """
    分析データをエクスポート
    """
    try:
        analytics_service = get_analytics_service(db)
        
        # 全分析データを取得
        export_data = {
            "bot_id": bot_id,
            "export_date": datetime.now(timezone.utc).isoformat(),
            "period_days": days,
            "overview": analytics_service.get_overview_statistics(bot_id=bot_id, days=days),
            "posting_time_analysis": analytics_service.get_posting_time_analysis(bot_id=bot_id, days=days),
            "content_performance": analytics_service.get_content_performance_analysis(bot_id=bot_id, days=days),
            "recommendations": analytics_service.generate_recommendations(bot_id)
        }
        
        if format == "json":
            return {
                "success": True,
                "data": export_data,
                "format": "json",
                "message": "分析データをJSON形式でエクスポートしました"
            }
        elif format == "csv":
            # CSV形式の場合は簡略化されたデータを返す
            csv_data = {
                "summary": {
                    "bot_id": bot_id,
                    "total_posts": export_data["overview"]["posts"]["total"],
                    "success_rate": export_data["overview"]["posts"]["success_rate"],
                    "avg_likes": export_data["overview"]["engagement"]["avg_likes"],
                    "avg_retweets": export_data["overview"]["engagement"]["avg_retweets"],
                    "avg_replies": export_data["overview"]["engagement"]["avg_replies"],
                    "best_posting_hours": ",".join(map(str, export_data["posting_time_analysis"]["best_posting_hours"]))
                }
            }
            return {
                "success": True,
                "data": csv_data,
                "format": "csv",
                "message": "分析データをCSV形式でエクスポートしました"
            }
        
    except Exception as e:
        logger.error(f"分析データエクスポートエラー: {e}")
        raise HTTPException(status_code=500, detail=f"データエクスポートに失敗しました: {str(e)}")


@router.get("/health")
async def analytics_health_check(db: Session = Depends(get_db)):
    """
    分析機能のヘルスチェック
    """
    try:
        analytics_service = get_analytics_service(db)
        
        # 基本的な動作確認
        test_overview = analytics_service.get_overview_statistics(days=1)
        
        return {
            "success": True,
            "status": "healthy",
            "message": "分析機能は正常に動作しています",
            "test_data_available": test_overview["posts"]["total"] > 0
        }
        
    except Exception as e:
        logger.error(f"分析機能ヘルスチェックエラー: {e}")
        return {
            "success": False,
            "status": "unhealthy",
            "message": f"分析機能に問題があります: {str(e)}"
        }