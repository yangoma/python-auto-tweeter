"""予約投稿スケジューラーサービス"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from ..core.config import settings
from ..models import ScheduledPost, Bot, TwitterAccount
from ..api.bots import post_thread
from ..services.sheets_service import sheets_service
import json
import uuid

logger = logging.getLogger(__name__)


class SchedulerService:
    """予約投稿スケジューラーサービス"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.engine = create_engine(settings.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.is_running = False
        
    def start(self):
        """スケジューラーを開始"""
        if not self.is_running:
            self.scheduler.start()
            self.is_running = True
            
            # 定期的な予約投稿チェックジョブを追加
            self.scheduler.add_job(
                self.check_scheduled_posts,
                IntervalTrigger(minutes=1),  # 1分ごとにチェック
                id="check_scheduled_posts",
                replace_existing=True
            )
            
            # 定期的なシート同期ジョブを追加
            self.scheduler.add_job(
                self.sync_with_sheets,
                IntervalTrigger(minutes=10),  # 10分ごとにシート同期
                id="sync_with_sheets", 
                replace_existing=True
            )
            
            logger.info("予約投稿スケジューラーを開始しました")
    
    def stop(self):
        """スケジューラーを停止"""
        if self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("予約投稿スケジューラーを停止しました")
    
    async def check_scheduled_posts(self):
        """期限が来た予約投稿をチェックして実行"""
        db = self.SessionLocal()
        try:
            now = datetime.now(timezone.utc)
            
            # 実行時刻が来ている pending の投稿を取得
            due_posts = db.query(ScheduledPost).filter(
                ScheduledPost.scheduled_time <= now,
                ScheduledPost.status == "pending"
            ).all()
            
            if not due_posts:
                return
            
            logger.info(f"{len(due_posts)} 件の予約投稿を実行します")
            
            for post in due_posts:
                try:
                    await self.execute_scheduled_post(post, db)
                except Exception as e:
                    logger.error(f"予約投稿実行エラー (ID: {post.id}): {e}")
                    
                    # エラー時のステータス更新
                    post.status = "failed"
                    post.error_message = str(e)
                    post.retry_count += 1
                    
                    # リトライ可能かチェック
                    if post.can_retry:
                        # 5分後にリトライ
                        post.scheduled_time = now + timedelta(minutes=5)
                        post.status = "pending"
                        logger.info(f"予約投稿 {post.id} を5分後にリトライします")
                    
                    db.commit()
            
        except Exception as e:
            logger.error(f"予約投稿チェックエラー: {e}")
        finally:
            db.close()
    
    async def execute_scheduled_post(self, scheduled_post: ScheduledPost, db):
        """予約投稿を実行"""
        try:
            # ステータスを processing に更新
            scheduled_post.status = "processing"
            db.commit()
            
            # ボットとTwitterアカウントを取得
            bot = db.query(Bot).filter(Bot.id == scheduled_post.bot_id).first()
            if not bot:
                raise Exception(f"ボット {scheduled_post.bot_id} が見つかりません")
            
            twitter_account = db.query(TwitterAccount).filter(
                TwitterAccount.id == bot.twitter_account_id
            ).first()
            if not twitter_account:
                raise Exception(f"Twitterアカウントが見つかりません")
            
            # 画像URLを処理
            images = []
            if scheduled_post.image_urls:
                try:
                    image_urls = json.loads(scheduled_post.image_urls)
                    for url in image_urls:
                        images.append({"url": url, "filename": f"scheduled_{uuid.uuid4()}.jpg"})
                except:
                    pass
            
            # 投稿を実行
            content_list = [scheduled_post.content]
            result = await post_thread(bot, twitter_account, content_list, db, images)
            
            # 結果に基づいてステータスを更新
            if result.get("successful_count", 0) > 0:
                scheduled_post.status = "posted"
                scheduled_post.twitter_post_id = result.get("thread_id")
                scheduled_post.posted_at = datetime.now(timezone.utc)
                
                # スプレッドシートのステータスも更新
                if scheduled_post.sheet_row_index and sheets_service.is_authenticated():
                    try:
                        sheets_service.update_post_status(
                            "投稿スケジュール",
                            scheduled_post.sheet_row_index - 1,  # 0-based index
                            "posted",
                            result.get("thread_url", "")
                        )
                    except Exception as e:
                        logger.warning(f"シートステータス更新エラー: {e}")
                
                logger.info(f"予約投稿 {scheduled_post.id} を正常に実行しました")
                
            else:
                scheduled_post.status = "failed"
                scheduled_post.error_message = result.get("message", "投稿に失敗しました")
                
                # スプレッドシートにエラーを記録
                if scheduled_post.sheet_row_index and sheets_service.is_authenticated():
                    try:
                        sheets_service.update_post_status(
                            "投稿スケジュール",
                            scheduled_post.sheet_row_index - 1,
                            "failed",
                            scheduled_post.error_message
                        )
                    except Exception as e:
                        logger.warning(f"シートステータス更新エラー: {e}")
            
            db.commit()
            
        except Exception as e:
            scheduled_post.status = "failed"
            scheduled_post.error_message = str(e)
            scheduled_post.retry_count += 1
            db.commit()
            raise
    
    async def sync_with_sheets(self):
        """スプレッドシートとの定期同期"""
        if not sheets_service.is_authenticated():
            return
        
        db = self.SessionLocal()
        try:
            # スプレッドシートから新しい投稿データを取得
            posts_data = sheets_service.get_post_schedule("投稿スケジュール")
            
            synced_count = 0
            
            for row_index, post_data in enumerate(posts_data, start=2):
                try:
                    # 必須フィールドをチェック
                    content = post_data.get('投稿内容') or post_data.get('content')
                    scheduled_time_str = post_data.get('投稿日時') or post_data.get('scheduled_time')
                    bot_id = post_data.get('ボットID') or post_data.get('bot_id')
                    status = post_data.get('ステータス') or post_data.get('status', 'pending')
                    
                    if not all([content, scheduled_time_str, bot_id]) or status in ['posted', 'failed']:
                        continue
                    
                    # 既存の予約投稿をチェック
                    existing_post = db.query(ScheduledPost).filter(
                        ScheduledPost.sheet_row_index == row_index,
                        ScheduledPost.sheet_id == sheets_service.spreadsheet_id
                    ).first()
                    
                    if existing_post and existing_post.status in ['posted', 'processing']:
                        continue  # 既に処理済みまたは処理中の場合はスキップ
                    
                    # 日時変換
                    try:
                        if isinstance(scheduled_time_str, str):
                            scheduled_time = datetime.fromisoformat(scheduled_time_str.replace('Z', '+00:00'))
                        else:
                            scheduled_time = scheduled_time_str
                        
                        if scheduled_time.tzinfo is None:
                            scheduled_time = scheduled_time.replace(tzinfo=timezone.utc)
                    except:
                        continue
                    
                    # ボットの存在確認
                    bot = db.query(Bot).filter(Bot.id == bot_id).first()
                    if not bot:
                        continue
                    
                    if existing_post:
                        # 既存投稿を更新
                        existing_post.content = content
                        existing_post.scheduled_time = scheduled_time
                        existing_post.status = "pending"
                        existing_post.updated_at = datetime.now(timezone.utc)
                    else:
                        # 新しい予約投稿を作成
                        image_urls_json = None
                        image_urls_str = post_data.get('画像URL') or post_data.get('image_urls', '')
                        if image_urls_str:
                            try:
                                image_urls = [url.strip() for url in image_urls_str.split(',') if url.strip()]
                                image_urls_json = json.dumps(image_urls)
                            except:
                                pass
                        
                        scheduled_post = ScheduledPost(
                            id=str(uuid.uuid4()),
                            content=content,
                            scheduled_time=scheduled_time,
                            bot_id=bot_id,
                            sheet_row_index=row_index,
                            sheet_id=sheets_service.spreadsheet_id,
                            image_urls=image_urls_json,
                            status="pending"
                        )
                        
                        db.add(scheduled_post)
                    
                    synced_count += 1
                    
                except Exception as e:
                    logger.warning(f"行 {row_index} の同期でエラー: {e}")
                    continue
            
            if synced_count > 0:
                db.commit()
                logger.info(f"スプレッドシートから {synced_count} 件の投稿を同期しました")
            
        except Exception as e:
            logger.error(f"スプレッドシート同期エラー: {e}")
        finally:
            db.close()
    
    def add_scheduled_post(self, post_id: str, scheduled_time: datetime):
        """個別の予約投稿ジョブを追加"""
        job_id = f"scheduled_post_{post_id}"
        
        self.scheduler.add_job(
            self.execute_single_post,
            DateTrigger(run_date=scheduled_time),
            args=[post_id],
            id=job_id,
            replace_existing=True
        )
        
        logger.info(f"予約投稿ジョブを追加しました: {job_id} ({scheduled_time})")
    
    async def execute_single_post(self, post_id: str):
        """単一の予約投稿を実行"""
        db = self.SessionLocal()
        try:
            scheduled_post = db.query(ScheduledPost).filter(
                ScheduledPost.id == post_id,
                ScheduledPost.status == "pending"
            ).first()
            
            if scheduled_post:
                await self.execute_scheduled_post(scheduled_post, db)
                
        except Exception as e:
            logger.error(f"予約投稿実行エラー (ID: {post_id}): {e}")
        finally:
            db.close()
    
    def remove_scheduled_post(self, post_id: str):
        """予約投稿ジョブを削除"""
        job_id = f"scheduled_post_{post_id}"
        
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"予約投稿ジョブを削除しました: {job_id}")
        except Exception:
            pass  # ジョブが存在しない場合は無視


# シングルトンインスタンス
scheduler_service = SchedulerService()