"""Google Sheets連携API"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel
import json
import uuid
import os

from ..core.database import get_db
from ..models import Bot, ScheduledPost
from ..services.sheets_service import sheets_service
from .dependencies import get_current_user

router = APIRouter()

# アプリケーション設定の管理
APP_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "app_config.json")

def load_app_config() -> dict:
    """アプリケーション設定を読み込み"""
    try:
        if os.path.exists(APP_CONFIG_FILE):
            with open(APP_CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"設定ファイル読み込みエラー: {e}")
    return {}

def save_app_config(new_config: dict) -> bool:
    """アプリケーション設定を保存"""
    try:
        # 既存設定を読み込み
        config = load_app_config()
        
        # 新しい設定をマージ
        config.update(new_config)
        
        # 設定ファイルに保存
        os.makedirs(os.path.dirname(APP_CONFIG_FILE), exist_ok=True)
        with open(APP_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"設定ファイル保存エラー: {e}")
        return False


class SheetsConfig(BaseModel):
    """スプレッドシート設定"""
    spreadsheet_id: str
    credentials_json: Optional[str] = None


@router.post("/upload-credentials")
async def upload_credentials(file: UploadFile = File(...)):
    """Google API認証ファイルをアップロード"""
    try:
        # ファイル形式チェック
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="JSONファイルのみアップロード可能です")
        
        # 認証ファイル保存ディレクトリ
        credentials_dir = os.path.join(os.path.dirname(__file__), "..", "..", "credentials")
        os.makedirs(credentials_dir, exist_ok=True)
        
        # ファイル保存
        credentials_path = os.path.join(credentials_dir, "google_credentials.json")
        contents = await file.read()
        
        # JSONとして有効かチェック
        try:
            credentials_data = json.loads(contents.decode('utf-8'))
            # サービスアカウントの必要なキーが存在するかチェック
            required_keys = ["type", "project_id", "private_key_id", "private_key", "client_email"]
            if not all(key in credentials_data for key in required_keys):
                raise HTTPException(status_code=400, detail="無効なGoogle Service Account認証ファイルです")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="無効なJSONファイルです")
        
        # ファイル保存
        with open(credentials_path, 'wb') as f:
            f.write(contents)
        
        # 環境変数を更新
        os.environ['GOOGLE_CREDENTIALS_PATH'] = credentials_path
        
        # 設定ファイルに保存
        save_app_config({'google_credentials_path': credentials_path})
        
        # サービスを再初期化
        sheets_service.__init__(credentials_path=credentials_path)
        
        return {
            "message": "認証ファイルをアップロードしました",
            "filename": file.filename,
            "credentials_path": credentials_path
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"アップロードエラー: {str(e)}")


@router.post("/configure")
async def configure_sheets(config: SheetsConfig):
    """スプレッドシート設定"""
    try:
        # スプレッドシートIDを設定
        sheets_service.spreadsheet_id = config.spreadsheet_id
        
        # 設定ファイルに保存
        save_app_config({'spreadsheet_id': config.spreadsheet_id})
        
        # 認証情報が提供されている場合は保存
        if config.credentials_json:
            try:
                credentials_data = json.loads(config.credentials_json)
                credentials_dir = os.path.join(os.path.dirname(__file__), "..", "..", "credentials")
                os.makedirs(credentials_dir, exist_ok=True)
                
                credentials_path = os.path.join(credentials_dir, "google_credentials.json")
                with open(credentials_path, 'w', encoding='utf-8') as f:
                    json.dump(credentials_data, f, ensure_ascii=False, indent=2)
                
                # 環境変数を更新
                os.environ['GOOGLE_CREDENTIALS_PATH'] = credentials_path
                
                # 設定ファイルに保存
                save_app_config({'google_credentials_path': credentials_path})
                
                # サービスを再初期化
                sheets_service.__init__(credentials_path=credentials_path, spreadsheet_id=config.spreadsheet_id)
                
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="無効なJSON認証情報です")
        
        # 接続テスト
        test_result = sheets_service.read_sheet("投稿スケジュール", "A1:A1")
        
        return {
            "message": "スプレッドシート設定を完了しました",
            "spreadsheet_id": config.spreadsheet_id,
            "connection_test": "成功" if test_result is not None else "失敗"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"設定エラー: {str(e)}")


@router.get("/config")
async def get_config():
    """現在の設定を取得"""
    # 保存された設定を読み込み
    saved_config = load_app_config()
    
    # 現在の設定と保存された設定をマージ
    spreadsheet_id = (getattr(sheets_service, 'spreadsheet_id', '') or 
                     saved_config.get('spreadsheet_id', ''))
    credentials_path = (os.environ.get('GOOGLE_CREDENTIALS_PATH', '') or 
                       saved_config.get('google_credentials_path', ''))
    
    return {
        "spreadsheet_id": spreadsheet_id,
        "credentials_configured": sheets_service.is_authenticated(),
        "credentials_path": credentials_path,
        "saved_config": saved_config
    }


@router.get("/test-connection")
async def test_sheets_connection():
    """Google Sheets API接続テスト"""
    try:
        if not sheets_service.is_authenticated():
            return {
                "connected": False,
                "message": "Google Sheets API認証が設定されていません",
                "setup_required": True
            }
        
        # 簡単な読み取りテストを実行
        test_data = sheets_service.read_sheet("投稿スケジュール", "A1:A1")
        
        return {
            "connected": True,
            "message": "Google Sheets API接続成功",
            "test_result": test_data
        }
        
    except Exception as e:
        return {
            "connected": False,
            "message": f"接続エラー: {str(e)}",
            "error": True
        }


@router.post("/sync-from-sheets")
async def sync_from_sheets(
    background_tasks: BackgroundTasks,
    sheet_name: str = "投稿スケジュール",
    db: Session = Depends(get_db)
):
    """スプレッドシートから投稿データを同期"""
    try:
        if not sheets_service.is_authenticated():
            raise HTTPException(
                status_code=400, 
                detail="Google Sheets API認証が必要です"
            )
        
        # スプレッドシートから投稿スケジュールを取得
        posts_data = sheets_service.get_post_schedule(sheet_name)
        
        if not posts_data:
            return {
                "message": "スプレッドシートにデータが見つかりませんでした",
                "synced_count": 0
            }
        
        synced_count = 0
        errors = []
        
        for row_index, post_data in enumerate(posts_data, start=2):  # ヘッダー行をスキップ
            try:
                # 必須フィールドをチェック
                content = post_data.get('投稿内容') or post_data.get('content')
                scheduled_time_str = post_data.get('投稿日時') or post_data.get('scheduled_time')
                bot_id = post_data.get('ボットID') or post_data.get('bot_id')
                
                if not all([content, scheduled_time_str, bot_id]):
                    errors.append(f"行 {row_index}: 必須フィールドが不足しています")
                    continue
                
                # 日時変換
                try:
                    if isinstance(scheduled_time_str, str):
                        scheduled_time = datetime.fromisoformat(scheduled_time_str.replace('Z', '+00:00'))
                    else:
                        scheduled_time = scheduled_time_str
                    
                    if scheduled_time.tzinfo is None:
                        scheduled_time = scheduled_time.replace(tzinfo=timezone.utc)
                        
                except Exception:
                    errors.append(f"行 {row_index}: 投稿日時の形式が正しくありません")
                    continue
                
                # ボットの存在確認
                bot = db.query(Bot).filter(Bot.id == bot_id).first()
                if not bot:
                    errors.append(f"行 {row_index}: ボットID {bot_id} が見つかりません")
                    continue
                
                # 既存の予約投稿をチェック（行番号で重複回避）
                existing_post = db.query(ScheduledPost).filter(
                    ScheduledPost.sheet_row_index == row_index,
                    ScheduledPost.sheet_id == sheets_service.spreadsheet_id
                ).first()
                
                if existing_post:
                    # 既存投稿を更新
                    existing_post.content = content
                    existing_post.scheduled_time = scheduled_time
                    existing_post.bot_id = bot_id
                    
                    # 画像URLの処理
                    image_urls_str = post_data.get('画像URL') or post_data.get('image_urls', '')
                    if image_urls_str:
                        try:
                            # カンマ区切りのURLを配列に変換
                            image_urls = [url.strip() for url in image_urls_str.split(',') if url.strip()]
                            existing_post.image_urls = json.dumps(image_urls)
                        except:
                            existing_post.image_urls = None
                    
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
                errors.append(f"行 {row_index}: {str(e)}")
                continue
        
        db.commit()
        
        # バックグラウンドタスクでスケジューラーを更新
        background_tasks.add_task(update_scheduler)
        
        return {
            "message": f"{synced_count} 件の投稿を同期しました",
            "synced_count": synced_count,
            "errors": errors
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"同期エラー: {str(e)}")


@router.post("/sync-to-sheets")
async def sync_to_sheets(
    sheet_name: str = "投稿スケジュール",
    db: Session = Depends(get_db)
):
    """データベースからスプレッドシートへ同期"""
    try:
        if not sheets_service.is_authenticated():
            raise HTTPException(
                status_code=400, 
                detail="Google Sheets API認証が必要です"
            )
        
        # データベースから予約投稿を取得
        scheduled_posts = db.query(ScheduledPost).order_by(ScheduledPost.scheduled_time).all()
        
        # ヘッダー行を作成
        headers = [
            "ID", "投稿内容", "画像URL", "投稿日時", "ボットID", "ステータス", "投稿結果", "作成日時"
        ]
        
        # データ行を作成
        data_rows = [headers]
        
        for post in scheduled_posts:
            # 画像URLを文字列に変換
            image_urls = ""
            if post.image_urls:
                try:
                    urls = json.loads(post.image_urls)
                    image_urls = ", ".join(urls)
                except:
                    pass
            
            row = [
                post.id,
                post.content,
                image_urls,
                post.scheduled_time.isoformat() if post.scheduled_time else "",
                post.bot_id,
                post.status,
                post.error_message or "",
                post.created_at.isoformat() if post.created_at else ""
            ]
            data_rows.append(row)
        
        # スプレッドシートに書き込み
        success = sheets_service.write_sheet(sheet_name, data_rows)
        
        if success:
            return {
                "message": f"{len(scheduled_posts)} 件の投稿をスプレッドシートに同期しました",
                "synced_count": len(scheduled_posts)
            }
        else:
            raise HTTPException(status_code=500, detail="スプレッドシートへの書き込みに失敗しました")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"同期エラー: {str(e)}")


@router.get("/scheduled-posts")
async def get_scheduled_posts(
    status: Optional[str] = None,
    bot_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """予約投稿一覧を取得"""
    try:
        query = db.query(ScheduledPost)
        
        if status:
            query = query.filter(ScheduledPost.status == status)
        
        if bot_id:
            query = query.filter(ScheduledPost.bot_id == bot_id)
        
        total = query.count()
        scheduled_posts = query.order_by(ScheduledPost.scheduled_time.desc()).offset(offset).limit(limit).all()
        
        posts_data = []
        for post in scheduled_posts:
            post_dict = post.to_dict()
            
            # ボット情報を追加
            bot = db.query(Bot).filter(Bot.id == post.bot_id).first()
            post_dict['bot_name'] = bot.name if bot else "Unknown Bot"
            
            posts_data.append(post_dict)
        
        return {
            "posts": posts_data,
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取得エラー: {str(e)}")


@router.post("/scheduled-posts")
async def create_scheduled_post(
    post_data: dict,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """予約投稿を作成"""
    try:
        # 必須フィールドの検証
        required_fields = ["content", "scheduled_time", "bot_id"]
        for field in required_fields:
            if field not in post_data:
                raise HTTPException(status_code=400, detail=f"必須フィールド '{field}' が不足しています")
        
        # 日時変換
        try:
            scheduled_time = datetime.fromisoformat(post_data["scheduled_time"].replace('Z', '+00:00'))
            if scheduled_time.tzinfo is None:
                scheduled_time = scheduled_time.replace(tzinfo=timezone.utc)
        except Exception:
            raise HTTPException(status_code=400, detail="投稿日時の形式が正しくありません")
        
        # ボットの存在確認
        bot = db.query(Bot).filter(Bot.id == post_data["bot_id"]).first()
        if not bot:
            raise HTTPException(status_code=404, detail="指定されたボットが見つかりません")
        
        # 画像URLの処理
        image_urls_json = None
        if "image_urls" in post_data and post_data["image_urls"]:
            try:
                image_urls_json = json.dumps(post_data["image_urls"])
            except:
                pass
        
        # 予約投稿を作成
        scheduled_post = ScheduledPost(
            id=str(uuid.uuid4()),
            content=post_data["content"],
            scheduled_time=scheduled_time,
            bot_id=post_data["bot_id"],
            image_urls=image_urls_json,
            status="pending",
            is_ai_generated=post_data.get("is_ai_generated", False),
            ai_prompt=post_data.get("ai_prompt"),
            source_data_id=post_data.get("source_data_id")
        )
        
        db.add(scheduled_post)
        db.commit()
        db.refresh(scheduled_post)
        
        # バックグラウンドタスクでスケジューラーを更新
        background_tasks.add_task(update_scheduler)
        
        return {
            "message": "予約投稿を作成しました",
            "post": scheduled_post.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"作成エラー: {str(e)}")


@router.put("/scheduled-posts/{post_id}")
async def update_scheduled_post(
    post_id: str,
    post_data: dict,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """予約投稿を更新"""
    try:
        scheduled_post = db.query(ScheduledPost).filter(ScheduledPost.id == post_id).first()
        if not scheduled_post:
            raise HTTPException(status_code=404, detail="予約投稿が見つかりません")
        
        # フィールドを更新
        if "content" in post_data:
            scheduled_post.content = post_data["content"]
        
        if "scheduled_time" in post_data:
            try:
                scheduled_time = datetime.fromisoformat(post_data["scheduled_time"].replace('Z', '+00:00'))
                if scheduled_time.tzinfo is None:
                    scheduled_time = scheduled_time.replace(tzinfo=timezone.utc)
                scheduled_post.scheduled_time = scheduled_time
            except Exception:
                raise HTTPException(status_code=400, detail="投稿日時の形式が正しくありません")
        
        if "bot_id" in post_data:
            bot = db.query(Bot).filter(Bot.id == post_data["bot_id"]).first()
            if not bot:
                raise HTTPException(status_code=404, detail="指定されたボットが見つかりません")
            scheduled_post.bot_id = post_data["bot_id"]
        
        if "image_urls" in post_data:
            try:
                scheduled_post.image_urls = json.dumps(post_data["image_urls"]) if post_data["image_urls"] else None
            except:
                pass
        
        if "status" in post_data:
            scheduled_post.status = post_data["status"]
        
        scheduled_post.updated_at = datetime.now(timezone.utc)
        
        db.commit()
        db.refresh(scheduled_post)
        
        # バックグラウンドタスクでスケジューラーを更新
        background_tasks.add_task(update_scheduler)
        
        return {
            "message": "予約投稿を更新しました",
            "post": scheduled_post.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"更新エラー: {str(e)}")


@router.delete("/scheduled-posts/{post_id}")
async def delete_scheduled_post(
    post_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """予約投稿を削除"""
    try:
        scheduled_post = db.query(ScheduledPost).filter(ScheduledPost.id == post_id).first()
        if not scheduled_post:
            raise HTTPException(status_code=404, detail="予約投稿が見つかりません")
        
        db.delete(scheduled_post)
        db.commit()
        
        # バックグラウンドタスクでスケジューラーを更新
        background_tasks.add_task(update_scheduler)
        
        return {"message": "予約投稿を削除しました"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"削除エラー: {str(e)}")


@router.post("/create-sample-sheet")
async def create_sample_sheet():
    """サンプルスプレッドシートを作成"""
    try:
        if not sheets_service.is_authenticated():
            raise HTTPException(
                status_code=400, 
                detail="Google Sheets API認証が必要です"
            )
        
        success = sheets_service.create_sample_sheets()
        
        if success:
            return {"message": "サンプルスプレッドシートを作成しました"}
        else:
            raise HTTPException(status_code=500, detail="サンプルシート作成に失敗しました")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"作成エラー: {str(e)}")


async def update_scheduler():
    """スケジューラーを更新（バックグラウンドタスク）"""
    # TODO: 実際のスケジューラー更新ロジックを実装
    # APSchedulerまたは類似のライブラリを使用して予約投稿を管理
    pass