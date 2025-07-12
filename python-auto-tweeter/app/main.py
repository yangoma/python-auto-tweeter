from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import uuid
import shutil
from pathlib import Path
from .core.config import settings
from .core.database import engine, Base
from .utils.logger import logger
from .api import users, twitter, bots, sheets, llm
from .services.scheduler_service import scheduler_service
import os

# データベーステーブルの作成
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションライフサイクル管理"""
    # 起動時の処理
    logger.info(f"{settings.app_name} starting up...")
    
    # 予約投稿スケジューラーを開始
    try:
        scheduler_service.start()
        logger.info("予約投稿スケジューラーを開始しました")
    except Exception as e:
        logger.error(f"スケジューラー開始エラー: {e}")
    
    yield
    
    # 終了時の処理
    logger.info(f"{settings.app_name} shutting down...")
    
    # 予約投稿スケジューラーを停止
    try:
        scheduler_service.stop()
        logger.info("予約投稿スケジューラーを停止しました")
    except Exception as e:
        logger.error(f"スケジューラー停止エラー: {e}")

# FastAPIアプリケーションの作成
app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan
)

# 静的ファイルとテンプレートの設定
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")

# ディレクトリが存在しない場合は作成
os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
# 画像アップロード用のディレクトリをマウント
uploads_dir = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(uploads_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")
templates = Jinja2Templates(directory=templates_dir)

# CORS の設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """ダッシュボードページ"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": settings.app_name
    })


@app.get("/twitter", response_class=HTMLResponse)
async def twitter_page(request: Request):
    """Twitter連携ページ"""
    return templates.TemplateResponse("twitter.html", {
        "request": request,
        "title": f"Twitter連携 - {settings.app_name}"
    })


@app.get("/bots", response_class=HTMLResponse)
async def bots_page(request: Request):
    """ボット管理ページ"""
    return templates.TemplateResponse("bots.html", {
        "request": request,
        "title": f"ボット管理 - {settings.app_name}"
    })


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    """分析ページ"""
    return templates.TemplateResponse("analytics.html", {
        "request": request,
        "title": f"分析 - {settings.app_name}"
    })


@app.get("/posts", response_class=HTMLResponse)
async def posts_page(request: Request):
    """投稿履歴ページ"""
    return templates.TemplateResponse("posts.html", {
        "request": request,
        "title": f"投稿履歴 - {settings.app_name}"
    })


@app.get("/sheets", response_class=HTMLResponse)
async def sheets_page(request: Request):
    """スプレッドシート連携ページ"""
    return templates.TemplateResponse("sheets.html", {
        "request": request,
        "title": f"スプレッドシート連携 - {settings.app_name}"
    })


@app.get("/health")
async def health():
    """ヘルスチェック"""
    return {"status": "healthy"}


@app.post("/api/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """画像をアップロード"""
    try:
        # ファイル形式のチェック
        allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail="サポートされていないファイル形式です。JPG、PNG、GIF、WebPのみ対応しています。"
            )
        
        # ファイルサイズチェック (5MB以下)
        contents = await file.read()
        if len(contents) > 5 * 1024 * 1024:  # 5MB
            raise HTTPException(
                status_code=400,
                detail="ファイルサイズが大きすぎます。5MB以下にしてください。"
            )
        
        # ユニークなファイル名を生成
        file_id = str(uuid.uuid4())
        filename = f"{file_id}{file_extension}"
        file_path = os.path.join(uploads_dir, "images", filename)
        
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # ファイルを保存
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # 相対URLを返す
        file_url = f"/uploads/images/{filename}"
        
        return {
            "message": "画像をアップロードしました",
            "file_id": file_id,
            "filename": filename,
            "url": file_url,
            "size": len(contents)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image upload error: {e}")
        raise HTTPException(status_code=500, detail="画像のアップロードに失敗しました")


# 統計データエンドポイント
@app.get("/api/stats")
async def get_stats():
    """ダッシュボード用の統計データを取得"""
    from sqlalchemy.orm import Session
    from .core.database import SessionLocal
    from .models.bot import Bot
    from .models.post import Post
    from .models.twitter_account import TwitterAccount
    
    db = SessionLocal()
    try:
        # アクティブなボット数
        active_bots = db.query(Bot).filter(Bot.status == "active").count()
        
        # 総ボット数
        total_bots = db.query(Bot).count()
        
        # 今日の投稿数（仮実装）
        today_posts = 0
        
        # 総投稿数
        total_posts = db.query(Post).count() if hasattr(Post, '__table__') else 0
        
        # 連携アカウント数
        linked_accounts = db.query(TwitterAccount).count()
        
        return {
            "active_bots": active_bots,
            "total_bots": total_bots,
            "today_posts": today_posts,
            "total_posts": total_posts,
            "linked_accounts": linked_accounts
        }
    finally:
        db.close()

# 投稿履歴API
@app.get("/api/posts")
async def get_posts():
    """投稿履歴を取得"""
    from sqlalchemy.orm import Session
    from .core.database import SessionLocal
    from .models.post import Post
    from .models.bot import Bot
    from .models.twitter_account import TwitterAccount
    
    db = SessionLocal()
    try:
        # 投稿一覧を取得（関連データも含む）
        posts = db.query(Post).order_by(Post.created_at.desc()).all()
        
        posts_data = []
        for post in posts:
            # ボット情報を取得
            bot = db.query(Bot).filter(Bot.id == post.bot_id).first()
            bot_name = bot.name if bot else "Unknown Bot"
            
            # Twitterアカウント情報を取得
            account_id = None
            if bot and bot.twitter_account_id:
                account = db.query(TwitterAccount).filter(
                    TwitterAccount.id == bot.twitter_account_id
                ).first()
                account_id = account.id if account else None
            
            # 画像情報を処理
            image_urls = []
            if post.image_urls:
                try:
                    import json
                    image_urls = json.loads(post.image_urls)
                except:
                    image_urls = []
            
            posts_data.append({
                "id": post.id,
                "bot_id": post.bot_id,
                "bot_name": bot_name,
                "account_id": account_id,
                "content": post.content,
                "status": post.status,
                "twitter_post_id": post.twitter_post_id,
                "error_message": post.error_message,
                "parent_post_id": post.parent_post_id,
                "thread_root_id": post.thread_root_id,
                "thread_order": post.thread_order,
                "in_reply_to_tweet_id": post.in_reply_to_tweet_id,
                "image_urls": image_urls,
                "posted_at": post.posted_at.isoformat() if post.posted_at else None,
                "created_at": post.created_at.isoformat() if post.created_at else None
            })
        
        return posts_data
        
    except Exception as e:
        print(f"Error getting posts: {e}")
        return []
    finally:
        db.close()

# APIルーターの登録
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(twitter.router, prefix="/api/twitter", tags=["twitter"])
app.include_router(bots.router, prefix="/api/bots", tags=["bots"])
app.include_router(sheets.router, prefix="/api/sheets", tags=["sheets"])
app.include_router(llm.router, prefix="/api/llm", tags=["llm"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )