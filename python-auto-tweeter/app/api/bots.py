from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import uuid
from ..core.database import get_db
from ..models import User, Bot, TwitterAccount
from ..schemas import BotResponse, BotCreate, BotUpdate
from .dependencies import get_current_user

router = APIRouter()


@router.get("/")
async def get_bots(
    db: Session = Depends(get_db)
):
    """ボット一覧を取得（認証なし版）"""
    try:
        bots = db.query(Bot).all()
        
        # レスポンス形式を統一
        bots_data = []
        for bot in bots:
            # Twitter アカウント情報を取得
            twitter_account = None
            if bot.twitter_account_id:
                twitter_account = db.query(TwitterAccount).filter(
                    TwitterAccount.id == bot.twitter_account_id
                ).first()
            
            bots_data.append({
                "id": bot.id,
                "name": bot.name,
                "description": bot.description,
                "status": "active" if getattr(bot, 'is_active', False) else "inactive",
                "twitter_account": {
                    "id": twitter_account.id if twitter_account else None,
                    "username": twitter_account.username if twitter_account else None,
                    "display_name": twitter_account.display_name if twitter_account else None
                } if twitter_account else None,
                "created_at": bot.created_at.isoformat() if hasattr(bot, 'created_at') and bot.created_at else None
            })
        
        return bots_data
        
    except Exception as e:
        print(f"Error getting bots: {e}")
        return []


@router.post("/")
async def create_bot(
    bot_data: dict,
    db: Session = Depends(get_db)
):
    """ボットを作成（認証なし版）"""
    try:
        # Twitter アカウントの存在チェック
        twitter_account = db.query(TwitterAccount).filter(
            TwitterAccount.id == bot_data.get("twitter_account_id")
        ).first()
        
        if not twitter_account:
            raise HTTPException(
                status_code=404,
                detail="Twitter account not found"
            )
        
        # ボット作成
        bot = Bot(
            id=str(uuid.uuid4()),
            name=bot_data.get("name"),
            description=bot_data.get("description", ""),
            twitter_account_id=bot_data.get("twitter_account_id"),
            is_active=bot_data.get("is_active", False),
            posting_interval=str(bot_data.get("posting_interval", 60)),
            max_posts_per_day=str(bot_data.get("max_posts_per_day", 10)),
            content_template=bot_data.get("content_template", ""),
            created_at=__import__("datetime").datetime.now()
        )
        
        db.add(bot)
        db.commit()
        db.refresh(bot)
        
        return {
            "message": "Bot created successfully",
            "bot": {
                "id": bot.id,
                "name": bot.name,
                "description": bot.description,
                "status": "active" if bot.is_active else "inactive"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Bot creation failed: {str(e)}")


@router.put("/{bot_id}")
async def update_bot(
    bot_id: str,
    bot_data: dict,
    db: Session = Depends(get_db)
):
    """ボットを更新（認証なし版）"""
    try:
        bot = db.query(Bot).filter(Bot.id == bot_id).first()
        
        if not bot:
            raise HTTPException(status_code=404, detail="Bot not found")
        
        # Twitter アカウントの存在チェック（変更される場合）
        if bot_data.get("twitter_account_id"):
            twitter_account = db.query(TwitterAccount).filter(
                TwitterAccount.id == bot_data.get("twitter_account_id")
            ).first()
            
            if not twitter_account:
                raise HTTPException(
                    status_code=404,
                    detail="Twitter account not found"
                )
        
        # フィールドを更新
        if "name" in bot_data:
            bot.name = bot_data["name"]
        if "description" in bot_data:
            bot.description = bot_data["description"]
        if "twitter_account_id" in bot_data:
            bot.twitter_account_id = bot_data["twitter_account_id"]
        if "posting_interval" in bot_data:
            bot.posting_interval = str(bot_data["posting_interval"])
        if "max_posts_per_day" in bot_data:
            bot.max_posts_per_day = str(bot_data["max_posts_per_day"])
        if "content_template" in bot_data:
            bot.content_template = bot_data["content_template"]
        if "is_active" in bot_data:
            bot.is_active = bot_data["is_active"]
        
        bot.updated_at = __import__("datetime").datetime.now()
        
        db.commit()
        db.refresh(bot)
        
        return {
            "message": "Bot updated successfully",
            "bot": {
                "id": bot.id,
                "name": bot.name,
                "description": bot.description,
                "status": "active" if bot.is_active else "inactive"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Bot update failed: {str(e)}")


@router.delete("/{bot_id}")
async def delete_bot(
    bot_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """ボットを削除"""
    bot = db.query(Bot).join(Bot.twitter_account).filter(
        Bot.id == bot_id,
        Bot.twitter_account.has(user_id=current_user.id)
    ).first()
    
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    db.delete(bot)
    db.commit()
    return {"message": "Bot deleted successfully"}


@router.post("/{bot_id}/start")
async def start_bot(
    bot_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """ボットを開始"""
    bot = db.query(Bot).join(Bot.twitter_account).filter(
        Bot.id == bot_id,
        Bot.twitter_account.has(user_id=current_user.id)
    ).first()
    
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot.is_active = True
    db.commit()
    return {"message": "Bot started successfully"}


@router.post("/{bot_id}/stop")
async def stop_bot(
    bot_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """ボットを停止"""
    bot = db.query(Bot).join(Bot.twitter_account).filter(
        Bot.id == bot_id,
        Bot.twitter_account.has(user_id=current_user.id)
    ).first()
    
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot.is_active = False
    db.commit()
    return {"message": "Bot stopped successfully"}


@router.post("/{bot_id}/toggle")
async def toggle_bot(
    bot_id: str,
    db: Session = Depends(get_db)
):
    """ボットのON/OFFを切り替え（認証なし版）"""
    bot = db.query(Bot).filter(Bot.id == bot_id).first()
    
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    # is_activeフィールドを切り替え
    bot.is_active = not bot.is_active
    
    db.commit()
    db.refresh(bot)
    return {
        "message": "Bot toggled successfully", 
        "status": "active" if bot.is_active else "inactive"
    }


@router.post("/{bot_id}/post")
async def manual_post(
    bot_id: str,
    post_data: dict,
    db: Session = Depends(get_db)
):
    """ボットで手動投稿を実行（ツリー投稿対応）"""
    try:
        # ボット取得
        bot = db.query(Bot).filter(Bot.id == bot_id).first()
        if not bot:
            raise HTTPException(status_code=404, detail="Bot not found")
        
        # Twitterアカウント取得
        twitter_account = db.query(TwitterAccount).filter(
            TwitterAccount.id == bot.twitter_account_id
        ).first()
        if not twitter_account:
            raise HTTPException(status_code=404, detail="Twitter account not found")
        
        # 投稿内容（ツリー投稿の場合は配列）
        contents = post_data.get("contents", [post_data.get("content", bot.content_template or "Hello from Bot!")])
        if isinstance(contents, str):
            contents = [contents]
        
        # 画像情報を取得
        images = post_data.get("images", [])
        
        # ツリー投稿を実行
        return await post_thread(bot, twitter_account, contents, db, images)
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Post failed: {str(e)}")


async def post_thread(bot, twitter_account, contents, db, images=[]):
    """ツリー投稿を実行"""
    try:
        import tweepy
        
        # 保存されたAPI認証情報を使用
        if not twitter_account.api_key or not twitter_account.api_secret:
            raise Exception("API KeyまたはAPI Secretが設定されていません。アカウント編集から設定してください。")
        
        # Twitter API v2 クライアント（投稿用）
        client = tweepy.Client(
            consumer_key=twitter_account.api_key,
            consumer_secret=twitter_account.api_secret,
            access_token=twitter_account.access_token,
            access_token_secret=twitter_account.access_token_secret,
            wait_on_rate_limit=True
        )
        
        # Twitter API v1.1クライアント（メディアアップロード用）
        auth = tweepy.OAuth1UserHandler(
            twitter_account.api_key,
            twitter_account.api_secret,
            twitter_account.access_token,
            twitter_account.access_token_secret
        )
        api_v1 = tweepy.API(auth)
        
        # 画像をTwitterにアップロード
        media_ids = []
        if images:
            import os
            uploads_dir = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
            
            for image in images[:4]:  # 最大4枚まで
                try:
                    # ローカルファイルパスを構築
                    image_path = os.path.join(uploads_dir, "images", image.get("filename", ""))
                    
                    if os.path.exists(image_path):
                        # Twitter API v1.1を使用してメディアをアップロード
                        media = api_v1.media_upload(image_path)
                        media_ids.append(media.media_id)
                    else:
                        print(f"Image file not found: {image_path}")
                except Exception as media_error:
                    print(f"Failed to upload image {image.get('filename', 'unknown')}: {media_error}")
                    continue
        
        # ツリー投稿結果を保存
        posted_tweets = []
        previous_tweet_id = None
        thread_root_id = None
        
        # 各投稿を順番に処理
        for i, content in enumerate(contents):
            try:
                # 最初の投稿か、前の投稿への返信かを決定
                tweet_params = {"text": content}
                if previous_tweet_id:
                    tweet_params["in_reply_to_tweet_id"] = previous_tweet_id
                
                # 最初の投稿の場合のみ画像を添付
                if i == 0 and media_ids:
                    tweet_params["media_ids"] = media_ids
                
                # ツイート投稿
                response = client.create_tweet(**tweet_params)
                tweet_id = str(response.data['id'])
                
                # 最初の投稿をルートとして設定
                if i == 0:
                    thread_root_id = tweet_id
                
                # 投稿履歴をデータベースに保存
                from ..models.post import Post
                import json
                
                # 画像情報をJSONで保存（最初の投稿のみ）
                image_urls_json = None
                media_ids_json = None
                if i == 0 and images:
                    image_urls_json = json.dumps([img.get("url") for img in images])
                    media_ids_json = json.dumps(media_ids)
                
                post_record = Post(
                    id=str(uuid.uuid4()),
                    bot_id=bot.id,
                    content=content,
                    twitter_post_id=tweet_id,
                    parent_post_id=posted_tweets[-1]['db_id'] if posted_tweets else None,
                    thread_root_id=thread_root_id if i > 0 else None,
                    thread_order=i,
                    in_reply_to_tweet_id=previous_tweet_id,
                    image_urls=image_urls_json,
                    media_ids=media_ids_json,
                    status="posted",
                    posted_at=__import__("datetime").datetime.now()
                )
                
                db.add(post_record)
                db.commit()
                db.refresh(post_record)
                
                # 結果を保存
                posted_tweets.append({
                    "db_id": post_record.id,
                    "tweet_id": tweet_id,
                    "content": content,
                    "order": i,
                    "url": f"https://twitter.com/{twitter_account.username}/status/{tweet_id}"
                })
                
                # 次の投稿のための準備
                previous_tweet_id = tweet_id
                
            except Exception as tweet_error:
                # 個別のツイート投稿に失敗した場合
                print(f"Tweet {i+1} failed: {tweet_error}")
                
                # 失敗した投稿も記録
                from ..models.post import Post
                post_record = Post(
                    id=str(uuid.uuid4()),
                    bot_id=bot.id,
                    content=content,
                    parent_post_id=posted_tweets[-1]['db_id'] if posted_tweets else None,
                    thread_root_id=thread_root_id if i > 0 else None,
                    thread_order=i,
                    in_reply_to_tweet_id=previous_tweet_id,
                    status="failed",
                    error_message=str(tweet_error),
                    posted_at=__import__("datetime").datetime.now()
                )
                
                db.add(post_record)
                db.commit()
                
                # 失敗した投稿も記録（ただしtweet_idはNone）
                posted_tweets.append({
                    "db_id": post_record.id,
                    "tweet_id": None,
                    "content": content,
                    "order": i,
                    "error": str(tweet_error),
                    "status": "failed"
                })
                
                # 失敗した場合は後続の投稿をスキップ
                break
        
        # 結果を返す
        successful_posts = [t for t in posted_tweets if t.get("tweet_id")]
        failed_posts = [t for t in posted_tweets if not t.get("tweet_id")]
        
        if successful_posts:
            return {
                "message": f"ツリー投稿に成功しました！({len(successful_posts)}/{len(contents)}件)",
                "thread_id": thread_root_id,
                "posts": posted_tweets,
                "successful_count": len(successful_posts),
                "total_count": len(contents),
                "thread_url": f"https://twitter.com/{twitter_account.username}/status/{thread_root_id}" if thread_root_id else None
            }
        else:
            return {
                "message": "ツリー投稿に失敗しました",
                "posts": posted_tweets,
                "successful_count": 0,
                "total_count": len(contents),
                "failed": True
            }
            
    except Exception as api_error:
        # APIエラーの場合は代替処理（シミュレーション）
        print(f"Twitter API Error: {api_error}")
        
        # シミュレーション投稿を記録
        from ..models.post import Post
        simulated_posts = []
        
        for i, content in enumerate(contents):
            # API認証情報が不足している場合は"simulated"、その他のエラーは"failed"
            status = "simulated" if "API Key" in str(api_error) or "認証情報" in str(api_error) else "failed"
            
            # 画像情報をJSONで保存（最初の投稿のみ）
            import json
            image_urls_json = None
            if i == 0 and images:
                image_urls_json = json.dumps([img.get("url") for img in images])
            
            post_record = Post(
                id=str(uuid.uuid4()),
                bot_id=bot.id,
                content=content,
                parent_post_id=simulated_posts[-1].id if simulated_posts else None,
                thread_order=i,
                image_urls=image_urls_json,
                status=status,
                error_message=str(api_error),
                posted_at=__import__("datetime").datetime.now()
            )
            
            db.add(post_record)
            simulated_posts.append(post_record)
        
        db.commit()
        
        if status == "simulated":
            return {
                "message": f"ツリー投稿をシミュレートしました ({len(contents)}件)",
                "posts": [{"content": p.content, "order": p.thread_order} for p in simulated_posts],
                "note": "実際の投稿には正しいTwitter API設定が必要です。Twitterアカウント編集画面でAPI KeyとSecretを設定してください。",
                "simulated": True,
                "total_count": len(contents)
            }
        else:
            return {
                "message": "ツリー投稿に失敗しました",
                "posts": [{"content": p.content, "order": p.thread_order, "error": str(api_error)} for p in simulated_posts],
                "error": str(api_error),
                "failed": True,
                "total_count": len(contents)
            }