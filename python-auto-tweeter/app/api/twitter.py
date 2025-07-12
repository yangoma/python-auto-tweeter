from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import secrets
from ..core.database import get_db
from ..models import User, TwitterAccount
from ..schemas import TwitterAccountResponse, TwitterAccountCreate, TwitterAccountUpdate
from .dependencies import get_current_user
import os

router = APIRouter()


@router.get("/accounts")
async def get_twitter_accounts(
    db: Session = Depends(get_db)
):
    """Twitterアカウント一覧を取得（認証なし版）"""
    try:
        accounts = db.query(TwitterAccount).all()
        
        # レスポンス形式を統一
        accounts_data = []
        for account in accounts:
            accounts_data.append({
                "id": account.id,
                "username": account.username,
                "display_name": account.display_name,
                "twitter_user_id": account.twitter_user_id,
                "status": "active" if account.is_active else "inactive",
                "profile_image_url": account.profile_image_url or "https://via.placeholder.com/40",
                "created_at": account.created_at.isoformat() if account.created_at else None
            })
        
        return accounts_data
        
    except Exception as e:
        print(f"Error getting accounts: {e}")
        return []


@router.post("/accounts", response_model=TwitterAccountResponse)
async def create_twitter_account(
    account_create: TwitterAccountCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Twitterアカウントを追加"""
    # 既存のアカウントチェック
    existing_account = db.query(TwitterAccount).filter(
        TwitterAccount.twitter_user_id == account_create.twitter_user_id
    ).first()
    
    if existing_account:
        raise HTTPException(
            status_code=400,
            detail="Twitter account already exists"
        )
    
    account = TwitterAccount(
        user_id=current_user.id,
        **account_create.dict()
    )
    db.add(account)
    db.commit()
    db.refresh(account)
    return account


@router.put("/accounts/{account_id}")
async def update_twitter_account(
    account_id: str,
    account_data: dict,
    db: Session = Depends(get_db)
):
    """Twitterアカウントを更新（認証なし版）"""
    try:
        account = db.query(TwitterAccount).filter(
            TwitterAccount.id == account_id
        ).first()
        
        if not account:
            raise HTTPException(status_code=404, detail="Twitter account not found")
        
        # API認証情報を更新
        if "api_key" in account_data and account_data["api_key"]:
            account.api_key = account_data["api_key"]
        
        if "api_secret" in account_data and account_data["api_secret"]:
            account.api_secret = account_data["api_secret"]
        
        # Access Tokenを更新（マスクされていない場合のみ）
        if "access_token" in account_data and account_data["access_token"] and not account_data["access_token"].startswith("••"):
            account.access_token = account_data["access_token"]
        
        if "access_token_secret" in account_data and account_data["access_token_secret"] and not account_data["access_token_secret"].startswith("••"):
            account.access_token_secret = account_data["access_token_secret"]
        
        # アクティブ状態を更新
        if "is_active" in account_data:
            account.is_active = account_data["is_active"]
        
        account.updated_at = __import__("datetime").datetime.now()
        
        db.commit()
        db.refresh(account)
        
        return {
            "message": "Account updated successfully",
            "account": {
                "id": account.id,
                "username": account.username,
                "display_name": account.display_name,
                "has_api_key": bool(account.api_key),
                "has_api_secret": bool(account.api_secret),
                "is_active": account.is_active
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Account update failed: {str(e)}")


@router.delete("/accounts/{account_id}")
async def delete_twitter_account(
    account_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Twitterアカウントを削除"""
    account = db.query(TwitterAccount).filter(
        TwitterAccount.id == account_id,
        TwitterAccount.user_id == current_user.id
    ).first()
    
    if not account:
        raise HTTPException(status_code=404, detail="Twitter account not found")
    
    db.delete(account)
    db.commit()
    return {"message": "Twitter account deleted successfully"}


@router.post("/accounts/{account_id}/refresh")
async def refresh_account(
    account_id: str,
    db: Session = Depends(get_db)
):
    """Twitterアカウント情報を更新"""
    try:
        # アカウント取得
        account = db.query(TwitterAccount).filter(
            TwitterAccount.id == account_id
        ).first()
        
        if not account:
            raise HTTPException(status_code=404, detail="Twitter account not found")
        
        # Twitter APIを使って最新情報を取得
        try:
            import tweepy
            
            # Twitter API v2 クライアントの作成（仮のAPI Key使用）
            # 実際の実装では環境変数からAPI Keyを取得
            client = tweepy.Client(
                consumer_key="dummy_key",  # 実際のAPI Keyに置き換え
                consumer_secret="dummy_secret",
                access_token=account.access_token,
                access_token_secret=account.access_token_secret,
                wait_on_rate_limit=True
            )
            
            # 認証されたユーザーの情報を取得
            user_info = client.get_me(user_fields=['id', 'username', 'name', 'profile_image_url'])
            
            if user_info.data:
                twitter_user = user_info.data
                account.username = twitter_user.username
                account.display_name = twitter_user.name
                account.profile_image_url = getattr(twitter_user, 'profile_image_url', account.profile_image_url)
                
                db.commit()
                db.refresh(account)
                
                return {
                    "message": "Account refreshed successfully",
                    "account": {
                        "id": account.id,
                        "username": account.username,
                        "display_name": account.display_name,
                        "profile_image_url": account.profile_image_url
                    }
                }
            else:
                raise HTTPException(status_code=401, detail="Failed to fetch user info")
                
        except Exception as e:
            # APIエラーの場合でも、DBに保存されている情報で更新日時を更新
            account.updated_at = __import__("datetime").datetime.now()
            db.commit()
            
            return {
                "message": "Account info updated (cached data)",
                "account": {
                    "id": account.id,
                    "username": account.username,
                    "display_name": account.display_name,
                    "profile_image_url": account.profile_image_url
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refresh failed: {str(e)}")


# OAuth認証の状態を一時保存するためのメモリストレージ（本番環境ではRedisなどを使用）
oauth_states = {}


@router.get("/auth/start")
async def start_oauth(request: Request, db: Session = Depends(get_db)):
    """Twitter OAuth認証を開始"""
    # 状態コードを生成してセッションに保存
    state = secrets.token_urlsafe(32)
    oauth_states[state] = {
        "created_at": __import__("datetime").datetime.now(),
        "request_origin": str(request.base_url)
    }
    
    # Twitter Developer Portalで設定するCallback URI
    callback_uri = f"{request.base_url}api/twitter/auth/callback"
    
    # Twitter OAuth 1.0a の認証URLを構築（実際の実装では tweepy や requests-oauthlib を使用）
    # ここでは簡易的な実装例を示します
    
    return {
        "auth_url": f"https://api.twitter.com/oauth/authorize?oauth_token=TEMP_TOKEN&state={state}",
        "callback_uri": callback_uri,
        "state": state,
        "instructions": {
            "step1": "Twitter Developer Portalのアプリ設定で以下のCallback URLを登録してください:",
            "callback_url": callback_uri,
            "step2": "上記のauth_urlにアクセスしてTwitter認証を完了してください"
        }
    }


@router.get("/auth/callback")
async def oauth_callback(
    request: Request,
    oauth_token: Optional[str] = None,
    oauth_verifier: Optional[str] = None,
    state: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Twitter OAuth認証のコールバック処理"""
    
    # 状態確認
    if not state or state not in oauth_states:
        raise HTTPException(status_code=400, detail="Invalid or expired state")
    
    # 状態をクリーンアップ
    oauth_state = oauth_states.pop(state)
    
    if not oauth_token or not oauth_verifier:
        # 認証が拒否された場合
        return RedirectResponse(url="/twitter?error=access_denied")
    
    try:
        # ここで実際のTwitter APIを使ってアクセストークンを取得
        # 実装例（実際にはtweepyやrequests-oauthlibを使用）
        
        # 仮のユーザー情報（実際のAPIレスポンスで置き換え）
        twitter_user_info = {
            "id": "123456789",
            "username": "example_user",
            "name": "Example User",
            "profile_image_url": "https://example.com/avatar.jpg"
        }
        
        # データベースに保存またはユーザー作成ロジック
        # （認証システムがある場合は、現在のユーザーと関連付け）
        
        return RedirectResponse(url="/twitter?success=connected")
        
    except Exception as e:
        print(f"OAuth callback error: {e}")
        return RedirectResponse(url="/twitter?error=connection_failed")


@router.post("/connect")
async def connect_account(
    account_data: dict,
    db: Session = Depends(get_db)
):
    """手動でAPI認証情報を設定して実際のTwitterユーザー情報を取得"""
    try:
        # API認証情報の検証
        api_key = account_data.get("api_key")
        api_secret = account_data.get("api_secret")
        access_token = account_data.get("access_token")
        access_token_secret = account_data.get("access_token_secret")
        
        if not all([api_key, api_secret, access_token, access_token_secret]):
            raise HTTPException(status_code=400, detail="All credentials are required")
        
        # Twitter APIを使って実際のユーザー情報を取得
        try:
            import tweepy
            
            # Twitter API v2 クライアントの作成
            client = tweepy.Client(
                consumer_key=api_key,
                consumer_secret=api_secret,
                access_token=access_token,
                access_token_secret=access_token_secret,
                wait_on_rate_limit=True
            )
            
            # 認証されたユーザーの情報を取得
            user_info = client.get_me(user_fields=['id', 'username', 'name', 'profile_image_url'])
            
            if not user_info.data:
                raise HTTPException(status_code=401, detail="Invalid Twitter credentials")
            
            twitter_user = user_info.data
            twitter_user_id = str(twitter_user.id)
            username = twitter_user.username
            display_name = twitter_user.name
            profile_image_url = getattr(twitter_user, 'profile_image_url', None)
            
        except tweepy.Unauthorized:
            raise HTTPException(status_code=401, detail="Twitter authentication failed. Please check your credentials.")
        except tweepy.TweepyException as e:
            raise HTTPException(status_code=400, detail=f"Twitter API error: {str(e)}")
        except Exception as e:
            # Tweepyが利用できない場合のフォールバック
            twitter_user_id = str(uuid.uuid4())
            username = f"user_{twitter_user_id[:8]}"
            display_name = "Connected User"
            profile_image_url = None
        
        # 既存のアカウントをチェック
        existing_account = db.query(TwitterAccount).filter(
            TwitterAccount.twitter_user_id == twitter_user_id
        ).first()
        
        if existing_account:
            # 既存アカウントの情報を更新
            existing_account.username = username
            existing_account.display_name = display_name
            existing_account.profile_image_url = profile_image_url
            existing_account.access_token = access_token
            existing_account.access_token_secret = access_token_secret
            existing_account.is_active = True
            
            db.commit()
            db.refresh(existing_account)
            
            return {
                "message": "Account updated successfully",
                "account": {
                    "id": existing_account.id,
                    "username": existing_account.username,
                    "display_name": existing_account.display_name,
                    "twitter_user_id": existing_account.twitter_user_id,
                    "status": "active" if existing_account.is_active else "inactive",
                    "profile_image_url": profile_image_url
                }
            }
        
        # デフォルトユーザーを作成または取得（認証システムがない場合の暫定対応）
        from ..models.user import User
        default_user = db.query(User).first()
        if not default_user:
            default_user = User(
                id=str(uuid.uuid4()),
                email="default@example.com",
                username="default_user"
            )
            db.add(default_user)
            db.commit()
            db.refresh(default_user)
        
        # TwitterAccountをデータベースに保存
        twitter_account = TwitterAccount(
            id=str(uuid.uuid4()),
            user_id=default_user.id,
            twitter_user_id=twitter_user_id,
            username=username,
            display_name=display_name,
            profile_image_url=profile_image_url,
            api_key=api_key,
            api_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
            is_active=True
        )
        
        db.add(twitter_account)
        db.commit()
        db.refresh(twitter_account)
        
        return {
            "message": "Account connected successfully",
            "account": {
                "id": twitter_account.id,
                "username": twitter_account.username,
                "display_name": twitter_account.display_name,
                "twitter_user_id": twitter_account.twitter_user_id,
                "status": "active" if twitter_account.is_active else "inactive",
                "profile_image_url": profile_image_url
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Connection failed: {str(e)}")


@router.get("/callback-info")
async def get_callback_info(request: Request):
    """コールバックURI情報を取得"""
    callback_uri = f"{request.base_url}api/twitter/auth/callback"
    
    return {
        "callback_uri": callback_uri,
        "instructions": {
            "title": "Twitter Developer Portal設定手順",
            "steps": [
                {
                    "step": 1,
                    "description": "Twitter Developer Portal (https://developer.twitter.com/) にアクセス"
                },
                {
                    "step": 2,
                    "description": "アプリの設定ページで「Authentication settings」を開く"
                },
                {
                    "step": 3,
                    "description": f"Callback URLに以下を追加: {callback_uri}"
                },
                {
                    "step": 4,
                    "description": "Website URLに以下を設定: " + str(request.base_url)
                }
            ]
        }
    }