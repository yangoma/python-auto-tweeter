from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..models import User


def get_current_user(db: Session = Depends(get_db)) -> User:
    """現在のユーザーを取得（個人利用のため簡易実装）"""
    # 個人利用版なので、最初のユーザーを返す
    user = db.query(User).first()
    if not user:
        # ユーザーが存在しない場合は作成
        user = User(
            email="user@example.com",
            name="Personal User",
            plan_type="personal"
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return user