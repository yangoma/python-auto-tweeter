#!/usr/bin/env python3
"""
サーバー起動スクリプト
"""
import uvicorn
import os
import sys


def main():
    """サーバーを起動"""
    print("🚀 Python Auto Tweeter を起動しています...")
    print("📁 作業ディレクトリ:", os.getcwd())
    print("🐍 Python パス:", sys.executable)
    
    # 環境変数の確認
    if os.path.exists('.env'):
        print("✅ .env ファイルが見つかりました")
    else:
        print("⚠️ .env ファイルが見つかりません")
    
    # データベースディレクトリの確認
    if not os.path.exists('data'):
        os.makedirs('data', exist_ok=True)
        print("📁 data ディレクトリを作成しました")
    
    if not os.path.exists('data/logs'):
        os.makedirs('data/logs', exist_ok=True)
        print("📁 data/logs ディレクトリを作成しました")
    
    print("🌐 サーバーを起動しています...")
    print("📍 アクセス URL:")
    print("   - http://localhost:8000")
    print("   - http://127.0.0.1:8000")
    print("   - API ドキュメント: http://localhost:8000/docs")
    print("⏹️  停止するには Ctrl+C を押してください")
    print("-" * 50)
    
    # サーバー起動
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()