#!/usr/bin/env python3
"""
WSL2環境用サーバー起動スクリプト
"""
import uvicorn
import os
import sys
import subprocess


def get_wsl_ip():
    """WSL2のIPアドレスを取得"""
    try:
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return "172.22.137.150"  # フォールバック


def main():
    """WSL2環境でサーバーを起動"""
    wsl_ip = get_wsl_ip()
    
    print("🐧 WSL2環境でPython Auto Tweeterを起動します")
    print("=" * 60)
    print(f"📍 WSL2 IP アドレス: {wsl_ip}")
    print("📁 作業ディレクトリ:", os.getcwd())
    
    # 環境変数の確認
    if os.path.exists('.env'):
        print("✅ .env ファイルが見つかりました")
    else:
        print("⚠️ .env ファイルが見つかりません")
    
    # データベースディレクトリの確認
    os.makedirs('data/logs', exist_ok=True)
    
    print("\n🌐 アクセス方法:")
    print("=" * 60)
    print("📌 WSL内からのアクセス:")
    print("   - http://localhost:8000")
    print("   - http://127.0.0.1:8000")
    print(f"   - http://{wsl_ip}:8000")
    print()
    print("📌 Windows側からのアクセス:")
    print("   ⚠️ 事前にWindows PowerShell（管理者権限）で以下を実行:")
    print(f"   netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress={wsl_ip}")
    print("   New-NetFirewallRule -DisplayName \"WSL2-Port8000\" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow")
    print()
    print("   その後、Windowsブラウザで:")
    print("   - http://localhost:8000")
    print("   - API ドキュメント: http://localhost:8000/docs")
    print()
    print("⏹️  停止するには Ctrl+C を押してください")
    print("=" * 60)
    
    # サーバー起動 - 0.0.0.0でバインドしてすべてのインターフェースで受信
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",  # 重要: すべてのインターフェースでリッスン
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()