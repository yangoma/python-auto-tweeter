#!/usr/bin/env python3
"""
簡単なテストスクリプト - サーバーが正常に動作しているかを確認
"""
import httpx
import asyncio


async def test_server():
    """サーバーの動作をテスト"""
    import subprocess
    
    # WSL2のIPアドレスを取得
    try:
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
        wsl_ip = result.stdout.strip()
    except:
        wsl_ip = "127.0.0.1"
    
    print(f"🔍 WSL2 IP: {wsl_ip}")
    
    # 複数のURLでテスト
    test_urls = [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        f"http://{wsl_ip}:8000"
    ]
    
    for base_url in test_urls:
        print(f"\n🧪 テスト URL: {base_url}")
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                # ルートエンドポイントテスト
                response = await client.get(f"{base_url}/")
                print(f"   GET / : {response.status_code}")
                print(f"   Response: {response.json()}")
                
                # ヘルスチェック
                response = await client.get(f"{base_url}/health")
                print(f"   GET /health : {response.status_code}")
                print(f"   Response: {response.json()}")
                
                print(f"   ✅ {base_url} で正常に動作しています！")
                break  # 成功したらループを抜ける
                
            except Exception as e:
                print(f"   ❌ {base_url} に接続できません: {e}")
    
    print(f"\n📌 WSL2環境でのアクセス方法:")
    print(f"   - WSL内: http://localhost:8000")
    print(f"   - Windows: 事前設定後 http://localhost:8000")
    print(f"   - 直接IP: http://{wsl_ip}:8000")


if __name__ == "__main__":
    asyncio.run(test_server())