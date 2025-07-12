# Python Auto Tweeter

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/fastapi-latest-green.svg)

個人利用を前提としたPython版SNS自動投稿ツール

## 概要

Python Auto Tweeterは、個人でのTwitter自動投稿を効率化するためのPythonアプリケーションです。軽量でローカル実行が可能で、データの完全なコントロールが可能です。

## 主な機能

- 🐦 **Twitter OAuth認証**: セキュアなAPI認証
- ⏰ **自動投稿・スケジューリング**: 予約投稿機能
- 🔗 **アフィリエイトリンク変換**: リンクの自動変換
- 📊 **投稿分析・メトリクス収集**: 投稿データの分析
- 🖥️ **Web UI**: 直感的なWebインターフェース
- 📱 **レスポンシブデザイン**: モバイル対応
- 🖼️ **画像アップロード**: 投稿に画像を添付
- 🧵 **スレッド投稿**: 連続投稿の管理
- 📈 **Google Sheets連携**: データ管理の効率化

## 技術スタック

### バックエンド
- **FastAPI**: 高性能なRESTful API
- **SQLAlchemy**: ORM（Object-Relational Mapping）
- **SQLite**: 軽量データベース
- **APScheduler**: バックグラウンドタスク管理
- **Alembic**: データベースマイグレーション

### フロントエンド
- **HTML5/CSS3/JavaScript**: レスポンシブWebUI
- **Bootstrap**: UIフレームワーク

### その他
- **Click**: CLI インターフェース
- **Pydantic**: データ検証
- **python-multipart**: ファイルアップロード

## インストール

### 必要要件
- Python 3.9以上
- Twitter Developer Account（API v2アクセス）

### セットアップ

1. **リポジトリのクローン**
   ```bash
   git clone https://github.com/your-username/python-auto-tweeter.git
   cd python-auto-tweeter
   ```

2. **依存関係のインストール**
   ```bash
   pip install -r requirements.txt
   ```

3. **環境変数の設定**
   ```bash
   cp .env.example .env
   # .envファイルを編集してTwitter APIキーを設定
   ```

4. **データベースの初期化**
   ```bash
   alembic upgrade head
   ```

## 使用方法

### API サーバー起動

#### 推奨方法: 起動スクリプトを使用
```bash
python start_server.py
```

#### その他の起動方法
```bash
# uvicornコマンドを直接使用
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Pythonモジュールとして実行
python -m app.main
```

### アクセス方法
- **メインページ**: http://localhost:8000
- **API ドキュメント**: http://localhost:8000/docs
- **ヘルスチェック**: http://localhost:8000/health

### WSL環境での注意事項
WSL2を使用している場合、Windowsからlocalhostにアクセスできない場合があります。
その場合は以下を試してください：

1. WSLのIPアドレスを確認:
   ```bash
   ip addr show eth0
   ```

2. Windows PowerShellで（管理者権限）:
   ```powershell
   netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=[WSL-IP]
   ```

### 動作確認
```bash
python test_server.py
```

## 開発

### 開発環境セットアップ
```bash
pip install -r requirements-dev.txt
```

### テスト実行
```bash
pytest
```

### コードフォーマット
```bash
black .
ruff check .
```

### データベースマイグレーション
```bash
# 新しいマイグレーションファイルを作成
alembic revision --autogenerate -m "description"

# マイグレーションを適用
alembic upgrade head
```

## プロジェクト構造

```
python-auto-tweeter/
├── app/
│   ├── api/                    # API エンドポイント
│   ├── core/                   # 設定とデータベース
│   ├── models/                 # データベースモデル
│   ├── schemas/                # Pydanticスキーマ
│   ├── services/               # ビジネスロジック
│   └── utils/                  # ユーティリティ
├── alembic/                    # データベースマイグレーション
├── templates/                  # HTMLテンプレート
├── static/                     # 静的ファイル
├── uploads/                    # アップロードファイル
├── data/                       # データファイル
├── credentials/                # 認証情報
└── tests/                      # テストファイル
```

## 設定

### 環境変数

```bash
# Twitter API設定
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret

# データベース設定
DATABASE_URL=sqlite:///./data/database.db

# アプリケーション設定
DEBUG=true
SECRET_KEY=your_secret_key
```

## API エンドポイント

### 主要なエンドポイント

- `GET /` - ダッシュボード
- `GET /docs` - API ドキュメント
- `POST /api/twitter/tweet` - ツイート投稿
- `GET /api/bots` - ボット一覧
- `POST /api/bots` - ボット作成
- `GET /api/posts` - 投稿履歴
- `POST /api/upload/image` - 画像アップロード

詳細なAPI仕様は `/docs` で確認できます。

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 貢献

プルリクエストや Issues は歓迎です。大きな変更を行う場合は、まず Issue を作成して変更内容について議論してください。

## サポート

問題や質問がある場合は、GitHub Issues を作成してください。

## 免責事項

このツールは個人利用を前提としています。Twitter APIの利用規約を遵守し、適切に使用してください。スパムや不適切な投稿は禁止されています。