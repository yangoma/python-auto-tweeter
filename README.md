# Python Auto Tweeter

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/fastapi-latest-green.svg)

個人利用を前提としたPython版SNS自動投稿ツール

## 概要

Python Auto Tweeterは、個人でのTwitter自動投稿を効率化するためのPythonアプリケーションです。軽量でローカル実行が可能で、データの完全なコントロールが可能です。

## 主な機能

### 🤖 AI & LLM統合
- **マルチLLM対応**: Ollama, OpenAI GPT, Claude, Gemini
- **自動投稿生成**: アフィリエイトデータからAI投稿生成
- **バッチ処理**: 複数商品の並列投稿生成
- **スケジュール自動生成**: AI による投稿スケジュールの最適化

### 📊 高度な分析機能
- **投稿パフォーマンス分析**: エンゲージメント率、成功率の詳細分析
- **投稿時間最適化**: 時間帯別パフォーマンス分析と最適時間提案
- **コンテンツ効果分析**: ハッシュタグ、メディア、文章長の効果測定
- **トレンド分析**: ハッシュタグトレンドと投稿推移の追跡
- **AI推奨事項**: データに基づく自動改善提案
- **ボット比較**: 複数ボット間のパフォーマンス比較

### 🔄 アフィリエイト連携
- **マルチフォーマット対応**: FANZA, マンガ, 汎用データの自動判別
- **データ正規化**: 異なるソースのデータを統一処理
- **柔軟検索**: 大文字小文字区別なし、部分一致対応

### 🐦 Twitter 機能
- **OAuth認証**: セキュアなAPI認証
- **自動投稿・スケジューリング**: 予約投稿機能
- **画像投稿**: 複数画像の同時投稿
- **スレッド投稿**: 連続投稿の管理

### 🖥️ ユーザーインターフェース
- **インタラクティブダッシュボード**: Chart.js による可視化
- **レスポンシブデザイン**: Tailwind CSS + Alpine.js
- **リアルタイム更新**: 分析データの自動更新
- **Google Sheets連携**: データ管理の効率化

## 技術スタック

### バックエンド
- **FastAPI**: 高性能なRESTful API
- **SQLAlchemy**: ORM（Object-Relational Mapping）
- **SQLite**: 軽量データベース
- **APScheduler**: バックグラウンドタスク管理
- **Alembic**: データベースマイグレーション
- **Pydantic**: データ検証とAPI仕様

### AI & LLM
- **Ollama**: ローカルLLM実行環境
- **OpenAI API**: GPT-4, GPT-3.5-turbo
- **Anthropic Claude**: Claude-3 Sonnet/Haiku
- **Google Gemini**: Gemini-1.5-Flash/Pro

### フロントエンド
- **Alpine.js**: リアクティブなJavaScriptフレームワーク
- **Chart.js**: インタラクティブなグラフライブラリ
- **Tailwind CSS**: ユーティリティファーストCSSフレームワーク
- **HTML5/CSS3**: モダンなWebUI

### 外部連携
- **Twitter API v2**: 投稿・メトリクス取得
- **Google Sheets API**: データソース連携
- **Google Drive API**: ファイル管理

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

### 🏠 コア機能
- `GET /` - メインダッシュボード
- `GET /analytics` - 分析ダッシュボード
- `GET /docs` - API ドキュメント（OpenAPI）
- `GET /health` - ヘルスチェック

### 🤖 ボット管理
- `GET /api/bots` - ボット一覧取得
- `POST /api/bots` - ボット作成
- `PUT /api/bots/{bot_id}` - ボット更新
- `DELETE /api/bots/{bot_id}` - ボット削除

### 🐦 投稿管理
- `GET /api/posts` - 投稿履歴取得
- `POST /api/twitter/tweet` - ツイート投稿
- `POST /api/scheduled-posts` - 予約投稿作成
- `GET /api/scheduled-posts` - 予約投稿一覧

### 🧠 AI & LLM
- `GET /api/llm/providers` - 利用可能プロバイダー
- `POST /api/llm/generate` - 汎用コンテンツ生成
- `POST /api/llm/generate-social-post` - SNS投稿生成
- `POST /api/llm/generate-from-affiliate-data/{product_id}` - アフィリエイト投稿生成
- `GET /api/llm/affiliate-products` - 商品一覧取得
- `POST /api/llm/auto-schedule-posts` - 自動スケジュール生成
- `POST /api/llm/batch-generate-posts` - バッチ投稿生成

### 📊 分析機能
- `GET /api/analytics/overview` - 分析概要
- `GET /api/analytics/posting-time` - 投稿時間分析
- `GET /api/analytics/content-performance` - コンテンツ分析
- `GET /api/analytics/trends` - トレンド分析
- `GET /api/analytics/recommendations/{bot_id}` - AI推奨事項
- `GET /api/analytics/dashboard/{bot_id}` - ボットダッシュボード
- `GET /api/analytics/comparison` - ボット比較分析
- `GET /api/analytics/export/{bot_id}` - データエクスポート

### 📈 Google Sheets連携
- `GET /api/sheets/status` - 認証状態確認
- `POST /api/sheets/authenticate` - Google認証
- `GET /api/sheets/data` - スプレッドシートデータ取得

### 🖼️ その他
- `POST /api/upload/image` - 画像アップロード

詳細なAPI仕様は http://localhost:8000/docs で確認できます。

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 貢献

プルリクエストや Issues は歓迎です。大きな変更を行う場合は、まず Issue を作成して変更内容について議論してください。

## サポート

問題や質問がある場合は、GitHub Issues を作成してください。

## 免責事項

このツールは個人利用を前提としています。Twitter APIの利用規約を遵守し、適切に使用してください。スパムや不適切な投稿は禁止されています。