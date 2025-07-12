# Python Auto Tweeter

個人利用を前提としたPython版SNS自動投稿ツール

## 概要

このツールは、個人でのTwitter自動投稿を効率化するためのPythonアプリケーションです。
軽量でローカル実行が可能で、データの完全なコントロールが可能です。

## 機能

- Twitter OAuth認証
- 自動投稿・スケジューリング
- アフィリエイトリンク変換
- 投稿分析・メトリクス収集
- Web UI (Streamlit)
- CLI インターフェース

## 技術スタック

- FastAPI: RESTful API
- SQLAlchemy: ORM
- SQLite: データベース
- APScheduler: バックグラウンドタスク
- Streamlit: Web UI
- Click: CLI

## セットアップ

1. Python 3.9以上をインストール
2. 依存関係をインストール:
   ```bash
   pip install -r requirements.txt
   ```
3. 環境変数を設定:
   ```bash
   cp .env.example .env
   # .envファイルを編集してTwitter APIキーを設定
   ```
4. データベースを初期化:
   ```bash
   alembic upgrade head
   ```

## 使用方法

### API サーバー起動

#### 方法1: 起動スクリプトを使用（推奨）
```bash
python start_server.py
```

#### 方法2: uvicornコマンドを直接使用
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 方法3: Pythonモジュールとして実行
```bash
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

### Web UI 起動
```bash
streamlit run ui/main.py
```

### CLI 使用
```bash
python -m app.cli --help
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