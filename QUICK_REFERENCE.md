# 🚀 クイックリファレンス

開発の継続に必要な基本的な情報をまとめた簡潔なガイドです。

## 📁 重要なファイル

### 🆕 新規追加されたファイル
```
app/api/analytics.py              # 分析API - 全分析機能のエンドポイント
app/models/analytics.py           # 分析データモデル - 5つの分析テーブル
app/services/analytics_service.py # 分析ロジック - パフォーマンス計算
app/services/affiliate_data_service.py # データ統一処理 - フォーマット判別
templates/analytics.html          # 分析UI - インタラクティブダッシュボード
```

### 🔄 更新されたファイル
```
app/api/llm.py                   # アフィリエイト処理改善・デバッグ機能追加
app/main.py                      # 分析APIルーター追加
app/models/__init__.py           # 分析モデルのインポート追加
```

## 🗃️ データベーステーブル

### 分析用テーブル（新規）
```sql
post_metrics         -- 投稿メトリクス（いいね・RT・返信数）
bot_analytics        -- ボット分析データ（期間別統計）
content_analysis     -- コンテンツ分析（ハッシュタグ・長さ効果）
trend_analysis       -- トレンド分析（ハッシュタグトレンド）
recommendation_log   -- 推奨事項ログ
```

### 既存テーブル
```sql
posts               -- 投稿データ（統計情報含む）
scheduled_posts     -- 予約投稿（AI生成フラグ追加済み）
bots               -- ボット設定
```

## 🎯 主要機能の使い方

### 1. 分析ダッシュボード
```
URL: http://localhost:8000/analytics
機能: ボット別・期間別の詳細分析
特徴: リアルタイム更新、グラフ表示、推奨事項
```

### 2. アフィリエイト投稿生成
```python
# API呼び出し例
POST /api/llm/generate-from-affiliate-data/{product_id}
{
  "provider": "ollama",
  "post_type": "promotional",
  "tone": "friendly"
}
```

### 3. 分析データ取得
```python
# 概要統計
GET /api/analytics/overview?bot_id={id}&days=30

# 投稿時間分析
GET /api/analytics/posting-time?bot_id={id}&days=30

# AI推奨事項
GET /api/analytics/recommendations/{bot_id}
```

## 🔧 開発環境

### サーバー起動
```bash
# 開発サーバー
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# アクセス
http://localhost:8000        # メインダッシュボード
http://localhost:8000/analytics  # 分析ダッシュボード
http://localhost:8000/docs  # API仕様
```

### データベース操作
```bash
# マイグレーション作成
alembic revision --autogenerate -m "説明"

# マイグレーション実行
alembic upgrade head

# データベース確認
sqlite3 data/database.db
.tables  # テーブル一覧
.schema post_metrics  # テーブル構造確認
```

## 🐛 トラブルシューティング

### よくある問題と解決法

#### 1. アフィリエイト商品が見つからない
```python
# デバッグエンドポイント使用
GET /api/llm/debug/search-product/{product_id}?sheet_name=シート名

# データフォーマット確認
GET /api/llm/debug/affiliate-data-format/{sheet_name}
```

#### 2. 分析データが表示されない
```python
# 分析機能ヘルスチェック
GET /api/analytics/health

# 投稿データの存在確認
GET /api/posts

# データベース直接確認
sqlite3 data/database.db
SELECT COUNT(*) FROM posts WHERE status = 'posted';
```

#### 3. LLM生成が失敗する
```python
# プロバイダー状況確認
GET /api/llm/providers

# 接続テスト
POST /api/llm/test-connection/{provider}

# 設定確認
GET /api/llm/config/{provider}
```

## 🚀 次の開発ステップ

### 優先度 高
1. **メトリクス自動収集**: 投稿後のエンゲージメントデータ取得
2. **分析データ蓄積**: 既存投稿データからの分析データ生成
3. **推奨事項適用**: AI推奨事項の自動適用機能

### 優先度 中
1. **A/Bテスト**: 投稿バリエーションの効果測定
2. **予測分析**: エンゲージメント予測モデル
3. **高度な可視化**: より詳細なグラフと分析

## 📝 開発ノート

### コードパターン

#### 新しい分析機能追加
```python
# 1. analytics_service.py に分析ロジック追加
def new_analysis_function(self, params):
    # 分析ロジック
    return results

# 2. analytics.py にAPIエンドポイント追加
@router.get("/new-analysis")
async def new_analysis(params, db: Session = Depends(get_db)):
    analytics_service = get_analytics_service(db)
    return analytics_service.new_analysis_function(params)

# 3. analytics.html にフロントエンド追加
// JavaScriptで新しいチャートやテーブル追加
```

#### 新しいアフィリエイトフォーマット追加
```python
# affiliate_data_service.py の format_keys に追加
self.format_keys[AffiliateDataFormat.NEW_FORMAT] = {
    "id_keys": ["新しいIDキー"],
    "title_keys": ["新しいタイトルキー"],
    # ...
}
```

### 重要な設定ファイル
```
llm_config.json      # LLM設定（API キー等）
app_config.json      # Google Sheets設定
.env                 # 環境変数
alembic.ini          # データベース設定
```

### ログ確認
```bash
# アプリケーションログ
tail -f logs/app.log

# 特定機能のログ
grep "analytics" logs/app.log
grep "affiliate" logs/app.log
```

---

💡 **開発TIP**: 新機能開発時は必ずヘルスチェックエンドポイントを実装し、デバッグ機能を充実させることで、問題の早期発見と解決が可能になります。