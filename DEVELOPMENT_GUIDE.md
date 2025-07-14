# 開発継続ガイド

## 📋 現在の実装状況

### ✅ 完了した機能

#### 1. アフィリエイトデータ処理システム
- **自動フォーマット判別**: FANZA、マンガ、汎用フォーマットの自動識別
- **データ正規化**: 異なるフォーマットを統一されたAPIで処理
- **柔軟な商品検索**: 大文字小文字区別なし、部分一致対応
- **後方互換性**: 既存のFANZAデータ処理を維持

#### 2. 分析機能（完全実装）
- **投稿パフォーマンス分析**: エンゲージメント率、成功率分析
- **投稿時間分析**: 最適投稿時間の算出
- **コンテンツ分析**: ハッシュタグ効果、メディア効果、文章長分析
- **トレンド分析**: 投稿推移、ハッシュタグトレンド
- **AI推奨事項**: データに基づく自動改善提案
- **ボット比較**: 複数ボット間のパフォーマンス比較
- **データエクスポート**: JSON/CSV形式でのデータ出力

#### 3. LLM機能拡張
- **自動スケジュール投稿**: アフィリエイト商品から自動投稿生成
- **バッチ投稿生成**: 複数商品の並列処理
- **プロバイダー管理**: 設定済みプロバイダーの管理機能
- **デバッグ機能**: 詳細なデバッグエンドポイント

## 🏗️ アーキテクチャ

### ディレクトリ構成
```
python-auto-tweeter/
├── app/
│   ├── api/
│   │   ├── analytics.py          # 分析API
│   │   ├── llm.py               # LLM API（拡張済み）
│   │   ├── bots.py              # ボット管理
│   │   ├── sheets.py            # スプレッドシート連携
│   │   └── twitter.py           # Twitter API
│   ├── models/
│   │   ├── analytics.py         # 分析データモデル
│   │   ├── post.py              # 投稿モデル
│   │   ├── scheduled_post.py    # 予約投稿モデル
│   │   └── bot.py               # ボットモデル
│   ├── services/
│   │   ├── analytics_service.py      # 分析ロジック
│   │   ├── affiliate_data_service.py # データ統一処理
│   │   ├── llm_service.py           # LLM統合サービス
│   │   └── sheets_service.py        # スプレッドシート処理
│   └── templates/
│       ├── analytics.html       # 分析ダッシュボード
│       ├── dashboard.html       # メインダッシュボード
│       └── ...
```

### データフロー

#### 分析データフロー
```
投稿データ → PostMetrics → 分析処理 → 推奨事項生成 → UI表示
     ↓            ↓              ↓
BotAnalytics → トレンド分析 → エクスポート
     ↓
ContentAnalysis
```

#### アフィリエイトデータ処理フロー
```
スプレッドシート → フォーマット判別 → データ正規化 → LLM生成 → 投稿
      ↓               ↓              ↓
   生データ    →   統一フォーマット  →  推奨事項
```

## 🔧 技術スタック

### バックエンド
- **FastAPI**: REST API フレームワーク
- **SQLAlchemy**: ORM（データベース操作）
- **Alembic**: データベースマイグレーション
- **Pydantic**: データバリデーション

### フロントエンド
- **Alpine.js**: リアクティブなUI
- **Chart.js**: グラフ表示
- **Tailwind CSS**: スタイリング

### 外部サービス連携
- **Google Sheets API**: データソース
- **Twitter API**: 投稿機能
- **各種LLM API**: Ollama, OpenAI, Claude, Gemini

## 🚀 次に開発すべき機能

### 優先度: 高

#### 1. リアルタイムメトリクス収集
```python
# 実装場所: app/services/metrics_collector.py
class MetricsCollector:
    async def collect_post_metrics(self, post_id: str):
        """投稿のリアルタイムメトリクスを収集"""
        # Twitter API v2でメトリクス取得
        # PostMetricsテーブルに保存
```

#### 2. 自動分析レポート
```python
# 実装場所: app/services/report_generator.py
class ReportGenerator:
    async def generate_weekly_report(self, bot_id: str):
        """週次分析レポートを自動生成"""
        # 分析データをまとめてレポート生成
        # メール通知やSlack連携
```

#### 3. A/Bテスト機能
```python
# 実装場所: app/models/ab_test.py
class ABTest(Base):
    """A/Bテスト管理モデル"""
    test_name = Column(String)
    variant_a_config = Column(JSON)
    variant_b_config = Column(JSON)
    success_metric = Column(String)
```

### 優先度: 中

#### 4. 予測分析
- エンゲージメント予測モデル
- 最適投稿時間の動的調整
- バイラル投稿の早期検出

#### 5. 高度なコンテンツ分析
- 感情分析（sentiment analysis）
- トピック分析（topic modeling）
- 画像認識による効果測定

#### 6. ワークフロー自動化
- 投稿パフォーマンスに基づく自動調整
- 失敗投稿の自動リトライ
- スケジュール最適化

### 優先度: 低

#### 7. 高度な可視化
- インタラクティブな地理的分析
- リアルタイムダッシュボード
- カスタムダッシュボード作成機能

## 🔍 開発時の注意点

### データベース

#### マイグレーション実行
```bash
# 新しいマイグレーション作成
alembic revision --autogenerate -m "分析テーブル追加"

# マイグレーション実行
alembic upgrade head
```

#### 分析データの初期化
```python
# app/scripts/init_analytics.py
async def initialize_analytics_data():
    """既存の投稿データから分析データを生成"""
    # 過去の投稿データを分析テーブルに移行
```

### API開発

#### 新しい分析エンドポイント追加
```python
# app/api/analytics.py に追加
@router.get("/custom-analysis")
async def custom_analysis(params: CustomAnalysisRequest):
    # カスタム分析ロジック
    analytics_service = get_analytics_service(db)
    return analytics_service.custom_analysis(params)
```

### フロントエンド

#### 新しいチャート追加
```javascript
// templates/analytics.html
function updateCustomChart() {
    const ctx = document.getElementById('customChart');
    new Chart(ctx, {
        type: 'customType',
        data: customData,
        options: customOptions
    });
}
```

## 🧪 テスト戦略

### 分析機能のテスト
```python
# tests/test_analytics.py
def test_posting_time_analysis():
    # テストデータ作成
    # 分析実行
    # 結果検証

def test_content_performance():
    # コンテンツ分析のテスト
```

### データ処理のテスト
```python
# tests/test_affiliate_data.py
def test_format_detection():
    # フォーマット判別のテスト

def test_data_normalization():
    # データ正規化のテスト
```

## 📊 監視とメトリクス

### 重要な監視項目
- 分析処理のレスポンス時間
- データ正規化の成功率
- LLM API の使用量と成功率
- データベースのパフォーマンス

### ログ監視
```python
# 重要なログポイント
logger.info("分析開始", extra={"bot_id": bot_id, "period": days})
logger.error("データ処理失敗", extra={"error": str(e), "data_format": format})
```

## 🔐 セキュリティ考慮事項

### API セキュリティ
- 分析データの機密性
- 大量データクエリの制限
- レート制限の実装

### データプライバシー
- 個人情報の除外
- データ匿名化
- 保存期間の制限

## 📈 パフォーマンス最適化

### データベース最適化
```sql
-- 重要なインデックス
CREATE INDEX idx_posts_posted_at_bot_id ON posts(posted_at, bot_id);
CREATE INDEX idx_post_metrics_post_id ON post_metrics(post_id);
CREATE INDEX idx_bot_analytics_date_bot_id ON bot_analytics(date, bot_id);
```

### キャッシュ戦略
```python
# Redis を使用した分析結果キャッシュ
@cache.memoize(timeout=3600)  # 1時間キャッシュ
def get_overview_statistics(bot_id, days):
    # 重い分析処理
```

## 🔄 継続的改善

### 定期タスク
1. **週次**: 分析データの整合性チェック
2. **月次**: パフォーマンス最適化レビュー
3. **四半期**: 新機能要件の整理

### 機能拡張の優先順位決定
1. ユーザーフィードバック
2. データ分析結果
3. 技術的負債の解消
4. 競合分析

---

## 🤝 開発チーム向け情報

### 開発環境セットアップ
```bash
# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 依存関係インストール
pip install -r requirements.txt

# データベース初期化
alembic upgrade head

# 開発サーバー起動
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Git ワークフロー
```bash
# 新機能開発
git checkout -b feature/new-analytics-feature
git commit -m "feat: 新しい分析機能を実装"

# プルリクエスト前のチェック
python -m pytest tests/
python -m black app/
python -m isort app/
```

### コード品質
- **Black**: コードフォーマッター
- **isort**: import文の整理
- **pytest**: テストフレームワーク
- **mypy**: 型チェック（推奨）

---

*このガイドは開発の継続性を保つために作成されました。新しい機能追加時は、このドキュメントも合わせて更新してください。*