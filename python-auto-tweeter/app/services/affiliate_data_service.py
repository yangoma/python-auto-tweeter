"""アフィリエイトデータ管理サービス"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class AffiliateDataFormat(Enum):
    """アフィリエイトデータフォーマット"""
    FANZA = "fanza"
    MANGA = "manga"
    GENERAL = "general"
    UNKNOWN = "unknown"


class AffiliateDataService:
    """アフィリエイトデータ統一管理サービス"""
    
    def __init__(self):
        # フォーマット判別のためのキーマッピング
        self.format_keys = {
            AffiliateDataFormat.FANZA: {
                "id_keys": ["Product ID", "Content ID"],
                "title_keys": ["Title"],
                "description_keys": ["Description"],
                "url_keys": ["URL", "Affiliate URL"],
                "image_keys": ["Sample Image URL"],
                "category_keys": ["Category", "Genre"]
            },
            AffiliateDataFormat.MANGA: {
                "id_keys": ["商品ID", "作品ID", "コンテンツID"],
                "title_keys": ["商品名", "作品名", "タイトル"],
                "description_keys": ["商品説明", "説明", "概要"],
                "url_keys": ["商品URL", "アフィリエイトURL", "リンク"],
                "image_keys": ["画像URL", "サンプル画像"],
                "category_keys": ["カテゴリ", "ジャンル", "分類"]
            },
            AffiliateDataFormat.GENERAL: {
                "id_keys": ["id", "product_id", "item_id"],
                "title_keys": ["name", "title", "product_name"],
                "description_keys": ["description", "summary", "details"],
                "url_keys": ["url", "link", "affiliate_url"],
                "image_keys": ["image", "thumbnail", "photo"],
                "category_keys": ["category", "type", "genre"]
            }
        }
    
    def detect_data_format(self, data: List[Dict[str, Any]]) -> AffiliateDataFormat:
        """
        データフォーマットを自動判別
        
        Args:
            data: アフィリエイトデータリスト
            
        Returns:
            AffiliateDataFormat: 判別されたフォーマット
        """
        if not data:
            return AffiliateDataFormat.UNKNOWN
        
        # 最初の数件のデータからキーを抽出
        sample_size = min(5, len(data))
        all_keys = set()
        
        for item in data[:sample_size]:
            all_keys.update(item.keys())
        
        logger.info(f"検出されたキー: {list(all_keys)}")
        
        # 各フォーマットとの一致度を計算
        format_scores = {}
        
        for format_type, format_keys in self.format_keys.items():
            if format_type == AffiliateDataFormat.UNKNOWN:
                continue
                
            score = 0
            total_key_types = len(format_keys)
            
            for key_type, expected_keys in format_keys.items():
                # 各キータイプで一致するキーがあるかチェック
                for expected_key in expected_keys:
                    if any(expected_key in key for key in all_keys):
                        score += 1
                        break
            
            # スコアを正規化（0-1の範囲）
            format_scores[format_type] = score / total_key_types
        
        logger.info(f"フォーマット判別スコア: {format_scores}")
        
        # 最も高いスコアのフォーマットを選択
        if not format_scores:
            return AffiliateDataFormat.UNKNOWN
        
        best_format = max(format_scores, key=format_scores.get)
        best_score = format_scores[best_format]
        
        # スコアが低い場合はUNKNOWNを返す
        if best_score < 0.3:
            logger.warning(f"フォーマット判別の信頼度が低いです（スコア: {best_score}）")
            return AffiliateDataFormat.UNKNOWN
        
        logger.info(f"判別されたフォーマット: {best_format.value} (スコア: {best_score})")
        return best_format
    
    def normalize_data(self, data: List[Dict[str, Any]], detected_format: AffiliateDataFormat = None) -> List[Dict[str, Any]]:
        """
        データを統一フォーマットに正規化
        
        Args:
            data: 元のアフィリエイトデータ
            detected_format: 検出されたフォーマット（Noneの場合は自動判別）
            
        Returns:
            List[Dict[str, Any]]: 正規化されたデータ
        """
        if not data:
            return []
        
        # フォーマットが指定されていない場合は自動判別
        if detected_format is None:
            detected_format = self.detect_data_format(data)
        
        if detected_format == AffiliateDataFormat.UNKNOWN:
            logger.warning("フォーマットが判別できませんでした。元のデータを返します。")
            return data
        
        logger.info(f"データを{detected_format.value}フォーマットとして正規化します")
        
        normalized_data = []
        format_keys = self.format_keys[detected_format]
        
        for item in data:
            normalized_item = self._normalize_single_item(item, format_keys)
            if normalized_item:
                normalized_data.append(normalized_item)
        
        logger.info(f"正規化完了: {len(normalized_data)}件のデータ")
        return normalized_data
    
    def _normalize_single_item(self, item: Dict[str, Any], format_keys: Dict[str, List[str]]) -> Optional[Dict[str, Any]]:
        """
        単一アイテムを正規化
        
        Args:
            item: 元のアイテムデータ
            format_keys: フォーマット用キーマッピング
            
        Returns:
            Optional[Dict[str, Any]]: 正規化されたアイテム（Noneの場合はスキップ）
        """
        normalized = {}
        
        # ID取得（複数のパターンを試す）
        item_id = self._find_value_by_keys(item, format_keys["id_keys"])
        
        # フォーマット特定のIDキーでも検索
        if not item_id:
            # よくあるIDキーパターンを試す
            common_id_keys = ["id", "ID", "商品ID", "product_id", "content_id", "item_id", "code", "商品コード"]
            item_id = self._find_value_by_keys(item, common_id_keys)
        
        # まだ見つからない場合は、値の形式からIDらしきものを探す
        if not item_id:
            for key, value in item.items():
                if value and isinstance(value, str):
                    # IDらしい形式（英数字、ハイフン、アンダースコア）をチェック
                    if len(str(value).strip()) >= 3 and all(c.isalnum() or c in ['-', '_'] for c in str(value).strip()):
                        item_id = str(value).strip()
                        logger.debug(f"IDらしき値を発見: キー='{key}', 値='{item_id}'")
                        break
        
        if not item_id:
            logger.debug(f"IDが見つからないためアイテムをスキップします: {list(item.keys())}")
            return None
        
        # 統一フォーマットにマッピング
        normalized.update({
            # 基本情報
            "id": item_id,
            "product_id": item_id,  # 後方互換性
            "content_id": item_id,  # 後方互換性
            "title": self._find_value_by_keys(item, format_keys["title_keys"], ""),
            "name": self._find_value_by_keys(item, format_keys["title_keys"], ""),  # 後方互換性
            "description": self._find_value_by_keys(item, format_keys["description_keys"], ""),
            "features": self._find_value_by_keys(item, format_keys["description_keys"], ""),  # 後方互換性
            
            # URL情報
            "url": self._find_value_by_keys(item, format_keys["url_keys"], ""),
            "affiliate_url": self._find_value_by_keys(item, format_keys["url_keys"], ""),
            "affiliate_link": self._find_value_by_keys(item, format_keys["url_keys"], ""),  # 後方互換性
            
            # カテゴリ情報
            "category": self._find_value_by_keys(item, format_keys["category_keys"], "商品"),
            
            # その他のデフォルト値
            "price": "価格情報なし",
            "target": "一般",
            "keywords": "",
            
            # 元データも保持
            "original_data": item
        })
        
        # 画像URLを収集
        image_urls = self._collect_image_urls(item, format_keys["image_keys"])
        normalized["image_urls"] = image_urls
        normalized["sample_images"] = image_urls  # 後方互換性
        normalized["sample_images_count"] = len(image_urls)
        
        # FANZAフォーマット互換性のため
        if "Product ID" in item:
            normalized["Product ID"] = item["Product ID"]
        if "Content ID" in item:
            normalized["Content ID"] = item["Content ID"]
        if "Title" in item:
            normalized["Title"] = item["Title"]
        if "Description" in item:
            normalized["Description"] = item["Description"]
        if "URL" in item:
            normalized["URL"] = item["URL"]
        if "Affiliate URL" in item:
            normalized["Affiliate URL"] = item["Affiliate URL"]
        
        return normalized
    
    def _find_value_by_keys(self, item: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
        """
        キーリストから最初に見つかった値を取得
        
        Args:
            item: データアイテム
            keys: 検索するキーのリスト
            default: デフォルト値
            
        Returns:
            Any: 見つかった値またはデフォルト値
        """
        for key in keys:
            # 完全一致
            if key in item and item[key]:
                return item[key]
            
            # 部分一致（キーが含まれている）
            for item_key in item.keys():
                if key in item_key and item[item_key]:
                    return item[item_key]
        
        return default
    
    def _collect_image_urls(self, item: Dict[str, Any], image_key_patterns: List[str]) -> List[str]:
        """
        画像URLを収集
        
        Args:
            item: データアイテム
            image_key_patterns: 画像キーのパターンリスト
            
        Returns:
            List[str]: 画像URLリスト
        """
        image_urls = []
        
        for key, value in item.items():
            if not value or not str(value).strip():
                continue
                
            # 画像関連のキーかチェック
            is_image_key = False
            for pattern in image_key_patterns:
                if pattern.lower() in key.lower():
                    is_image_key = True
                    break
            
            if is_image_key:
                url = str(value).strip()
                if url and url.startswith(('http://', 'https://')):
                    image_urls.append(url)
        
        return image_urls
    
    def get_product_by_id(self, data: List[Dict[str, Any]], product_id: str, detected_format: AffiliateDataFormat = None) -> Optional[Dict[str, Any]]:
        """
        商品IDで商品データを取得
        
        Args:
            data: アフィリエイトデータリスト
            product_id: 検索する商品ID
            detected_format: 検出されたフォーマット
            
        Returns:
            Optional[Dict[str, Any]]: 見つかった商品データ（正規化済み）
        """
        if not data:
            return None
        
        logger.info(f"商品ID '{product_id}' を検索中...")
        
        # まず元データで直接検索（正規化前）
        for item in data:
            # 全ての値をチェック（大文字小文字区別なし）
            for key, value in item.items():
                if value and str(value).lower() == product_id.lower():
                    logger.info(f"元データで商品ID発見: キー='{key}', 値='{value}'")
                    # 見つかったら正規化して返す
                    normalized_data = self.normalize_data([item], detected_format)
                    return normalized_data[0] if normalized_data else None
        
        # 正規化後のデータでも検索
        normalized_data = self.normalize_data(data, detected_format)
        logger.info(f"正規化後のデータ件数: {len(normalized_data)}")
        
        # IDで検索
        for i, item in enumerate(normalized_data):
            logger.debug(f"商品 {i+1}: id='{item.get('id')}', product_id='{item.get('product_id')}', content_id='{item.get('content_id')}'")
            
            if (item.get("id") == product_id or 
                item.get("product_id") == product_id or 
                item.get("content_id") == product_id):
                logger.info(f"正規化データで商品ID発見: {item.get('title', 'Unknown')}")
                return item
        
        # 見つからない場合、利用可能なIDをログ出力
        available_ids = []
        for item in normalized_data[:10]:  # 最初の10件
            item_id = item.get("id") or item.get("product_id") or item.get("content_id")
            if item_id:
                available_ids.append(item_id)
        
        logger.warning(f"商品ID '{product_id}' が見つかりませんでした。利用可能なID（最初の10件）: {available_ids}")
        
        return None


# シングルトンインスタンス
affiliate_data_service = AffiliateDataService()