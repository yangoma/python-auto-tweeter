"""Google Sheets API連携サービス"""

import os
import json
from typing import List, Dict, Any, Optional
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class SheetsService:
    """Google Sheets API連携サービス"""
    
    def __init__(self, credentials_path: Optional[str] = None, spreadsheet_id: Optional[str] = None):
        """
        初期化
        
        Args:
            credentials_path: Google API認証情報のJSONファイルパス
            spreadsheet_id: 操作対象のスプレッドシートID
        """
        # 保存された設定を読み込み
        saved_config = self._load_saved_config()
        
        self.credentials_path = (credentials_path or 
                               os.environ.get('GOOGLE_CREDENTIALS_PATH') or
                               saved_config.get('google_credentials_path'))
        self.spreadsheet_id = (spreadsheet_id or 
                             os.environ.get('GOOGLE_SPREADSHEET_ID') or
                             saved_config.get('spreadsheet_id'))
        self.service = None
        self.scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive.readonly'
        ]
        
        self._authenticate()
    
    def _load_saved_config(self) -> dict:
        """保存された設定を読み込み"""
        try:
            config_file = os.path.join(os.path.dirname(__file__), "..", "..", "app_config.json")
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"保存された設定の読み込みに失敗: {e}")
        return {}
    
    def _authenticate(self):
        """Google API認証"""
        try:
            if self.credentials_path and os.path.exists(self.credentials_path):
                # サービスアカウント認証
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path, scopes=self.scopes
                )
                self.service = build('sheets', 'v4', credentials=credentials)
                logger.info("Google Sheets API認証成功（サービスアカウント）")
            else:
                logger.warning("Google Sheets API認証情報が見つかりません")
                self.service = None
        except Exception as e:
            logger.error(f"Google Sheets API認証エラー: {e}")
            self.service = None
    
    def is_authenticated(self) -> bool:
        """認証状態を確認"""
        return self.service is not None
    
    def read_sheet(self, sheet_name: str, range_name: str = None) -> List[List[str]]:
        """
        シートからデータを読み取り
        
        Args:
            sheet_name: シート名
            range_name: 読み取り範囲（例: "A1:E10"）
        
        Returns:
            List[List[str]]: シートデータ
        """
        if not self.is_authenticated():
            raise Exception("Google Sheets API認証が必要です")
        
        try:
            if range_name:
                range_notation = f"{sheet_name}!{range_name}"
            else:
                range_notation = sheet_name
            
            logger.info(f"シート読み込み開始: {range_notation}, スプレッドシートID: {self.spreadsheet_id}")
            
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_notation
            ).execute()
            
            values = result.get('values', [])
            logger.info(f"シート '{sheet_name}' から {len(values)} 行のデータを読み込みました")
            if values:
                logger.debug(f"最初の数行のデータ: {values[:3]}")
            else:
                logger.warning(f"シート '{sheet_name}' にデータが見つかりません")
            
            return values
            
        except HttpError as e:
            logger.error(f"シート読み取りエラー: {e}")
            logger.error(f"エラー詳細: status={e.resp.status}, reason={e.resp.reason}")
            raise Exception(f"シート読み取りに失敗しました: {e}")
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            logger.error(f"エラータイプ: {type(e).__name__}")
            raise Exception(f"シート読み取りに失敗しました: {e}")
    
    def write_sheet(self, sheet_name: str, data: List[List[Any]], range_name: str = "A1") -> bool:
        """
        シートにデータを書き込み
        
        Args:
            sheet_name: シート名
            data: 書き込みデータ
            range_name: 書き込み開始位置
        
        Returns:
            bool: 成功/失敗
        """
        if not self.is_authenticated():
            raise Exception("Google Sheets API認証が必要です")
        
        try:
            range_notation = f"{sheet_name}!{range_name}"
            
            body = {
                'values': data
            }
            
            result = self.service.spreadsheets().values().update(
                spreadsheetId=self.spreadsheet_id,
                range=range_notation,
                valueInputOption='RAW',
                body=body
            ).execute()
            
            logger.info(f"{result.get('updatedCells')} セルを更新しました")
            return True
            
        except HttpError as e:
            logger.error(f"シート書き込みエラー: {e}")
            return False
    
    def append_rows(self, sheet_name: str, data: List[List[Any]]) -> bool:
        """
        シートに行を追加
        
        Args:
            sheet_name: シート名
            data: 追加するデータ
        
        Returns:
            bool: 成功/失敗
        """
        if not self.is_authenticated():
            raise Exception("Google Sheets API認証が必要です")
        
        try:
            body = {
                'values': data
            }
            
            result = self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=sheet_name,
                valueInputOption='RAW',
                body=body
            ).execute()
            
            logger.info(f"{len(data)} 行を追加しました")
            return True
            
        except HttpError as e:
            logger.error(f"行追加エラー: {e}")
            return False
    
    def get_post_schedule(self, sheet_name: str = "投稿スケジュール") -> List[Dict[str, Any]]:
        """
        投稿スケジュールを取得
        
        Args:
            sheet_name: シート名
        
        Returns:
            List[Dict]: 投稿スケジュールデータ
        """
        try:
            data = self.read_sheet(sheet_name)
            
            if not data or len(data) < 2:
                return []
            
            # ヘッダー行を取得
            headers = data[0]
            posts = []
            
            for row in data[1:]:
                if len(row) >= len(headers):
                    post_data = dict(zip(headers, row))
                    
                    # 空の行はスキップ
                    if not any(post_data.values()):
                        continue
                    
                    # 日時変換
                    if 'scheduled_time' in post_data or '投稿日時' in post_data:
                        time_key = 'scheduled_time' if 'scheduled_time' in post_data else '投稿日時'
                        try:
                            post_data['scheduled_time'] = datetime.fromisoformat(post_data[time_key])
                        except:
                            post_data['scheduled_time'] = None
                    
                    posts.append(post_data)
            
            return posts
            
        except Exception as e:
            logger.error(f"投稿スケジュール取得エラー: {e}")
            return []
    
    def update_post_status(self, sheet_name: str, row_index: int, status: str, result: str = "") -> bool:
        """
        投稿ステータスを更新
        
        Args:
            sheet_name: シート名
            row_index: 行インデックス（1-based）
            status: 新しいステータス
            result: 投稿結果
        
        Returns:
            bool: 成功/失敗
        """
        try:
            # ステータス列を更新（例：G列）
            status_range = f"{sheet_name}!G{row_index + 1}"
            self.write_sheet(sheet_name, [[status]], f"G{row_index + 1}")
            
            # 結果列を更新（例：H列）
            if result:
                result_range = f"{sheet_name}!H{row_index + 1}"
                self.write_sheet(sheet_name, [[result]], f"H{row_index + 1}")
            
            return True
            
        except Exception as e:
            logger.error(f"ステータス更新エラー: {e}")
            return False
    
    def get_affiliate_data(self, sheet_name: str = "アフィリ用データ") -> List[Dict[str, Any]]:
        """
        アフィリエイト用データを取得
        
        Args:
            sheet_name: シート名
        
        Returns:
            List[Dict]: アフィリエイトデータ
        """
        try:
            logger.info(f"アフィリエイトデータ取得開始: シート名='{sheet_name}'")
            data = self.read_sheet(sheet_name)
            
            if not data:
                logger.warning(f"シート '{sheet_name}' が空です")
                return []
            
            if len(data) < 2:
                logger.warning(f"シート '{sheet_name}' にヘッダー以外のデータがありません (行数: {len(data)})")
                return []
            
            headers = data[0]
            logger.info(f"ヘッダー: {headers}")
            
            affiliate_data = []
            
            for i, row in enumerate(data[1:], start=2):
                if len(row) >= len(headers):
                    # 行の長さをヘッダーに合わせる
                    row_data = row[:len(headers)]
                    item_data = dict(zip(headers, row_data))
                    
                    # 空の行はスキップ
                    if not any(str(v).strip() for v in item_data.values() if v):
                        logger.debug(f"行 {i} は空のためスキップします")
                        continue
                    
                    affiliate_data.append(item_data)
                    logger.debug(f"行 {i} のデータ: {item_data}")
                else:
                    logger.warning(f"行 {i} の列数が不足しています (期待: {len(headers)}, 実際: {len(row)})")
            
            logger.info(f"アフィリエイトデータ取得完了: {len(affiliate_data)} 件")
            return affiliate_data
            
        except Exception as e:
            logger.error(f"アフィリエイトデータ取得エラー: {e}")
            logger.error(f"エラータイプ: {type(e).__name__}")
            return []
    
    def create_sample_sheets(self) -> bool:
        """
        サンプルシートを作成
        
        Returns:
            bool: 成功/失敗
        """
        try:
            # 投稿スケジュールシートのサンプル
            schedule_headers = [
                "ID", "投稿内容", "画像URL", "投稿日時", "ボットID", "ステータス", "投稿結果", "作成日時"
            ]
            
            schedule_sample = [
                ["1", "おはようございます！今日も一日頑張りましょう！", "", "2024-01-01 09:00:00", "bot_001", "pending", "", "2023-12-31 10:00:00"],
                ["2", "ランチタイムです。美味しいご飯で午後も頑張りましょう！", "", "2024-01-01 12:00:00", "bot_001", "pending", "", "2023-12-31 10:00:00"]
            ]
            
            # アフィリエイトデータシートのサンプル（新フォーマット）
            affiliate_headers = [
                "Content ID", "Product ID", "Title", "URL", "Affiliate URL", "Date", 
                "Description", "Created At", "Sample Image URL 1", "Sample Image URL 2", 
                "Sample Image URL 3", "Sample Image URL 4", "Sample Image URL 5", 
                "Sample Image URL 6", "Sample Image URL 7", "Sample Image URL 8", 
                "Sample Image URL 9", "Sample Image URL 10", "Sample Image URL 11", 
                "Sample Image URL 12", "Sample Image URL 13", "Sample Image URL 14", 
                "Sample Image URL 15", "Sample Image URL 16", "Sample Image URL 17", 
                "Sample Image URL 18", "Sample Image URL 19", "Sample Image URL 20"
            ]
            
            affiliate_sample = [
                [
                    "CONT001", "PROD001", "プログラミング学習本", 
                    "https://example.com/products/book1", "https://example.com/affiliate/book1",
                    "2024-01-15", "初心者向けのPython学習書籍。実践的なサンプルコード付き。",
                    "2024-01-01 10:00:00",
                    "https://example.com/images/book1_cover.jpg",
                    "https://example.com/images/book1_sample1.jpg",
                    "https://example.com/images/book1_sample2.jpg",
                    "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""
                ],
                [
                    "CONT002", "PROD002", "ワイヤレスイヤホン", 
                    "https://example.com/products/earphone1", "https://example.com/affiliate/earphone1",
                    "2024-01-20", "高音質ノイズキャンセリング機能付きワイヤレスイヤホン。",
                    "2024-01-01 10:00:00",
                    "https://example.com/images/earphone1_main.jpg",
                    "https://example.com/images/earphone1_side.jpg",
                    "https://example.com/images/earphone1_case.jpg",
                    "https://example.com/images/earphone1_wearing.jpg",
                    "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""
                ]
            ]
            
            # シートに書き込み
            self.write_sheet("投稿スケジュール", [schedule_headers] + schedule_sample)
            self.write_sheet("アフィリ用データ", [affiliate_headers] + affiliate_sample)
            
            logger.info("サンプルシートを作成しました")
            return True
            
        except Exception as e:
            logger.error(f"サンプルシート作成エラー: {e}")
            return False


# シングルトンインスタンス
sheets_service = SheetsService()