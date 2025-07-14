"""LLM API エンドポイント"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import asyncio
import logging
from ..services.llm_service import llm_service, LLMProvider

logger = logging.getLogger(__name__)

router = APIRouter()


class GenerateContentRequest(BaseModel):
    """コンテンツ生成リクエスト"""
    prompt: str
    provider: Optional[str] = None
    model: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.7
    system_prompt: Optional[str] = None


class GenerateSocialPostRequest(BaseModel):
    """ソーシャル投稿生成リクエスト"""
    product_data: Optional[Dict[str, Any]] = None
    post_type: str = "promotional"  # promotional, informational, engaging
    target_audience: str = "general"
    tone: str = "friendly"  # friendly, professional, casual, exciting
    include_hashtags: bool = True
    max_length: int = 280
    provider: Optional[str] = None


class GenerateThreadRequest(BaseModel):
    """スレッド投稿生成リクエスト"""
    topic: str
    num_posts: int = 3
    max_length_per_post: int = 280
    provider: Optional[str] = None


class LLMConfigRequest(BaseModel):
    """LLM設定リクエスト"""
    provider: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None


class AutoScheduleRequest(BaseModel):
    """自動スケジュール生成リクエスト"""
    sheet_name: str = "FANZAデータ"
    provider: str = "ollama"
    model: Optional[str] = None
    bot_id: str
    post_count: int = 10
    start_date: str  # YYYY-MM-DD format
    interval_hours: int = 24
    post_type: str = "promotional"
    tone: str = "casual"
    max_length: int = 280
    include_hashtags: bool = True
    target_audience: str = "general"


class BatchGenerateRequest(BaseModel):
    """バッチ投稿生成リクエスト"""
    sheet_name: str = "FANZAデータ"
    provider: str = "ollama"
    model: Optional[str] = None
    product_ids: List[str]  # 特定の商品IDリスト
    post_type: str = "promotional"
    tone: str = "casual"
    max_length: int = 280
    include_hashtags: bool = True
    target_audience: str = "general"
    save_as_drafts: bool = True  # 下書きとして保存するか


class CustomModelRequest(BaseModel):
    """カスタムモデル追加リクエスト"""
    provider: str
    model_name: str


@router.get("/providers")
async def get_providers():
    """利用可能なLLMプロバイダー一覧を取得"""
    try:
        providers = llm_service.get_available_providers()
        return {
            "providers": providers,
            "default_provider": llm_service.default_provider.value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"プロバイダー取得エラー: {str(e)}")


@router.post("/test-connection/{provider}")
async def test_connection(provider: str):
    """LLM接続テスト"""
    try:
        # プロバイダー名を LLMProvider に変換
        provider_enum = None
        for p in LLMProvider:
            if p.value == provider.lower():
                provider_enum = p
                break
        
        if not provider_enum:
            raise HTTPException(status_code=400, detail=f"サポートされていないプロバイダー: {provider}")
        
        result = await llm_service.test_connection(provider_enum)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"接続テストエラー: {str(e)}")


@router.post("/generate")
async def generate_content(request: GenerateContentRequest):
    """汎用コンテンツ生成"""
    try:
        # プロバイダー文字列を LLMProvider に変換
        provider = None
        if request.provider:
            for p in LLMProvider:
                if p.value == request.provider.lower():
                    provider = p
                    break
        
        result = await llm_service.generate_content(
            prompt=request.prompt,
            provider=provider,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_prompt=request.system_prompt
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"コンテンツ生成エラー: {str(e)}")


@router.post("/generate-social-post")
async def generate_social_post(request: GenerateSocialPostRequest):
    """ソーシャルメディア投稿生成"""
    try:
        # プロバイダー文字列を LLMProvider に変換
        provider = None
        if request.provider:
            for p in LLMProvider:
                if p.value == request.provider.lower():
                    provider = p
                    break
        
        result = await llm_service.generate_social_media_post(
            product_data=request.product_data,
            post_type=request.post_type,
            target_audience=request.target_audience,
            tone=request.tone,
            include_hashtags=request.include_hashtags,
            max_length=request.max_length,
            provider=provider
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ソーシャル投稿生成エラー: {str(e)}")


@router.post("/generate-thread")
async def generate_thread(request: GenerateThreadRequest):
    """スレッド投稿生成"""
    try:
        # プロバイダー文字列を LLMProvider に変換
        provider = None
        if request.provider:
            for p in LLMProvider:
                if p.value == request.provider.lower():
                    provider = p
                    break
        
        result = await llm_service.generate_thread_posts(
            topic=request.topic,
            num_posts=request.num_posts,
            max_length_per_post=request.max_length_per_post,
            provider=provider
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"スレッド生成エラー: {str(e)}")


@router.post("/generate-from-affiliate-data/{product_id}")
async def generate_from_affiliate_data(product_id: str, request: GenerateSocialPostRequest, sheet_name: str = "FANZAデータ"):
    """アフィリエイトデータから投稿生成"""
    try:
        from ..services.sheets_service import sheets_service
        from ..services.affiliate_data_service import affiliate_data_service
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info(f"投稿生成開始: product_id={product_id}, provider={request.provider}")
        
        if not sheets_service.is_authenticated():
            logger.error("Google Sheets API認証が完了していません")
            raise HTTPException(status_code=400, detail="Google Sheets API認証が必要です")
        
        # アフィリエイトデータを取得
        affiliate_data = sheets_service.get_affiliate_data("FANZAデータ")
        logger.info(f"取得したアフィリエイトデータ件数: {len(affiliate_data)}")
        
        # 最初の数件のデータ構造をログ出力
        if affiliate_data:
            logger.info(f"データサンプル（最初の1件）のキー: {list(affiliate_data[0].keys())}")
            for i, item in enumerate(affiliate_data[:3]):
                logger.debug(f"サンプルデータ {i+1}: {item}")
        
        # データフォーマットを自動判別
        detected_format = affiliate_data_service.detect_data_format(affiliate_data)
        logger.info(f"検出されたデータフォーマット: {detected_format.value}")
        
        # 指定されたproduct_idのデータを検索（フォーマット対応）
        product_data = affiliate_data_service.get_product_by_id(affiliate_data, product_id, detected_format)
        
        if not product_data:
            # 利用可能なIDを取得（正規化後）
            normalized_data = affiliate_data_service.normalize_data(affiliate_data, detected_format)
            available_ids = [item.get('id') for item in normalized_data[:5] if item.get('id')]
            logger.error(f"商品ID {product_id} が見つかりません。利用可能なID: {available_ids}")
            raise HTTPException(status_code=404, detail=f"商品ID {product_id} が見つかりません")
        
        # プロバイダー文字列を LLMProvider に変換
        provider = None
        if request.provider:
            for p in LLMProvider:
                if p.value == request.provider.lower():
                    provider = p
                    break
        
        if not provider:
            logger.error(f"無効なプロバイダー: {request.provider}")
            raise HTTPException(status_code=400, detail=f"サポートされていないプロバイダー: {request.provider}")
        
        logger.info(f"LLM投稿生成開始: provider={provider.value}")
        logger.info(f"生成パラメータ: post_type={request.post_type}, tone={request.tone}, max_length={request.max_length}")
        
        # 投稿生成
        try:
            logger.info("=== LLM投稿生成処理開始 ===")
            logger.info(f"使用プロバイダー: {provider.value}")
            logger.info(f"使用商品データ: {product_data.get('Title') or product_data.get('商品名', 'Unknown')}")
            logger.info(f"生成パラメータ: type={request.post_type}, tone={request.tone}, audience={request.target_audience}")
            
            result = await llm_service.generate_social_media_post(
                product_data=product_data,
                post_type=request.post_type,
                target_audience=request.target_audience,
                tone=request.tone,
                include_hashtags=request.include_hashtags,
                max_length=request.max_length,
                provider=provider
            )
            
            logger.info("=== LLM投稿生成処理完了 ===")
            logger.info(f"生成結果: success={result.get('success')}")
            logger.info(f"プロバイダー: {result.get('provider')}")
            logger.info(f"モデル: {result.get('model')}")
            
            if not result.get('success'):
                logger.error(f"LLM生成失敗: {result.get('error')}")
                logger.error(f"失敗理由詳細: {result}")
            else:
                content = result.get('content', '')
                logger.info(f"生成されたコンテンツ長: {len(content)}")
                logger.info(f"コンテンツプレビュー: {content[:100]}..." if len(content) > 100 else f"コンテンツ: {content}")
                
        except Exception as llm_error:
            logger.error("=== LLM生成中に例外発生 ===")
            logger.error(f"例外内容: {llm_error}")
            logger.error(f"例外タイプ: {type(llm_error).__name__}")
            logger.error(f"例外args: {llm_error.args}")
            import traceback
            logger.error(f"スタックトレース: {traceback.format_exc()}")
            raise
        
        # 生成結果にソースデータ情報を追加
        if result["success"]:
            result["source_data"] = {
                "product_id": product_id,
                "product_name": product_data.get("title", "Unknown Product"),
                "category": product_data.get("category", "商品"),
                "detected_format": detected_format.value
            }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"アフィリエイト投稿生成エラー: {str(e)}")


@router.get("/affiliate-products")
async def get_affiliate_products(sheet_name: str = "アフィリ用データ"):
    """アフィリエイト商品一覧を取得"""
    try:
        from ..services.sheets_service import sheets_service
        from ..services.affiliate_data_service import affiliate_data_service
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info("アフィリエイト商品一覧取得API開始")
        
        if not sheets_service.is_authenticated():
            logger.error("Google Sheets API認証が完了していません")
            raise HTTPException(status_code=400, detail="Google Sheets API認証が必要です")
        
        logger.info(f"スプレッドシート '{sheet_name}' からアフィリエイトデータを取得中...")
        affiliate_data = sheets_service.get_affiliate_data(sheet_name)
        logger.info(f"シート '{sheet_name}' から取得したアフィリエイトデータ件数: {len(affiliate_data)}")
        
        # データフォーマットを自動判別し正規化
        detected_format = affiliate_data_service.detect_data_format(affiliate_data)
        logger.info(f"検出されたデータフォーマット: {detected_format.value}")
        
        normalized_data = affiliate_data_service.normalize_data(affiliate_data, detected_format)
        logger.info(f"正規化後の商品件数: {len(normalized_data)}")
        
        return {
            "products": normalized_data,
            "total": len(normalized_data),
            "sheet_name": sheet_name,
            "detected_format": detected_format.value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"商品一覧取得エラー: {e}")
        logger.error(f"エラータイプ: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=f"商品一覧取得エラー: {str(e)}")


@router.post("/generate-batch-posts")
async def generate_batch_posts(
    background_tasks: BackgroundTasks,
    product_ids: List[str],
    generation_settings: GenerateSocialPostRequest
):
    """複数商品の投稿を一括生成"""
    try:
        from ..services.sheets_service import sheets_service
        
        if not sheets_service.is_authenticated():
            raise HTTPException(status_code=400, detail="Google Sheets API認証が必要です")
        
        # バックグラウンドタスクで一括生成を実行
        background_tasks.add_task(
            execute_batch_generation,
            product_ids,
            generation_settings
        )
        
        return {
            "message": f"{len(product_ids)} 商品の投稿生成を開始しました",
            "product_count": len(product_ids),
            "status": "processing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"一括生成エラー: {str(e)}")


async def execute_batch_generation(product_ids: List[str], settings: GenerateSocialPostRequest):
    """一括生成のバックグラウンド処理"""
    try:
        from ..services.sheets_service import sheets_service
        from ..core.database import SessionLocal
        from ..models import ScheduledPost
        import uuid
        from datetime import datetime, timezone, timedelta
        
        # アフィリエイトデータを取得
        affiliate_data = sheets_service.get_affiliate_data("FANZAデータ")
        
        # データフォーマットを自動判別
        from ..services.affiliate_data_service import affiliate_data_service
        detected_format = affiliate_data_service.detect_data_format(affiliate_data)
        
        # プロバイダー設定
        provider = None
        if settings.provider:
            for p in LLMProvider:
                if p.value == settings.provider.lower():
                    provider = p
                    break
        
        db = SessionLocal()
        try:
            generated_count = 0
            
            for product_id in product_ids:
                try:
                    # 商品データを検索（フォーマット対応）
                    product_data = affiliate_data_service.get_product_by_id(affiliate_data, product_id, detected_format)
                    
                    if not product_data:
                        continue
                    
                    # 投稿内容を生成
                    result = await llm_service.generate_social_media_post(
                        product_data=product_data,
                        post_type=settings.post_type,
                        target_audience=settings.target_audience,
                        tone=settings.tone,
                        include_hashtags=settings.include_hashtags,
                        max_length=settings.max_length,
                        provider=provider
                    )
                    
                    if result["success"]:
                        # 予約投稿として保存（24時間後にスケジュール）
                        scheduled_time = datetime.now(timezone.utc) + timedelta(hours=24 + generated_count)
                        
                        scheduled_post = ScheduledPost(
                            id=str(uuid.uuid4()),
                            content=result["content"],
                            scheduled_time=scheduled_time,
                            bot_id="default_bot",  # デフォルトボット（要調整）
                            is_ai_generated=True,
                            ai_prompt=f"Generated from product: {product_id}",
                            source_data_id=product_id,
                            status="pending"
                        )
                        
                        db.add(scheduled_post)
                        generated_count += 1
                        
                        # 生成間隔を空ける（API制限対策）
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    print(f"商品 {product_id} の生成エラー: {e}")
                    continue
            
            db.commit()
            print(f"一括生成完了: {generated_count} 件の投稿を生成しました")
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"一括生成バックグラウンド処理エラー: {e}")


@router.get("/config")
async def get_llm_config():
    """LLM設定を取得"""
    try:
        configs = llm_service.get_all_configs()
        return {
            "configs": configs,
            "available_providers": llm_service.get_available_providers()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"設定取得エラー: {str(e)}")


@router.get("/config/configured")
async def get_configured_providers():
    """設定済みのLLMプロバイダーのみを取得"""
    try:
        all_providers = llm_service.get_available_providers()
        configured_providers = []
        
        for provider in all_providers:
            provider_name = provider.get('name', '')
            config = llm_service.get_provider_config(provider_name)
            
            # プロバイダーが正しく設定されているかチェック
            is_configured = False
            if provider_name == 'ollama':
                # Ollamaは常に利用可能
                is_configured = True
            else:
                # その他のプロバイダーはAPIキーが設定されているかチェック
                api_key = config.get('api_key', '')
                is_configured = bool(api_key and api_key.strip())
            
            if is_configured:
                configured_providers.append({
                    "name": provider_name,
                    "display_name": provider_name.upper(),
                    "available": True,
                    "config": {
                        "api_key_configured": bool(config.get('api_key')) if provider_name != 'ollama' else False,
                        "base_url": config.get('base_url', ''),
                        "model": config.get('model', '')
                    }
                })
        
        return {
            "configured_providers": configured_providers,
            "count": len(configured_providers)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"設定済みプロバイダー取得エラー: {str(e)}")


@router.post("/config")
async def update_llm_config(request: LLMConfigRequest):
    """LLM設定を更新"""
    try:
        # プロバイダーの有効性をチェック
        valid_providers = [p.value for p in LLMProvider]
        if request.provider.lower() not in valid_providers:
            raise HTTPException(status_code=400, detail=f"サポートされていないプロバイダー: {request.provider}")
        
        # 設定を構築
        config_data = {}
        if request.api_key is not None:
            config_data['api_key'] = request.api_key
        if request.base_url is not None:
            config_data['base_url'] = request.base_url
        if request.model is not None:
            config_data['model'] = request.model
        
        # 設定を更新（バリデーション付き）
        success, message = llm_service.update_provider_config(request.provider.lower(), config_data)
        
        if success:
            return {
                "message": message,
                "provider": request.provider,
                "config": llm_service.get_provider_config(request.provider.lower())
            }
        else:
            # バリデーションエラーの場合は400、その他は500
            if "無効な" in message or "形式" in message:
                raise HTTPException(status_code=400, detail=message)
            else:
                raise HTTPException(status_code=500, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"設定更新エラー: {str(e)}")


@router.get("/config/{provider}")
async def get_provider_config(provider: str):
    """特定プロバイダーの設定を取得"""
    try:
        # プロバイダーの有効性をチェック
        valid_providers = [p.value for p in LLMProvider]
        if provider.lower() not in valid_providers:
            raise HTTPException(status_code=400, detail=f"サポートされていないプロバイダー: {provider}")
        
        config = llm_service.get_provider_config(provider.lower())
        
        # APIキーは安全のため、存在有無のみ返す
        safe_config = config.copy()
        if 'api_key' in safe_config:
            safe_config['api_key_configured'] = bool(safe_config['api_key'])
            del safe_config['api_key']
        
        return {
            "provider": provider,
            "config": safe_config
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"設定取得エラー: {str(e)}")


@router.delete("/config/{provider}")
async def reset_provider_config(provider: str):
    """特定プロバイダーの設定をリセット"""
    try:
        # プロバイダーの有効性をチェック
        valid_providers = [p.value for p in LLMProvider]
        if provider.lower() not in valid_providers:
            raise HTTPException(status_code=400, detail=f"サポートされていないプロバイダー: {provider}")
        
        # デフォルト設定にリセット
        default_config = llm_service.load_config().get(provider.lower(), {})
        default_config.update({
            'api_key': '',
            'base_url': '',
            'model': llm_service.default_models.get(LLMProvider(provider.lower()), '')
        })
        
        success, message = llm_service.update_provider_config(provider.lower(), default_config)
        
        if success:
            return {
                "message": f"{provider} の設定をリセットしました",
                "provider": provider
            }
        else:
            raise HTTPException(status_code=500, detail=f"設定のリセットに失敗しました: {message}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"設定リセットエラー: {str(e)}")


@router.get("/models/{provider}")
async def get_provider_models(provider: str):
    """プロバイダーの利用可能モデル一覧を取得"""
    try:
        # プロバイダーの有効性をチェック
        valid_providers = [p.value for p in LLMProvider]
        if provider.lower() not in valid_providers:
            raise HTTPException(status_code=400, detail=f"サポートされていないプロバイダー: {provider}")
        
        # Ollamaの場合は実際のモデル一覧も取得
        installed_models = []
        if provider.lower() == 'ollama':
            installed_models = await llm_service.get_ollama_models()
        
        all_models = llm_service.get_all_models(provider.lower())
        
        return {
            "provider": provider,
            "models": all_models,
            "installed_models": installed_models if provider.lower() == 'ollama' else [],
            "custom_models": llm_service.custom_models.get(provider.lower(), [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"モデル一覧取得エラー: {str(e)}")


@router.post("/models/add")
async def add_custom_model(request: CustomModelRequest):
    """カスタムモデルを追加"""
    try:
        # プロバイダーの有効性をチェック
        valid_providers = [p.value for p in LLMProvider]
        if request.provider.lower() not in valid_providers:
            raise HTTPException(status_code=400, detail=f"サポートされていないプロバイダー: {request.provider}")
        
        if not request.model_name.strip():
            raise HTTPException(status_code=400, detail="モデル名を入力してください")
        
        success = llm_service.add_custom_model(request.provider.lower(), request.model_name.strip())
        
        if success:
            return {
                "message": f"{request.provider} に {request.model_name} を追加しました",
                "provider": request.provider,
                "model_name": request.model_name,
                "all_models": llm_service.get_all_models(request.provider.lower())
            }
        else:
            raise HTTPException(status_code=500, detail="カスタムモデルの追加に失敗しました")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"カスタムモデル追加エラー: {str(e)}")


@router.delete("/models/{provider}/{model_name}")
async def remove_custom_model(provider: str, model_name: str):
    """カスタムモデルを削除"""
    try:
        # プロバイダーの有効性をチェック
        valid_providers = [p.value for p in LLMProvider]
        if provider.lower() not in valid_providers:
            raise HTTPException(status_code=400, detail=f"サポートされていないプロバイダー: {provider}")
        
        success = llm_service.remove_custom_model(provider.lower(), model_name)
        
        if success:
            return {
                "message": f"{provider} から {model_name} を削除しました",
                "provider": provider,
                "model_name": model_name,
                "all_models": llm_service.get_all_models(provider.lower())
            }
        else:
            raise HTTPException(status_code=500, detail="カスタムモデルの削除に失敗しました")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"カスタムモデル削除エラー: {str(e)}")


@router.get("/ollama/installed-models")
async def get_ollama_installed_models():
    """Ollamaのインストール済みモデル一覧を取得"""
    try:
        models = await llm_service.get_ollama_models()
        return {
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollamaモデル一覧取得エラー: {str(e)}")


@router.get("/debug/system-status")
async def get_system_debug_status():
    """システム状況の詳細デバッグ情報を取得"""
    import logging
    import os
    import sys
    
    try:
        # LLM設定状況
        config_status = llm_service.get_all_configs()
        
        # プロバイダー利用可能性
        providers_status = llm_service.get_available_providers()
        
        # Ollama特別チェック
        ollama_status = None
        try:
            ollama_models = await llm_service.get_ollama_models()
            ollama_config = llm_service.get_provider_config('ollama')
            ollama_status = {
                "base_url": ollama_config.get('base_url', 'http://localhost:11434'),
                "available_models": ollama_models,
                "model_count": len(ollama_models),
                "connection_test": "pending"
            }
            
            # Ollama接続テスト
            try:
                test_result = await llm_service.test_connection(LLMProvider.OLLAMA)
                ollama_status["connection_test"] = test_result
            except Exception as test_error:
                ollama_status["connection_test"] = {"error": str(test_error)}
                
        except Exception as ollama_error:
            ollama_status = {"error": str(ollama_error)}
        
        # システム情報
        system_info = {
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "environment_vars": {
                "OPENAI_API_KEY": "設定済み" if os.environ.get('OPENAI_API_KEY') else "未設定",
                "CLAUDE_API_KEY": "設定済み" if os.environ.get('CLAUDE_API_KEY') else "未設定", 
                "GEMINI_API_KEY": "設定済み" if os.environ.get('GEMINI_API_KEY') else "未設定",
                "OLLAMA_ENDPOINT": os.environ.get('OLLAMA_ENDPOINT', '未設定')
            }
        }
        
        # ログ設定情報
        root_logger = logging.getLogger()
        log_info = {
            "level": root_logger.level,
            "level_name": logging.getLevelName(root_logger.level),
            "handlers": [
                {
                    "type": type(handler).__name__,
                    "level": handler.level,
                    "level_name": logging.getLevelName(handler.level)
                } for handler in root_logger.handlers
            ]
        }
        
        return {
            "llm_config": config_status,
            "providers": providers_status,
            "ollama_detailed": ollama_status,
            "system": system_info,
            "logging": log_info,
            "timestamp": "now"
        }
        
    except Exception as e:
        import traceback
        return {
            "error": f"システム状況取得エラー: {str(e)}",
            "traceback": traceback.format_exc(),
            "debug_mode": True
        }


@router.post("/debug/test-generation")
async def test_generation_process():
    """LLM生成プロセスを段階的にテスト"""
    import logging
    import traceback
    from datetime import datetime
    
    debug_log = []
    
    def add_log(message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        debug_log.append(f"[{timestamp}] {level}: {message}")
        if level == "INFO":
            logger.info(message)
        elif level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
    
    try:
        add_log("=== 生成プロセステスト開始 ===")
        
        # ステップ1: LLMサービス設定確認
        add_log("ステップ1: LLMサービス設定確認")
        config = llm_service.get_all_configs()
        add_log(f"設定されたプロバイダー: {list(config.keys())}")
        
        # ステップ2: プロバイダー利用可能性確認
        add_log("ステップ2: プロバイダー確認")
        providers = llm_service.get_available_providers()
        available_providers = [p for p in providers if p['available']]
        add_log(f"利用可能プロバイダー: {[p['name'] for p in available_providers]}")
        
        if not available_providers:
            add_log("❌ 利用可能なプロバイダーがありません", "ERROR")
            return {"debug_log": debug_log, "success": False, "error": "No available providers"}
        
        # ステップ3: テスト用データ準備
        add_log("ステップ3: テスト用データ準備")
        test_product = {
            "Title": "テスト商品",
            "Description": "これはテスト用の商品です。",
            "Product ID": "TEST001",
            "URL": "https://example.com/test"
        }
        add_log(f"テスト商品データ: {test_product}")
        
        # ステップ4: 各プロバイダーでテスト生成
        generation_results = {}
        
        for provider_info in available_providers:
            provider_name = provider_info['name']
            add_log(f"ステップ4-{provider_name}: {provider_name}でテスト生成開始")
            
            try:
                # プロバイダー文字列を LLMProvider に変換
                provider_enum = None
                for p in LLMProvider:
                    if p.value == provider_name:
                        provider_enum = p
                        break
                
                if not provider_enum:
                    add_log(f"❌ プロバイダー変換失敗: {provider_name}", "ERROR")
                    continue
                
                add_log(f"プロバイダー変換成功: {provider_enum}")
                
                # 簡単な生成テスト
                test_request = GenerateSocialPostRequest(
                    product_data=test_product,
                    post_type="promotional",
                    target_audience="general", 
                    tone="friendly",
                    include_hashtags=True,
                    max_length=100,
                    provider=provider_name
                )
                
                add_log(f"リクエスト作成完了: {test_request}")
                
                result = await llm_service.generate_social_media_post(
                    product_data=test_product,
                    post_type="promotional",
                    target_audience="general",
                    tone="friendly",
                    include_hashtags=True,
                    max_length=100,
                    provider=provider_enum
                )
                
                generation_results[provider_name] = result
                
                if result.get('success'):
                    add_log(f"✅ {provider_name} 生成成功: {len(result.get('content', ''))} 文字")
                else:
                    add_log(f"❌ {provider_name} 生成失敗: {result.get('error')}", "ERROR")
                
            except Exception as provider_error:
                error_msg = f"❌ {provider_name} 例外発生: {str(provider_error)}"
                add_log(error_msg, "ERROR")
                add_log(f"スタックトレース: {traceback.format_exc()}", "ERROR")
                generation_results[provider_name] = {"success": False, "error": str(provider_error)}
        
        add_log("=== 生成プロセステスト完了 ===")
        
        return {
            "debug_log": debug_log,
            "success": True,
            "generation_results": generation_results,
            "available_providers": [p['name'] for p in available_providers],
            "test_data": test_product
        }
        
    except Exception as e:
        add_log(f"=== テスト中に重大エラー発生 ===", "ERROR")
        add_log(f"エラー: {str(e)}", "ERROR")
        add_log(f"スタックトレース: {traceback.format_exc()}", "ERROR")
        
        return {
            "debug_log": debug_log,
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@router.post("/quick-ollama-test")
async def quick_ollama_test():
    """Ollamaでの簡単な投稿生成テスト"""
    try:
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("=== Ollama簡単テスト開始 ===")
        
        # Ollamaの利用可能モデルを取得
        available_models = await llm_service.get_ollama_models()
        logger.info(f"利用可能なOllamaモデル: {available_models}")
        
        if not available_models:
            return {
                "success": False,
                "error": "Ollamaにモデルがインストールされていません。`ollama pull llama2` などでモデルをインストールしてください。",
                "available_models": []
            }
        
        # 最初に見つかったモデルを使用
        model_to_use = available_models[0]
        logger.info(f"使用するモデル: {model_to_use}")
        
        # シンプルなテストプロンプト
        test_prompt = "プログラミング学習の本について、Twitterで投稿する短い文章を日本語で作成してください。ハッシュタグも含めて、魅力的で読みやすい投稿を140文字以内で作成してください。"
        
        logger.info(f"テストプロンプト: {test_prompt}")
        
        # Ollama生成テスト
        result = await llm_service._generate_ollama(
            prompt=test_prompt,
            model=model_to_use,
            max_tokens=200,
            temperature=0.7,
            system_prompt="あなたはSNS投稿の専門家です。"
        )
        
        logger.info(f"生成結果: {result}")
        
        if result.get("success"):
            return {
                "success": True,
                "content": result.get("content"),
                "model_used": model_to_use,
                "available_models": available_models,
                "message": "Ollama投稿生成成功！"
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "不明なエラー"),
                "model_used": model_to_use,
                "available_models": available_models
            }
            
    except Exception as e:
        import traceback
        logger.error(f"Ollama簡単テストエラー: {e}")
        logger.error(f"スタックトレース: {traceback.format_exc()}")
        
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@router.post("/auto-schedule-posts")
async def auto_schedule_posts(request: AutoScheduleRequest, background_tasks: BackgroundTasks):
    """アフィリエイト商品から自動投稿スケジュール生成"""
    try:
        from ..services.sheets_service import sheets_service
        from ..core.database import get_db
        from ..models import ScheduledPost, Bot
        from sqlalchemy.orm import Session
        from datetime import datetime, timezone, timedelta
        import uuid
        import json
        import random
        
        logger.info(f"自動スケジュール生成開始: {request.post_count}件, シート={request.sheet_name}")
        
        # Google Sheets認証チェック
        if not sheets_service.is_authenticated():
            raise HTTPException(status_code=400, detail="Google Sheets API認証が必要です")
        
        # ボットの存在確認
        db = next(get_db())
        try:
            bot = db.query(Bot).filter(Bot.id == request.bot_id).first()
            if not bot:
                raise HTTPException(status_code=404, detail="指定されたボットが見つかりません")
            
            # アフィリエイトデータを取得
            affiliate_data = sheets_service.get_affiliate_data(request.sheet_name)
            if not affiliate_data:
                raise HTTPException(status_code=404, detail=f"シート '{request.sheet_name}' にデータが見つかりません")
            
            # データフォーマットを自動判別し正規化
            from ..services.affiliate_data_service import affiliate_data_service
            detected_format = affiliate_data_service.detect_data_format(affiliate_data)
            normalized_data = affiliate_data_service.normalize_data(affiliate_data, detected_format)
            
            if len(normalized_data) < request.post_count:
                logger.warning(f"要求された件数({request.post_count})より商品データが少ない({len(normalized_data)}件)")
            
            # 開始日時を解析
            try:
                start_datetime = datetime.strptime(request.start_date, "%Y-%m-%d")
                start_datetime = start_datetime.replace(hour=9, minute=0, second=0, tzinfo=timezone.utc)  # 朝9時開始
            except ValueError:
                raise HTTPException(status_code=400, detail="開始日の形式が正しくありません (YYYY-MM-DD)")
            
            generated_posts = []
            successful_posts = 0
            errors = []
            
            # ランダムに商品を選択
            selected_products = random.sample(normalized_data, min(request.post_count, len(normalized_data)))
            
            for i, product in enumerate(selected_products):
                try:
                    # 投稿時間を計算
                    post_time = start_datetime + timedelta(hours=request.interval_hours * i)
                    
                    # 商品データは既に正規化済みなのでそのまま使用
                    
                    # LLMで投稿内容を生成
                    provider_enum = None
                    for p in LLMProvider:
                        if p.value == request.provider.lower():
                            provider_enum = p
                            break
                    
                    if not provider_enum:
                        raise Exception(f"無効なプロバイダー: {request.provider}")
                    
                    result = await llm_service.generate_social_media_post(
                        product_data=product,
                        post_type=request.post_type,
                        target_audience=request.target_audience,
                        tone=request.tone,
                        include_hashtags=request.include_hashtags,
                        max_length=request.max_length,
                        provider=provider_enum,
                        model=request.model
                    )
                    
                    if result.get("success"):
                        # 画像URLを取得（正規化済みデータから）
                        image_urls = product.get("image_urls", [])
                        
                        # 予約投稿を作成
                        scheduled_post = ScheduledPost(
                            id=str(uuid.uuid4()),
                            content=result["content"],
                            scheduled_time=post_time,
                            bot_id=request.bot_id,
                            image_urls=json.dumps(image_urls[:4]) if image_urls else None,  # 最大4枚
                            status="pending",
                            is_ai_generated=True,
                            ai_prompt=f"Auto-generated from product: {product.get('title', 'Unknown')}",
                            source_data_id=product.get("id")
                        )
                        
                        db.add(scheduled_post)
                        generated_posts.append({
                            "product_id": product.get("id"),
                            "product_title": product.get("title", "Unknown"),
                            "scheduled_time": post_time.isoformat(),
                            "content_preview": result["content"][:100] + "..." if len(result["content"]) > 100 else result["content"],
                            "image_count": len(image_urls)
                        })
                        successful_posts += 1
                        
                    else:
                        errors.append(f"商品 {product.get('title', 'Unknown')}: 投稿生成失敗")
                        
                except Exception as e:
                    error_msg = f"商品 {product.get('title', 'Unknown')}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            db.commit()
            
            return {
                "success": True,
                "message": f"{successful_posts}件の投稿を自動スケジュールしました",
                "generated_posts": generated_posts,
                "successful_count": successful_posts,
                "total_requested": request.post_count,
                "errors": errors,
                "next_post_time": start_datetime.isoformat(),
                "last_post_time": (start_datetime + timedelta(hours=request.interval_hours * (successful_posts - 1))).isoformat() if successful_posts > 0 else None
            }
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"自動スケジュール生成エラー: {e}")
        raise HTTPException(status_code=500, detail=f"自動スケジュール生成に失敗しました: {str(e)}")


@router.post("/batch-generate-posts")
async def batch_generate_posts(request: BatchGenerateRequest):
    """複数商品から一括投稿生成"""
    try:
        from ..services.sheets_service import sheets_service
        import asyncio
        
        logger.info(f"バッチ投稿生成開始: {len(request.product_ids)}件")
        
        # Google Sheets認証チェック
        if not sheets_service.is_authenticated():
            raise HTTPException(status_code=400, detail="Google Sheets API認証が必要です")
        
        # アフィリエイトデータを取得
        affiliate_data = sheets_service.get_affiliate_data(request.sheet_name)
        if not affiliate_data:
            raise HTTPException(status_code=404, detail=f"シート '{request.sheet_name}' にデータが見つかりません")
        
        # データフォーマットを自動判別
        from ..services.affiliate_data_service import affiliate_data_service
        detected_format = affiliate_data_service.detect_data_format(affiliate_data)
        
        # 商品IDに基づいてデータを選択
        selected_products = []
        for product_id in request.product_ids:
            product_data = affiliate_data_service.get_product_by_id(affiliate_data, product_id, detected_format)
            if product_data:
                selected_products.append(product_data)
        
        if not selected_products:
            raise HTTPException(status_code=404, detail="指定された商品IDが見つかりません")
        
        # プロバイダー設定
        provider_enum = None
        for p in LLMProvider:
            if p.value == request.provider.lower():
                provider_enum = p
                break
        
        if not provider_enum:
            raise HTTPException(status_code=400, detail=f"無効なプロバイダー: {request.provider}")
        
        generated_posts = []
        successful_posts = 0
        errors = []
        
        # 並列処理で投稿を生成
        async def generate_single_post(product):
            try:
                # 商品データは既に正規化済みなのでそのまま使用
                
                # LLMで投稿内容を生成
                result = await llm_service.generate_social_media_post(
                    product_data=product,
                    post_type=request.post_type,
                    target_audience=request.target_audience,
                    tone=request.tone,
                    include_hashtags=request.include_hashtags,
                    max_length=request.max_length,
                    provider=provider_enum,
                    model=request.model
                )
                
                if result.get("success"):
                    return {
                        "success": True,
                        "product_id": product.get("id"),
                        "product_title": product.get("title", "Unknown"),
                        "content": result["content"],
                        "image_urls": product.get("image_urls", [])[:4],  # 最大4枚
                        "provider": request.provider,
                        "model": result.get("model", request.model),
                        "usage": result.get("usage", {})
                    }
                else:
                    return {
                        "success": False,
                        "product_id": product.get("id"),
                        "product_title": product.get("title", "Unknown"),
                        "error": result.get("error", "投稿生成失敗")
                    }
                    
            except Exception as e:
                return {
                    "success": False,
                    "product_id": product.get("id"),
                    "product_title": product.get("title", "Unknown"),
                    "error": str(e)
                }
        
        # 並列実行（セマフォで同時実行数を制限）
        semaphore = asyncio.Semaphore(3)  # 同時に3つまで
        
        async def limited_generate(product):
            async with semaphore:
                return await generate_single_post(product)
        
        # すべての投稿を並列生成
        tasks = [limited_generate(product) for product in selected_products]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 結果を整理
        for result in results:
            if isinstance(result, Exception):
                errors.append(f"実行エラー: {str(result)}")
            elif result.get("success"):
                generated_posts.append(result)
                successful_posts += 1
            else:
                errors.append(f"商品 {result.get('product_title', 'Unknown')}: {result.get('error', 'Unknown error')}")
        
        return {
            "success": True,
            "message": f"{successful_posts}件の投稿を生成しました",
            "generated_posts": generated_posts,
            "successful_count": successful_posts,
            "total_requested": len(request.product_ids),
            "errors": errors,
            "save_as_drafts": request.save_as_drafts,
            "batch_id": str(__import__('uuid').uuid4())  # バッチ処理ID
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"バッチ投稿生成エラー: {e}")
        raise HTTPException(status_code=500, detail=f"バッチ投稿生成に失敗しました: {str(e)}")


@router.post("/debug/gemini-connection-test")
async def debug_gemini_connection():
    """Gemini接続テストの詳細デバッグ"""
    import logging
    import traceback
    from datetime import datetime
    
    logger = logging.getLogger(__name__)
    debug_log = []
    
    def add_log(message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        debug_log.append(f"[{timestamp}] {level}: {message}")
        if level == "INFO":
            logger.info(message)
        elif level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
    
    try:
        add_log("=== Gemini接続デバッグテスト開始 ===")
        
        # 現在の設定を確認
        config = llm_service.get_provider_config('gemini')
        api_key = config.get('api_key', '')
        base_url = config.get('base_url', 'https://generativelanguage.googleapis.com')
        model = config.get('model', 'gemini-1.5-flash')
        
        add_log(f"設定確認 - APIキー: {api_key[:10] if api_key else 'なし'}..., モデル: {model}")
        
        # APIキー形式チェック
        if not api_key:
            return {"debug_log": debug_log, "success": False, "error": "APIキーが設定されていません"}
        
        valid, error_msg = llm_service._validate_api_key("gemini", api_key)
        if not valid:
            add_log(f"APIキー形式エラー: {error_msg}", "ERROR")
            return {"debug_log": debug_log, "success": False, "error": error_msg}
        
        add_log("APIキー形式チェック: OK")
        
        # 実際のAPI呼び出しテスト
        add_log("実際のAPI呼び出しテスト開始")
        
        test_result = await llm_service._test_gemini_connection()
        
        add_log(f"テスト結果: {test_result}")
        
        return {
            "debug_log": debug_log,
            "success": test_result.get("connected", False),
            "test_result": test_result,
            "config_check": {
                "api_key_length": len(api_key) if api_key else 0,
                "api_key_format": "OK" if valid else "NG",
                "base_url": base_url,
                "model": model
            }
        }
        
    except Exception as e:
        add_log(f"デバッグテスト中にエラー: {str(e)}", "ERROR")
        add_log(f"スタックトレース: {traceback.format_exc()}", "ERROR")
        
        return {
            "debug_log": debug_log,
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@router.post("/debug/network-test")
async def debug_network_test():
    """基本的なネットワーク接続テスト"""
    import aiohttp
    import asyncio
    import time
    
    results = {}
    
    # 基本的な接続テスト
    test_urls = [
        ("Google", "https://www.google.com"),
        ("Gemini Base", "https://generativelanguage.googleapis.com"),
        ("GitHub", "https://api.github.com")
    ]
    
    for name, url in test_urls:
        try:
            start_time = time.time()
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as response:
                    end_time = time.time()
                    results[name] = {
                        "success": True,
                        "status": response.status,
                        "time_ms": round((end_time - start_time) * 1000, 2),
                        "headers": dict(response.headers)
                    }
        except Exception as e:
            results[name] = {
                "success": False,
                "error": str(e),
                "time_ms": None
            }
    
    # Gemini APIキー形式チェック
    config = llm_service.get_provider_config('gemini')
    api_key = config.get('api_key', '')
    
    api_key_check = {
        "configured": bool(api_key),
        "length": len(api_key) if api_key else 0,
        "starts_with_AIza": api_key.startswith("AIza") if api_key else False,
        "is_39_chars": len(api_key) == 39 if api_key else False
    }
    
    return {
        "network_tests": results,
        "api_key_check": api_key_check,
        "timestamp": time.time()
    }


@router.post("/debug/simple-gemini-test")
async def simple_gemini_test():
    """最もシンプルなGemini APIテスト"""
    import aiohttp
    import json
    
    config = llm_service.get_provider_config('gemini')
    api_key = config.get('api_key', '')
    
    if not api_key:
        return {"success": False, "error": "APIキーが設定されていません"}
    
    # 最もシンプルなペイロード
    simple_payload = {
        "contents": [
            {
                "parts": [
                    {"text": "Hi"}
                ]
            }
        ]
    }
    
    # v1とv1betaの両方をテスト
    test_results = {}
    
    for version in ["v1", "v1beta"]:
        for model in ["gemini-1.5-flash", "gemini-pro"]:
            test_key = f"{version}/{model}"
            
            try:
                url = f"https://generativelanguage.googleapis.com/{version}/models/{model}:generateContent?key={api_key}"
                
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
                    async with session.post(
                        url,
                        headers={"Content-Type": "application/json"},
                        json=simple_payload
                    ) as response:
                        response_text = await response.text()
                        
                        test_results[test_key] = {
                            "status": response.status,
                            "response_length": len(response_text),
                            "response_preview": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                            "success": response.status == 200
                        }
                        
                        # 最初に成功したものを返す
                        if response.status == 200:
                            try:
                                json_result = json.loads(response_text)
                                return {
                                    "success": True,
                                    "working_config": {
                                        "version": version,
                                        "model": model,
                                        "url": url.split('?')[0]
                                    },
                                    "response": json_result,
                                    "all_tests": test_results
                                }
                            except json.JSONDecodeError:
                                test_results[test_key]["json_error"] = "JSONデコードエラー"
                        
            except Exception as e:
                test_results[test_key] = {
                    "error": str(e),
                    "success": False
                }
    
    return {
        "success": False,
        "error": "全てのテストが失敗しました",
        "all_tests": test_results
    }


@router.get("/debug/affiliate-data-format/{sheet_name}")
async def debug_affiliate_data_format(sheet_name: str):
    """アフィリエイトデータフォーマットのデバッグ情報を取得"""
    try:
        from ..services.sheets_service import sheets_service
        from ..services.affiliate_data_service import affiliate_data_service
        
        if not sheets_service.is_authenticated():
            return {"error": "Google Sheets API認証が必要です"}
        
        # 元データを取得
        raw_data = sheets_service.get_affiliate_data(sheet_name)
        
        if not raw_data:
            return {"error": f"シート '{sheet_name}' にデータが見つかりません"}
        
        # フォーマット判別
        detected_format = affiliate_data_service.detect_data_format(raw_data)
        
        # 正規化
        normalized_data = affiliate_data_service.normalize_data(raw_data, detected_format)
        
        # サンプルデータ
        sample_raw = raw_data[:3] if len(raw_data) >= 3 else raw_data
        sample_normalized = normalized_data[:3] if len(normalized_data) >= 3 else normalized_data
        
        return {
            "sheet_name": sheet_name,
            "raw_data_count": len(raw_data),
            "normalized_data_count": len(normalized_data),
            "detected_format": detected_format.value,
            "raw_data_keys": list(raw_data[0].keys()) if raw_data else [],
            "normalized_data_keys": list(normalized_data[0].keys()) if normalized_data else [],
            "sample_raw_data": sample_raw,
            "sample_normalized_data": sample_normalized,
            "format_detection_info": {
                "available_formats": [f.value for f in affiliate_data_service.format_keys.keys()],
                "format_mapping": affiliate_data_service.format_keys
            }
        }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@router.get("/debug/search-product/{product_id}")
async def debug_search_product(product_id: str, sheet_name: str = "FANZAデータ"):
    """特定の商品IDを詳細検索してデバッグ情報を提供"""
    try:
        from ..services.sheets_service import sheets_service
        from ..services.affiliate_data_service import affiliate_data_service
        
        if not sheets_service.is_authenticated():
            return {"error": "Google Sheets API認証が必要です"}
        
        # 元データを取得
        raw_data = sheets_service.get_affiliate_data(sheet_name)
        
        if not raw_data:
            return {"error": f"シート '{sheet_name}' にデータが見つかりません"}
        
        # 検索結果
        search_results = {
            "searched_id": product_id,
            "sheet_name": sheet_name,
            "total_items": len(raw_data),
            "found_in_raw": False,
            "found_in_normalized": False,
            "raw_matches": [],
            "normalized_matches": [],
            "similar_ids": [],
            "all_ids_sample": []
        }
        
        # 元データで検索
        for i, item in enumerate(raw_data):
            for key, value in item.items():
                if value and str(value).lower() == product_id.lower():
                    search_results["found_in_raw"] = True
                    search_results["raw_matches"].append({
                        "index": i,
                        "key": key,
                        "value": value,
                        "item_preview": {k: v for k, v in list(item.items())[:5]}
                    })
        
        # フォーマット判別と正規化
        detected_format = affiliate_data_service.detect_data_format(raw_data)
        normalized_data = affiliate_data_service.normalize_data(raw_data, detected_format)
        
        # 正規化データで検索
        for i, item in enumerate(normalized_data):
            if (item.get("id") == product_id or 
                item.get("product_id") == product_id or 
                item.get("content_id") == product_id):
                search_results["found_in_normalized"] = True
                search_results["normalized_matches"].append({
                    "index": i,
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "item_preview": {k: v for k, v in list(item.items())[:10] if k != "original_data"}
                })
        
        # 類似IDを検索（部分一致）
        for item in normalized_data:
            item_id = item.get("id")
            if item_id and product_id.lower() in item_id.lower():
                search_results["similar_ids"].append({
                    "id": item_id,
                    "title": item.get("title", "Unknown"),
                    "similarity": "contains"
                })
        
        # 全IDのサンプル
        for item in normalized_data[:20]:
            item_id = item.get("id")
            if item_id:
                search_results["all_ids_sample"].append({
                    "id": item_id,
                    "title": item.get("title", "Unknown")[:50]
                })
        
        return search_results
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }