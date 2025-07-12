"""LLM API エンドポイント"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import asyncio
from ..services.llm_service import llm_service, LLMProvider

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
    product_data: Dict[str, Any]
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
async def generate_from_affiliate_data(product_id: str, request: GenerateSocialPostRequest):
    """アフィリエイトデータから投稿生成"""
    try:
        from ..services.sheets_service import sheets_service
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info(f"投稿生成開始: product_id={product_id}, provider={request.provider}")
        
        if not sheets_service.is_authenticated():
            logger.error("Google Sheets API認証が完了していません")
            raise HTTPException(status_code=400, detail="Google Sheets API認証が必要です")
        
        # アフィリエイトデータを取得
        affiliate_data = sheets_service.get_affiliate_data("アフィリ用データ")
        logger.info(f"取得したアフィリエイトデータ件数: {len(affiliate_data)}")
        
        # 指定されたproduct_idのデータを検索（新フォーマット対応）
        product_data = None
        for item in affiliate_data:
            # 新フォーマット（Product ID, Content ID）と旧フォーマット（商品ID, product_id）の両方をチェック
            if (item.get("Product ID") == product_id or 
                item.get("Content ID") == product_id or
                item.get("商品ID") == product_id or 
                item.get("product_id") == product_id):
                product_data = item
                logger.info(f"商品データ発見: {product_data.get('Title', product_data.get('商品名', 'Unknown'))}")
                break
        
        if not product_data:
            logger.error(f"商品ID {product_id} が見つかりません。利用可能なID: {[item.get('Product ID') or item.get('Content ID') or item.get('商品ID') or item.get('product_id') for item in affiliate_data[:5]]}")
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
                "product_name": (product_data.get("Title") or 
                               product_data.get("商品名") or 
                               product_data.get("product_name") or 
                               "Unknown Product"),
                "category": (product_data.get("カテゴリ") or 
                           product_data.get("category") or 
                           "商品")
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
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info("アフィリエイト商品一覧取得API開始")
        
        if not sheets_service.is_authenticated():
            logger.error("Google Sheets API認証が完了していません")
            raise HTTPException(status_code=400, detail="Google Sheets API認証が必要です")
        
        logger.info(f"スプレッドシート '{sheet_name}' からアフィリエイトデータを取得中...")
        affiliate_data = sheets_service.get_affiliate_data(sheet_name)
        logger.info(f"シート '{sheet_name}' から取得したアフィリエイトデータ件数: {len(affiliate_data)}")
        
        # 商品一覧を整理（新フォーマット対応）
        products = []
        for i, item in enumerate(affiliate_data):
            logger.debug(f"商品 {i+1}: {item}")
            
            # サンプル画像URLを収集
            sample_images = []
            for j in range(1, 21):  # Sample Image URL 1 から 20 まで
                image_key = f"Sample Image URL {j}"
                if image_key in item and item[image_key] and str(item[image_key]).strip():
                    sample_images.append(item[image_key])
            
            product = {
                "content_id": item.get("Content ID"),
                "product_id": item.get("Product ID"),
                "title": item.get("Title"),
                "url": item.get("URL"),
                "affiliate_url": item.get("Affiliate URL"),
                "date": item.get("Date"),
                "description": item.get("Description"),
                "created_at": item.get("Created At"),
                "sample_images": sample_images,
                "sample_images_count": len(sample_images),
                
                # 後方互換性のため（AI生成で使用）
                "id": item.get("Product ID") or item.get("Content ID"),
                "name": item.get("Title"),
                "features": item.get("Description"),
                "affiliate_link": item.get("Affiliate URL"),
                "category": "商品",  # デフォルト値
                "target": "一般",    # デフォルト値
                "keywords": ""       # デフォルト値
            }
            
            # Content IDまたはProduct IDが存在する商品のみ追加
            if product["content_id"] or product["product_id"]:
                products.append(product)
                logger.debug(f"商品追加: {product['product_id']} - {product['title']}")
            else:
                logger.warning(f"Content IDまたはProduct IDが見つからないためスキップ: {item}")
        
        logger.info(f"最終的な商品件数: {len(products)}")
        
        return {
            "products": products,
            "total": len(products),
            "sheet_name": sheet_name
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
        affiliate_data = sheets_service.get_affiliate_data("アフィリ用データ")
        
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
                    # 商品データを検索（新フォーマット対応）
                    product_data = None
                    for item in affiliate_data:
                        if (item.get("Product ID") == product_id or 
                            item.get("Content ID") == product_id or
                            item.get("商品ID") == product_id or 
                            item.get("product_id") == product_id):
                            product_data = item
                            break
                    
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
        
        # 設定を更新
        success = llm_service.update_provider_config(request.provider.lower(), config_data)
        
        if success:
            return {
                "message": f"{request.provider} の設定を更新しました",
                "provider": request.provider,
                "config": llm_service.get_provider_config(request.provider.lower())
            }
        else:
            raise HTTPException(status_code=500, detail="設定の保存に失敗しました")
            
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
        
        success = llm_service.update_provider_config(provider.lower(), default_config)
        
        if success:
            return {
                "message": f"{provider} の設定をリセットしました",
                "provider": provider
            }
        else:
            raise HTTPException(status_code=500, detail="設定のリセットに失敗しました")
            
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