"""LLM API連携サービス"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import json
import aiohttp
import asyncio

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """LLMプロバイダー"""
    OPENAI = "openai"
    OLLAMA = "ollama"
    CLAUDE = "claude"
    GEMINI = "gemini"


class LLMService:
    """LLM API連携サービス"""
    
    def __init__(self):
        # 設定をJSON形式で保存するファイル
        self.config_file = os.path.join(os.path.dirname(__file__), "..", "..", "llm_config.json")
        self.config = self.load_config()
        
        # 環境変数からも読み込み（後方互換性）
        if not self.config.get('openai', {}).get('api_key'):
            if os.environ.get('OPENAI_API_KEY'):
                self.config.setdefault('openai', {})['api_key'] = os.environ.get('OPENAI_API_KEY')
        
        if not self.config.get('claude', {}).get('api_key'):
            if os.environ.get('CLAUDE_API_KEY'):
                self.config.setdefault('claude', {})['api_key'] = os.environ.get('CLAUDE_API_KEY')
        
        if not self.config.get('gemini', {}).get('api_key'):
            if os.environ.get('GEMINI_API_KEY'):
                self.config.setdefault('gemini', {})['api_key'] = os.environ.get('GEMINI_API_KEY')
        
        if not self.config.get('ollama', {}).get('base_url'):
            self.config.setdefault('ollama', {})['base_url'] = os.environ.get('OLLAMA_ENDPOINT', 'http://localhost:11434')
        
        # デフォルト設定
        self.default_provider = LLMProvider.OLLAMA  # Ollamaを優先に変更
        self.default_models = {
            LLMProvider.OPENAI: "gpt-4",
            LLMProvider.OLLAMA: "llama2",  # Ollamaデフォルトモデル
            LLMProvider.CLAUDE: "claude-3-sonnet-20240229",
            LLMProvider.GEMINI: "gemini-pro"
        }
        
        # カスタムモデル設定
        self.custom_models = self.config.get('custom_models', {})
        
        self.max_retries = 3
        self.timeout = 30
    
    def load_config(self) -> dict:
        """設定ファイルを読み込み"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"設定ファイル読み込みエラー: {e}")
        
        # デフォルト設定
        return {
            "openai": {
                "api_key": "",
                "base_url": "",
                "model": "gpt-4"
            },
            "claude": {
                "api_key": "",
                "base_url": "",
                "model": "claude-3-sonnet-20240229"
            },
            "gemini": {
                "api_key": "",
                "base_url": "",
                "model": "gemini-pro"
            },
            "ollama": {
                "api_key": "",
                "base_url": "http://localhost:11434",
                "model": "llama2"
            }
        }
    
    def save_config(self) -> bool:
        """設定ファイルを保存"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"設定ファイル保存エラー: {e}")
            return False
    
    def update_provider_config(self, provider: str, config: dict) -> bool:
        """プロバイダー設定を更新"""
        try:
            if provider not in self.config:
                self.config[provider] = {}
            
            self.config[provider].update(config)
            return self.save_config()
        except Exception as e:
            logger.error(f"プロバイダー設定更新エラー: {e}")
            return False
    
    def get_provider_config(self, provider: str) -> dict:
        """プロバイダー設定を取得"""
        return self.config.get(provider, {})
    
    def get_all_configs(self) -> dict:
        """全プロバイダー設定を取得"""
        return self.config.copy()
    
    async def generate_content(
        self,
        prompt: str,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        LLMを使用してコンテンツを生成
        
        Args:
            prompt: 生成プロンプト
            provider: LLMプロバイダー
            model: 使用するモデル
            max_tokens: 最大トークン数
            temperature: 生成温度
            system_prompt: システムプロンプト
        
        Returns:
            Dict: 生成結果
        """
        provider = provider or self.default_provider
        provider_config = self.get_provider_config(provider.value)
        model = model or provider_config.get('model') or self.default_models[provider]
        
        try:
            if provider == LLMProvider.OPENAI:
                return await self._generate_openai(
                    prompt, model, max_tokens, temperature, system_prompt
                )
            elif provider == LLMProvider.OLLAMA:
                return await self._generate_ollama(
                    prompt, model, max_tokens, temperature, system_prompt
                )
            elif provider == LLMProvider.CLAUDE:
                return await self._generate_claude(
                    prompt, model, max_tokens, temperature, system_prompt
                )
            elif provider == LLMProvider.GEMINI:
                return await self._generate_gemini(
                    prompt, model, max_tokens, temperature, system_prompt
                )
            else:
                raise ValueError(f"サポートされていないプロバイダー: {provider}")
                
        except Exception as e:
            logger.error(f"LLM生成エラー ({provider.value}): {e}")
            return {
                "success": False,
                "error": str(e),
                "content": None,
                "provider": provider.value,
                "model": model
            }
    
    async def _generate_openai(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """OpenAI APIでコンテンツ生成"""
        config = self.get_provider_config('openai')
        api_key = config.get('api_key')
        base_url = config.get('base_url')
        
        if not api_key:
            raise Exception("OpenAI API keyが設定されていません")
        
        try:
            import openai
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            
            client = openai.AsyncOpenAI(**client_kwargs)
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            content = response.choices[0].message.content
            
            return {
                "success": True,
                "content": content,
                "provider": "openai",
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            raise Exception(f"OpenAI API エラー: {e}")
    
    async def _generate_ollama(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Ollama APIでコンテンツ生成"""
        config = self.get_provider_config('ollama')
        base_url = config.get('base_url', 'http://localhost:11434')
        
        logger.info(f"=== Ollama生成開始 ===")
        logger.info(f"Base URL: {base_url}")
        logger.info(f"Model: {model}")
        logger.info(f"Prompt length: {len(prompt)}")
        logger.info(f"System prompt length: {len(system_prompt) if system_prompt else 0}")
        
        try:
            # Ollamaの場合、システムプロンプトをプロンプトに統合
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            logger.info(f"Full prompt length: {len(full_prompt)}")
            logger.info(f"Max tokens: {max_tokens}, Temperature: {temperature}")
            
            payload = {
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
            
            logger.info(f"Ollama API呼び出し開始: {base_url}/api/generate")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                async with session.post(
                    f"{base_url}/api/generate",
                    json=payload
                ) as response:
                    logger.info(f"Ollama APIレスポンス: status={response.status}")
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama APIエラー詳細: {error_text}")
                        raise Exception(f"Ollama API エラー: HTTP {response.status} - {error_text}")
                    
                    result = await response.json()
                    logger.info(f"Ollama生成結果: {result}")
                    
                    generated_content = result.get("response", "")
                    logger.info(f"生成されたコンテンツ: '{generated_content}' (長さ: {len(generated_content)})")
                    
                    if not generated_content:
                        logger.warning("Ollamaから空のレスポンスが返されました")
                        return {
                            "success": False,
                            "error": "Ollamaから空のレスポンスが返されました",
                            "content": None,
                            "provider": "ollama",
                            "model": model
                        }
                    
                    return {
                        "success": True,
                        "content": generated_content,
                        "provider": "ollama",
                        "model": model,
                        "usage": {
                            "eval_count": result.get("eval_count", 0),
                            "eval_duration": result.get("eval_duration", 0)
                        }
                    }
                    
        except Exception as e:
            logger.error(f"=== Ollama生成エラー ===")
            logger.error(f"エラー詳細: {str(e)}")
            logger.error(f"エラータイプ: {type(e).__name__}")
            raise Exception(f"Ollama API エラー: {e}")
    
    async def _generate_claude(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Claude APIでコンテンツ生成"""
        config = self.get_provider_config('claude')
        api_key = config.get('api_key')
        base_url = config.get('base_url', 'https://api.anthropic.com')
        
        if not api_key:
            raise Exception("Claude API keyが設定されていません")
        
        try:
            headers = {
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Claude API エラー: {response.status}")
                    
                    result = await response.json()
                    content = result["content"][0]["text"]
                    
                    return {
                        "success": True,
                        "content": content,
                        "provider": "claude",
                        "model": model,
                        "usage": {
                            "input_tokens": result["usage"]["input_tokens"],
                            "output_tokens": result["usage"]["output_tokens"]
                        }
                    }
                    
        except Exception as e:
            raise Exception(f"Claude API エラー: {e}")
    
    async def _generate_gemini(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Gemini APIでコンテンツ生成"""
        config = self.get_provider_config('gemini')
        api_key = config.get('api_key')
        base_url = config.get('base_url', 'https://generativelanguage.googleapis.com')
        
        if not api_key:
            raise Exception("Gemini API keyが設定されていません")
        
        try:
            # Geminiの場合、システムプロンプトをプロンプトに統合
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": full_prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens
                }
            }
            
            # APIキーをクエリパラメータに含める
            url = f"{base_url}/v1beta/models/{model}:generateContent?key={api_key}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Gemini API エラー: {response.status} - {error_text}")
                    
                    result = await response.json()
                    
                    # レスポンスからコンテンツを抽出
                    content = ""
                    if "candidates" in result and len(result["candidates"]) > 0:
                        candidate = result["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            parts = candidate["content"]["parts"]
                            content = "".join([part.get("text", "") for part in parts])
                    
                    return {
                        "success": True,
                        "content": content,
                        "provider": "gemini",
                        "model": model,
                        "usage": result.get("usageMetadata", {})
                    }
                    
        except Exception as e:
            raise Exception(f"Gemini API エラー: {e}")
    
    async def generate_social_media_post(
        self,
        product_data: Dict[str, Any],
        post_type: str = "promotional",
        target_audience: str = "general",
        tone: str = "friendly",
        include_hashtags: bool = True,
        max_length: int = 280,
        provider: Optional[LLMProvider] = None
    ) -> Dict[str, Any]:
        """
        商品データからソーシャルメディア投稿を生成
        
        Args:
            product_data: 商品データ
            post_type: 投稿タイプ（promotional, informational, engaging）
            target_audience: ターゲット層
            tone: トーン（friendly, professional, casual, exciting）
            include_hashtags: ハッシュタグを含めるか
            max_length: 最大文字数
            provider: LLMプロバイダー
        
        Returns:
            Dict: 生成された投稿内容
        """
        logger.info("=== ソーシャルメディア投稿生成開始 ===")
        logger.info(f"商品データキー: {list(product_data.keys())}")
        logger.info(f"要求されたプロバイダー: {provider.value if provider else 'デフォルト'}")
        logger.info(f"投稿設定: type={post_type}, tone={tone}, length={max_length}")
        # システムプロンプトを構築
        system_prompt = f"""
あなたはソーシャルメディア投稿の専門家です。
商品情報を元に、効果的で魅力的な{post_type}投稿を作成してください。

投稿要件:
- 文字数: {max_length}文字以内
- トーン: {tone}
- ターゲット: {target_audience}
- ハッシュタグ: {"含める" if include_hashtags else "含めない"}
- 自然で読みやすい日本語
- 商品の魅力を的確に伝える
- アクションを促す内容
"""
        
        # 商品データからプロンプトを構築
        product_info = []
        for key, value in product_data.items():
            if value and str(value).strip():
                product_info.append(f"{key}: {value}")
        
        prompt = f"""
以下の商品情報を元に、ソーシャルメディア投稿を作成してください:

{chr(10).join(product_info)}

投稿タイプ: {post_type}
ターゲット層: {target_audience}
トーン: {tone}

効果的で魅力的な投稿文を作成してください。
"""
        
        logger.info(f"システムプロンプト長: {len(system_prompt)}")
        logger.info(f"ユーザープロンプト長: {len(prompt)}")
        logger.info(f"使用予定プロバイダー: {provider.value if provider else self.default_provider.value}")
        
        # Ollamaの場合はより適切なモデルを自動選択
        actual_provider = provider or self.default_provider
        if actual_provider == LLMProvider.OLLAMA:
            logger.info("Ollama生成のため、利用可能なモデルを確認中...")
            available_models = await self.get_ollama_models()
            logger.info(f"利用可能なOllamaモデル: {available_models}")
            
            # より良いモデルがあれば使用
            preferred_models = ["gemma3:27b", "gemma3", "llama3", "llama2:13b", "llama2"]
            selected_model = None
            
            for preferred in preferred_models:
                for available in available_models:
                    if preferred in available:
                        selected_model = available
                        break
                if selected_model:
                    break
            
            if not selected_model and available_models:
                selected_model = available_models[0]
            
            if selected_model:
                logger.info(f"選択されたOllamaモデル: {selected_model}")
                # モデルを一時的に設定
                temp_config = self.get_provider_config('ollama').copy()
                temp_config['model'] = selected_model
                self.config['ollama']['model'] = selected_model
        
        try:
            result = await self.generate_content(
                prompt=prompt,
                system_prompt=system_prompt,
                provider=provider,
                max_tokens=max_length // 2,  # 文字数制限を考慮
                temperature=0.8  # 創造性を高める
            )
            
            logger.info("=== ソーシャルメディア投稿生成完了 ===")
            logger.info(f"生成結果: {result}")
            return result
            
        except Exception as e:
            logger.error(f"=== ソーシャルメディア投稿生成エラー ===")
            logger.error(f"エラー: {e}")
            logger.error(f"エラータイプ: {type(e).__name__}")
            
            return {
                "success": False,
                "error": str(e),
                "content": None,
                "provider": provider.value if provider else self.default_provider.value
            }
    
    async def generate_thread_posts(
        self,
        topic: str,
        num_posts: int = 3,
        max_length_per_post: int = 280,
        provider: Optional[LLMProvider] = None
    ) -> Dict[str, Any]:
        """
        トピックからスレッド投稿を生成
        
        Args:
            topic: トピック
            num_posts: 投稿数
            max_length_per_post: 投稿ごとの最大文字数
            provider: LLMプロバイダー
        
        Returns:
            Dict: 生成された投稿スレッド
        """
        system_prompt = f"""
あなたはソーシャルメディアコンテンツの専門家です。
与えられたトピックについて、{num_posts}個の連続した投稿を作成してください。

要件:
- 各投稿は{max_length_per_post}文字以内
- 論理的な流れで構成
- 各投稿が独立して読めるが、全体で一つのストーリーを形成
- 読者の関心を引く内容
- 日本語で自然な表現
"""
        
        prompt = f"""
以下のトピックについて、{num_posts}個の投稿からなるスレッドを作成してください:

トピック: {topic}

各投稿を「投稿1:」「投稿2:」のように番号を付けて分けて出力してください。
"""
        
        result = await self.generate_content(
            prompt=prompt,
            system_prompt=system_prompt,
            provider=provider,
            max_tokens=num_posts * max_length_per_post // 2,
            temperature=0.7
        )
        
        if result["success"]:
            # 生成されたコンテンツを投稿ごとに分割
            content = result["content"]
            posts = []
            
            import re
            post_pattern = r"投稿(\d+):\s*(.*?)(?=投稿\d+:|$)"
            matches = re.findall(post_pattern, content, re.DOTALL)
            
            for match in matches:
                post_content = match[1].strip()
                if post_content:
                    posts.append(post_content)
            
            result["posts"] = posts
            result["thread_count"] = len(posts)
        
        return result
    
    async def test_connection(self, provider: LLMProvider) -> Dict[str, Any]:
        """LLM接続テスト"""
        try:
            if provider == LLMProvider.OLLAMA:
                return await self._test_ollama_connection()
            else:
                # 他のプロバイダーの場合は通常のテスト
                test_prompt = "Hello, this is a test message. Please respond with 'Connection successful!'"
                
                result = await self.generate_content(
                    prompt=test_prompt,
                    provider=provider,
                    max_tokens=50,
                    temperature=0.1
                )
                
                return {
                    "provider": provider.value,
                    "connected": result["success"],
                    "response": result.get("content", ""),
                    "error": result.get("error")
                }
            
        except Exception as e:
            return {
                "provider": provider.value,
                "connected": False,
                "response": None,
                "error": str(e)
            }
    
    async def _test_ollama_connection(self) -> Dict[str, Any]:
        """Ollama専用接続テスト"""
        config = self.get_provider_config('ollama')
        base_url = config.get('base_url', 'http://localhost:11434')
        
        try:
            # まずOllamaサーバーが起動しているかチェック
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{base_url}/api/tags") as response:
                    if response.status == 200:
                        result = await response.json()
                        models = result.get('models', [])
                        
                        if models:
                            # モデルが1つでもあれば簡単なテストを実行
                            first_model = models[0]['name']
                            test_result = await self._test_ollama_generate(base_url, first_model)
                            
                            return {
                                "provider": "ollama",
                                "connected": test_result["success"],
                                "response": test_result.get("response", f"利用可能モデル: {len(models)} 個"),
                                "error": test_result.get("error"),
                                "available_models": len(models),
                                "test_model": first_model if test_result["success"] else None
                            }
                        else:
                            return {
                                "provider": "ollama",
                                "connected": True,
                                "response": "Ollamaサーバーに接続成功（モデルなし）",
                                "error": None,
                                "available_models": 0
                            }
                    else:
                        return {
                            "provider": "ollama",
                            "connected": False,
                            "response": None,
                            "error": f"Ollamaサーバーエラー: HTTP {response.status}"
                        }
                        
        except Exception as e:
            return {
                "provider": "ollama",
                "connected": False,
                "response": None,
                "error": f"Ollama接続エラー: {str(e)}"
            }
    
    async def _test_ollama_generate(self, base_url: str, model: str) -> Dict[str, Any]:
        """Ollama生成テスト"""
        try:
            payload = {
                "model": model,
                "prompt": "Test",
                "stream": False,
                "options": {
                    "num_predict": 10,
                    "temperature": 0.1
                }
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(f"{base_url}/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "response": f"生成テスト成功 ({model})",
                            "content": result.get("response", "")
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"生成テスト失敗: HTTP {response.status}"
                        }
        except Exception as e:
            return {
                "success": False,
                "error": f"生成テストエラー: {str(e)}"
            }
    
    def get_available_providers(self) -> List[Dict[str, Any]]:
        """利用可能なプロバイダー一覧を取得"""
        providers = []
        
        for provider in LLMProvider:
            config = self.get_provider_config(provider.value)
            
            status = {
                "name": provider.value,
                "display_name": provider.value.upper(),
                "available": False,
                "models": [],
                "config": {
                    "api_key_configured": bool(config.get('api_key')),
                    "base_url": config.get('base_url', ''),
                    "model": config.get('model', '')
                }
            }
            
            if provider == LLMProvider.OPENAI:
                status["available"] = bool(config.get('api_key'))
                status["models"] = self.get_all_models('openai')
            elif provider == LLMProvider.CLAUDE:
                status["available"] = bool(config.get('api_key'))
                status["models"] = self.get_all_models('claude')
            elif provider == LLMProvider.GEMINI:
                status["available"] = bool(config.get('api_key'))
                status["models"] = self.get_all_models('gemini')
            elif provider == LLMProvider.OLLAMA:
                status["available"] = True  # Ollamaは常に利用可能と仮定
                status["models"] = self.get_all_models('ollama')
            
            providers.append(status)
        
        return providers
    
    async def get_ollama_models(self) -> List[str]:
        """Ollamaからインストール済みモデル一覧を取得"""
        config = self.get_provider_config('ollama')
        base_url = config.get('base_url', 'http://localhost:11434')
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f"{base_url}/api/tags") as response:
                    if response.status == 200:
                        result = await response.json()
                        models = []
                        for model in result.get('models', []):
                            name = model.get('name', '')
                            if name:
                                models.append(name)
                        return sorted(models)
                    else:
                        logger.warning(f"Ollama API エラー: {response.status}")
                        return []
        except Exception as e:
            logger.warning(f"Ollama モデル取得エラー: {e}")
            return []
    
    def add_custom_model(self, provider: str, model_name: str) -> bool:
        """カスタムモデルを追加"""
        try:
            if provider not in self.custom_models:
                self.custom_models[provider] = []
            
            if model_name not in self.custom_models[provider]:
                self.custom_models[provider].append(model_name)
                
            # 設定ファイルに保存
            self.config['custom_models'] = self.custom_models
            return self.save_config()
        except Exception as e:
            logger.error(f"カスタムモデル追加エラー: {e}")
            return False
    
    def remove_custom_model(self, provider: str, model_name: str) -> bool:
        """カスタムモデルを削除"""
        try:
            if provider in self.custom_models and model_name in self.custom_models[provider]:
                self.custom_models[provider].remove(model_name)
                
                # 空のリストは削除
                if not self.custom_models[provider]:
                    del self.custom_models[provider]
                
                # 設定ファイルに保存
                self.config['custom_models'] = self.custom_models
                return self.save_config()
            return True
        except Exception as e:
            logger.error(f"カスタムモデル削除エラー: {e}")
            return False
    
    def get_all_models(self, provider: str) -> List[str]:
        """プロバイダーの全モデル（デフォルト + カスタム）を取得"""
        base_models = []
        
        if provider == 'openai':
            base_models = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4o", "gpt-4o-mini"]
        elif provider == 'claude':
            base_models = ["claude-3-5-sonnet-20241022", "claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
        elif provider == 'gemini':
            base_models = ["gemini-pro", "gemini-pro-vision", "gemini-1.5-pro", "gemini-1.5-flash"]
        elif provider == 'ollama':
            base_models = ["llama2", "llama3", "mistral", "codellama", "llama2:13b", "gemma"]
        
        # カスタムモデルを追加
        custom_models = self.custom_models.get(provider, [])
        all_models = base_models + custom_models
        
        # 重複を除去して返す
        return list(dict.fromkeys(all_models))


# シングルトンインスタンス
llm_service = LLMService()