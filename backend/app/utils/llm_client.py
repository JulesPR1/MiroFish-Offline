"""
LLM Client Wrapper
Unified OpenAI-compatible API calls (LM Studio, Ollama, OpenAI, etc.)
Supports Ollama num_ctx parameter to prevent prompt truncation when using Ollama
"""

import json
import os
import re
import logging
import threading
from collections import OrderedDict
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config

logger = logging.getLogger('mirofish.llm')


class _LRUCache:
    """Thread-safe LRU cache for chat_json responses (structured outputs only)."""

    def __init__(self, maxsize: int = 200):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _make_key(self, messages: List[Dict], temperature: float, max_tokens: int) -> tuple:
        msg_tuple = tuple((m.get("role", ""), m.get("content", "")) for m in messages)
        return (msg_tuple, temperature, max_tokens)

    def get(self, messages: List[Dict], temperature: float, max_tokens: int) -> Optional[Dict]:
        key = self._make_key(messages, temperature, max_tokens)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            self.misses += 1
            return None

    def set(self, messages: List[Dict], temperature: float, max_tokens: int, value: Dict):
        key = self._make_key(messages, temperature, max_tokens)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)


class LLMClient:
    """LLM Client"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 300.0
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY not configured")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

        # Ollama context window size — prevents prompt truncation.
        # Read from env OLLAMA_NUM_CTX, default 8192 (Ollama default is only 2048).
        self._num_ctx = int(os.environ.get('OLLAMA_NUM_CTX', '8192'))

        # Optional LRU cache for chat_json (structured outputs only, disabled by default)
        self._cache: Optional[_LRUCache] = (
            _LRUCache(maxsize=Config.LLM_CACHE_SIZE) if Config.LLM_CACHE_ENABLED else None
        )

    def _is_ollama(self) -> bool:
        """Check if we're talking to an Ollama server."""
        return '11434' in (self.base_url or '')

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 8132,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        Send chat request

        Args:
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Max token count
            response_format: Response format (e.g., JSON mode)

        Returns:
            Model response text
        """
        try:
            logger.debug(f"Chat request: model={self.model}, num_messages={len(messages)}, temp={temperature}, max_tokens={max_tokens}")
            if response_format:
                logger.debug(f"Response format: {response_format}")

            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if response_format:
                kwargs["response_format"] = response_format

            # For Ollama: pass num_ctx via extra_body to prevent prompt truncation
            if self._is_ollama() and self._num_ctx:
                logger.debug(f"Using Ollama-specific settings: num_ctx={self._num_ctx}")
                kwargs["extra_body"] = {
                    "options": {"num_ctx": self._num_ctx}
                }

            logger.debug(f"Sending request to LLM at {self.base_url}...")
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content

            # Some models (like MiniMax M2.5) include <think>thinking content in response, need to remove
            content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
            logger.debug(f"LLM response received, length: {len(content)} characters")
            return content

        except Exception as e:
            logger.error(f"LLM chat failed: {str(e)}")
            logger.error(f"Base URL: {self.base_url}, Model: {self.model}")
            raise

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 8192
    ) -> Dict[str, Any]:
        """
        Send chat request and return JSON

        Args:
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Max token count

        Returns:
            Parsed JSON object
        """
        try:
            logger.debug("Requesting JSON response from LLM...")

            # Cache probe (only for chat_json — deterministic structured outputs)
            if self._cache is not None:
                cached = self._cache.get(messages, temperature, max_tokens)
                if cached is not None:
                    logger.debug(
                        f"chat_json cache HIT (hits={self._cache.hits}, misses={self._cache.misses})"
                    )
                    return cached

            # Note: LM Studio doesn't support json_schema format, so we rely on prompt to guide JSON output
            response = self.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            logger.info(f"Raw LLM response length: {len(response)} chars")
            logger.debug(f"Raw LLM response (first 300 chars): {response[:300]}")
            logger.debug(f"Raw LLM response (last 300 chars): {response[-300:]}")

            # Clean markdown code block markers
            cleaned_response = response.strip()
            logger.debug(f"After strip: {len(cleaned_response)} chars")

            # Remove markdown code block - handle both opening and closing markers
            cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
            logger.debug(f"After removing opening markers: {len(cleaned_response)} chars")

            cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
            logger.debug(f"After removing closing markers: {len(cleaned_response)} chars")

            cleaned_response = cleaned_response.strip()
            logger.debug(f"After final strip: {len(cleaned_response)} chars")

            if cleaned_response != response:
                logger.debug("Removed markdown code block markers from response")

            logger.debug(f"Cleaned response (first 300 chars): {cleaned_response[:300]}")

            if not cleaned_response:
                logger.error(f"Response is empty after cleaning! Raw response was:\n{response}")
                raise ValueError("LLM returned empty response after cleaning")

            try:
                result = json.loads(cleaned_response)
                logger.debug(f"JSON parsing successful, keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
                # Store in cache on successful parse
                if self._cache is not None:
                    self._cache.set(messages, temperature, max_tokens, result)
                return result
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {str(e)}")
                logger.error(f"Full cleaned response:\n{cleaned_response}")
                raise ValueError(f"Invalid JSON format from LLM: {str(e)}")

        except Exception as e:
            logger.error(f"chat_json failed: {str(e)}")
            raise
