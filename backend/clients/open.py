"""Muse Spark Open on Baseten (OpenAI-compatible Model APIs)."""
import os

import openai

from backend import config
from backend.clients import ModelError
from backend.clients.meta import TIMEOUT_S, chat_completion


class OpenClient:
    def __init__(self, api_key=None, http_client=None):
        self._key = api_key or os.getenv("BASETEN_API_KEY")
        self._http = http_client
        self._oa = None

    def chat(self, messages, images=(), json_schema=None) -> str:
        if not self._key:
            raise ModelError("BASETEN_API_KEY is not set")
        if self._oa is None:
            self._oa = openai.OpenAI(base_url=config.OPEN_BASE_URL, api_key=self._key, max_retries=0,
                                     timeout=TIMEOUT_S, http_client=self._http)
        return chat_completion(self._oa, config.OPEN_MODEL, "open", messages, images, json_schema)
