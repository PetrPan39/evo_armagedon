import os
import time
import openai
from typing import Optional

openai.api_key = os.getenv('OPENAI_API_KEY')

class OpenAIClient:
    @staticmethod
    def call_with_retry(**kwargs) -> Optional[dict]:
        max_retries = 5
        backoff = 1
        for i in range(max_retries):
            try:
                return openai.ChatCompletion.create(**kwargs)
            except Exception as e:
                time.sleep(backoff)
                backoff *= 2
        return None
