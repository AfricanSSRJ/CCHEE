import os
import time


class GPT4oClient:
    def __init__(self, model="gpt-4o", temperature=0.7, max_tokens=512, api_key=None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages, temperature=None, max_tokens=None, retries=3):
        last_error = None
        for _ in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                )
                return response.choices[0].message.content
            except Exception as error:
                last_error = error
                time.sleep(2)
        raise last_error
