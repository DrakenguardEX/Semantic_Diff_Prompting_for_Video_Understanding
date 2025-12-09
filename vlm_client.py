import os
import base64
import time
from io import BytesIO
from typing import Optional
from PIL import Image
from openai import OpenAI, RateLimitError, APIError

class VLMClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            env_key = os.environ.get("OPENAI_API_KEY")
            if not env_key or env_key.startswith("your-api"):
                raise ValueError(
                    "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
                    "or pass api_key parameter. Get your key at: https://platform.openai.com/account/api-keys"
                )
            self.client = OpenAI(api_key=env_key)
        self.model = model

    @staticmethod
    def pil_to_base64(img: Image.Image) -> str:
        buf = BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def safe_openai_call(self, fn, retries=5):
        for i in range(retries):
            try:
                return fn()
            except RateLimitError as e:
                wait_time = 2 ** i
                print(f"⚠️ Rate limit hit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            except APIError as e:
                print(f"⚠️ OpenAI API error: {e}. Retrying in 5s...")
                time.sleep(5)
        raise RuntimeError("❌ Failed after multiple retries due to rate limits or API errors.")

    def describe_single(self, image: Image.Image, prompt: str, max_tokens: int = 200) -> str:
        img_b64 = self.pil_to_base64(image)

        def call():
            return self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                            },
                        ],
                    }
                ],
                max_tokens=max_tokens,
            )

        resp = self.safe_openai_call(call)
        return resp.choices[0].message.content

    def describe_pair(self, prev_image: Image.Image, curr_image: Image.Image,
                      prompt: str, max_tokens: int = 200) -> str:
        prev_b64 = self.pil_to_base64(prev_image)
        curr_b64 = self.pil_to_base64(curr_image)

        def call():
            return self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{prev_b64}"},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{curr_b64}"},
                            },
                        ],
                    }
                ],
                max_tokens=max_tokens,
            )

        resp = self.safe_openai_call(call)
        return resp.choices[0].message.content
