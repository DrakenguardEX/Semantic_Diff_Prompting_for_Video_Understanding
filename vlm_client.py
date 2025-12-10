import os
import base64
from io import BytesIO
from typing import Optional
from PIL import Image
from openai import OpenAI

class VLMClient:
    def __init__(self, model: str = "gpt-4.1-mini"):
        self.client = OpenAI()
        self.model = model

    @staticmethod
    def pil_to_base64(img: Image.Image) -> str:
        buf = BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def describe_single(self, image: Image.Image, prompt: str, max_tokens: int = 200) -> str:
        img_b64 = self.pil_to_base64(image)
        resp = self.client.chat.completions.create(
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
        return resp.choices[0].message.content

    def describe_pair(self, prev_image: Image.Image, curr_image: Image.Image,
                      prompt: str, max_tokens: int = 200) -> str:
        """take two consecutive frames as input, useful for semantic difference"""
        prev_b64 = self.pil_to_base64(prev_image)
        curr_b64 = self.pil_to_base64(curr_image)

        resp = self.client.chat.completions.create(
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
        return resp.choices[0].message.content

