import os
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def main():
    img_path = "test_img1.jpg"
    img = Image.open(img_path).convert("RGB")
    img_b64 = pil_to_base64(img)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one or two sentences."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                ],
            }
        ],
        max_tokens=200,
    )

    print(resp.choices[0].message.content)

if __name__ == "__main__":
    main()
