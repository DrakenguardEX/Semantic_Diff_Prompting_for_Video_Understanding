# run_video_diff.py
import os
from PIL import Image
from vlm_client import VLMClient
from extract_frames import extract_frames
from transformers import AutoTokenizer

BASELINE_PROMPT = "Describe this frame."
DIFF_PROMPT = (
    "You are given two consecutive frames from a video. "
    "Describe only what changed in the current frame compared to the previous one. "
    "Do not repeat static background unless it changes."
)

def load_images(folder):
    files = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))
    return [Image.open(os.path.join(folder, f)).convert("RGB") for f in files]

def count_tokens(texts, model_name="gpt2"):
    tok = AutoTokenizer.from_pretrained(model_name)
    return sum(len(tok.encode(t)) for t in texts)

def main():
    video_path = "data/video1.mp4"
    frame_dir = "data/video1_frames"

    # 1) 拆帧
    extract_frames(video_path, frame_dir, max_frames=8)

    # 2) 加载帧
    images = load_images(frame_dir)

    # 3) 初始化 VLM
    vlm = VLMClient(model="gpt-4.1-mini")

    # 4) baseline
    baseline_texts = []
    for img in images:
        baseline_texts.append(vlm.describe_single(img, BASELINE_PROMPT))

    # 5) diff
    diff_texts = ["Initial frame. No previous frame to compare."]
    for i in range(1, len(images)):
        diff_texts.append(
            vlm.describe_pair(images[i-1], images[i], DIFF_PROMPT)
        )

    # 6) “视频摘要”——最简单版本：直接把整段文本按时间拼起来
    baseline_summary = "\n".join(
        f"Frame {i}: {t}" for i, t in enumerate(baseline_texts)
    )
    diff_summary = "\n".join(
        f"Frame {i}: {t}" for i, t in enumerate(diff_texts)
    )

    # 7) token 统计
    baseline_tokens = count_tokens([baseline_summary])
    diff_tokens = count_tokens([diff_summary])

    print("=== Baseline Summary ===")
    print(baseline_summary)
    print("\n=== Diff Summary ===")
    print(diff_summary)

    print("\n=== Token Comparison ===")
    print("Baseline summary tokens:", baseline_tokens)
    print("Diff summary tokens    :", diff_tokens)

if __name__ == "__main__":
    main()

