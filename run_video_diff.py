import os
import json
from typing import List
from PIL import Image
from transformers import AutoTokenizer
from vlm_client import VLMClient
from metrics import compute_lexical_redundancy

FRAMES_ROOT = "frames"
RESULTS_ROOT = "results"

BASELINE_PROMPT = "Describe this frame."
DIFF_PROMPT = (
    "You are given two consecutive frames from a video. "
    "Describe only what changed in the current frame compared to the previous one. "
    "Focus on the main object and its motion. "
    "Do not repeat static background, lighting, or objects that stay the same."
)

tokenizer = AutoTokenizer.from_pretrained("gpt2")


def list_all_videos_from_frames(frames_root: str):
    """
    let available videos from the frames/ directory structure be:
        frames/
            folding/
                folding_001/
                folding_002/
            closing/
                losing_001/

    Return a list of dicts:
        [
            {
                "class": "folding",
                "video_id": "folding_001",
                "frame_dir": "frames/folding/folding_001"
            },
            ...
        ]
    """
    videos = []
    for cls in os.listdir(frames_root):
        cls_dir = os.path.join(frames_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for vid in os.listdir(cls_dir):
            frame_dir = os.path.join(cls_dir, vid)
            if not os.path.isdir(frame_dir):
                continue
            videos.append({
                "class": cls,
                "video_id": vid,
                "frame_dir": frame_dir,
            })
    return videos


def load_images_in_order(frame_dir: str) -> (List[Image.Image], List[str]):
    files = sorted(
        f for f in os.listdir(frame_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    images = []
    for f in files:
        path = os.path.join(frame_dir, f)
        images.append(Image.open(path).convert("RGB"))
    return images, files


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def process_one_video(meta: dict, max_frames: int = 8):
    cls = meta["class"]
    vid = meta["video_id"]
    frame_dir = meta["frame_dir"]

    print(f"\n=== Processing {cls} / {vid} ===")

    # If the result already exists, optionally skip to avoid rerunning after interruptions
    out_dir = os.path.join(RESULTS_ROOT, cls)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{vid}.json")
    if os.path.exists(out_path):
        print(f"[SKIP] Result already exists: {out_path}")
        return

    # load frames
    images, frame_files = load_images_in_order(frame_dir)
    if len(images) == 0:
        print(f"[WARN] No frames in {frame_dir}, skip.")
        return

    # Optionally limit to the first max_frames frames to avoid processing overly long sequences
    if len(images) > max_frames:
        images = images[:max_frames]
        frame_files = frame_files[:max_frames]

    print(f"[INFO] Loaded {len(images)} frames.")

    vlm = VLMClient(model="gpt-4.1-mini") # initial the model

    # Baseline: describe each frame independently
    baseline_texts: List[str] = []
    for i, img in enumerate(images):
        text = vlm.describe_single(img, BASELINE_PROMPT)
        baseline_texts.append(text)
        print(f"[VLM] baseline frame {i} done")

    # Diff: starting from frame 1, describe only the changes
    diff_texts: List[str] = ["Initial frame. No previous frame to compare."]
    for i in range(1, len(images)):
        prev_img = images[i - 1]
        curr_img = images[i]
        text = vlm.describe_pair(prev_img, curr_img, DIFF_PROMPT)
        diff_texts.append(text)
        print(f"[VLM] diff frame {i} done")

    # Token statistics (compute on the entire summary)
    baseline_summary = "\n".join(baseline_texts)
    diff_summary = "\n".join(diff_texts)
    baseline_tokens = count_tokens(baseline_summary)
    diff_tokens = count_tokens(diff_summary)

    # Lexical Redundancy
    avg_red_base, red_list_base = compute_lexical_redundancy(baseline_texts)
    avg_red_diff, red_list_diff = compute_lexical_redundancy(diff_texts)

    print(f"[METRIC] Lexical redundancy baseline avg: {avg_red_base:.3f}")
    print(f"[METRIC] Lexical redundancy diff     avg: {avg_red_diff:.3f}")
    print(f"[METRIC] Tokens baseline: {baseline_tokens}, diff: {diff_tokens}")

    # save result to JSON
    result = {
        "video_id": vid,
        "class": cls,
        "num_frames": len(images),
        "frame_dir": frame_dir,
        "frame_files": frame_files,
        "baseline_texts": baseline_texts,
        "diff_texts": diff_texts,
        "baseline_tokens": baseline_tokens,
        "diff_tokens": diff_tokens,
        "lexical_redundancy_baseline_avg": avg_red_base,
        "lexical_redundancy_diff_avg": avg_red_diff,
        "lexical_redundancy_baseline_all": red_list_base,
        "lexical_redundancy_diff_all": red_list_diff,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved result to {out_path}")


def main():
    videos = list_all_videos_from_frames(FRAMES_ROOT)
    print(f"[INFO] Found {len(videos)} videos from frames.")

    for meta in videos:
        process_one_video(meta, max_frames=8)


if __name__ == "__main__":
    main()






