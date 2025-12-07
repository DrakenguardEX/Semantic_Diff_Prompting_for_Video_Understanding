import os
from typing import List
from PIL import Image
from vlm_client import VLMClient
from transformers import AutoTokenizer

BASELINE_PROMPT = "Describe this frame."
DIFF_PROMPT = (
    "You are given two consecutive frames from a video. "
    "Describe only what changed in the current frame compared to the previous one. "
    "Do not repeat objects, background, or attributes that stayed the same."
)

def load_images_in_order(folder: str) -> List[Image.Image]:
    """Load a series of frames in orders of file names"""
    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    imgs = []
    for f in files:
        path = os.path.join(folder, f)
        imgs.append(Image.open(path).convert("RGB"))
    print(f"Loaded {len(imgs)} frames from {folder}: {files}")
    return imgs

def run_baseline(vlm: VLMClient, images: List[Image.Image]):
    results = []
    for i, img in enumerate(images):
        text = vlm.describe_single(img, BASELINE_PROMPT)
        results.append({"frame_idx": i, "mode": "baseline", "text": text})
    return results

def run_diff(vlm: VLMClient, images: List[Image.Image]):
    results = []
    # initial frame that has no "previous" frame
    results.append({
        "frame_idx": 0,
        "mode": "diff",
        "text": "Initial frame. No previous frame to compare.",
    })
    for i in range(1, len(images)):
        prev_img = images[i - 1]
        curr_img = images[i]
        text = vlm.describe_pair(prev_img, curr_img, DIFF_PROMPT)
        results.append({"frame_idx": i, "mode": "diff", "text": text})
    return results

def pretty_print_compare(baseline, diff):
    print("\n======= Baseline vs Diff (Semantic Diff Demo) =======")
    for b, d in zip(baseline, diff):
        idx = b["frame_idx"]
        print(f"\n[Frame {idx}]")
        print("Baseline:", b["text"])
        print("Diff    :", d["text"])


#########
# Token Consumption evaluation
def count_total_tokens(texts, model_name="gpt2"):
    # use gpt2 as tokenizer for rough evaluation here, will introduce more metrics later
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return sum(len(tokenizer.encode(t)) for t in texts)
#########

def main():
    vlm = VLMClient(model="gpt-4.1-mini")
    frames_folder = "test_frame_diff"  # name of folder with a consecutive of frames for testing
    images = load_images_in_order(frames_folder)
    
    baseline_results = run_baseline(vlm, images) # baseline video understanding that describe each frame

    diff_results = run_diff(vlm, images) # semantic diff that describe the changes

    pretty_print_compare(baseline_results, diff_results)
    baseline_texts = [x["text"] for x in baseline_results]
    diff_texts = [x["text"] for x in diff_results]

    baseline_tokens = count_total_tokens(baseline_texts)
    diff_tokens = count_total_tokens(diff_texts)

    print("\n======= Rough Token Statistics =======")
    print("Total tokens (baseline):", baseline_tokens)
    print("Total tokens (diff)    :", diff_tokens)

if __name__ == "__main__":
    main()
