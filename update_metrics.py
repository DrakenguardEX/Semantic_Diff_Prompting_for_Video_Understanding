"""
Recompute all metrics (tokens, lexical redundancy, info density)
for existing result JSON files, without calling the VLM again.
Also removes obsolete alignment-related fields.

Use when:
- Metric definitions in metrics.py are changed, or
- you want to clean up old JSON results.
"""

import os
import json
from typing import List
from transformers import AutoTokenizer

from metrics import (
    compute_lexical_redundancy,
    information_density_class_aware,
    average_information_density_class_aware,
)

RESULTS_ROOT = "results"

# Remove those old fields in json file
DROP_KEYS = [
    "visual_changes",
    "text_changes_baseline",
    "text_changes_diff",
    "motion_text_alignment_baseline",
    "motion_text_alignment_diff",
]

tokenizer = AutoTokenizer.from_pretrained("gpt2")


def recompute_tokens(texts: List[str]) -> int:
    """Count tokens using the same tokenizer as in run_video_diff.py."""
    summary = "\n".join(texts)
    return len(tokenizer.encode(summary))


def update_one_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    baseline_texts = data.get("baseline_texts", [])
    diff_texts = data.get("diff_texts", [])
    cls = data.get("class", "__default__")

    if not baseline_texts or not diff_texts:
        print(f"[WARN] {path} has no texts, skip.")
        return

    # tokens consumption
    baseline_tokens = recompute_tokens(baseline_texts)
    diff_tokens = recompute_tokens(diff_texts)

    # lexical redundancy
    avg_red_base, red_list_base = compute_lexical_redundancy(baseline_texts)
    avg_red_diff, red_list_diff = compute_lexical_redundancy(diff_texts)

    # information density
    info_base = average_information_density_class_aware(baseline_texts, cls)
    info_diff = average_information_density_class_aware(diff_texts, cls)
    info_base_pf = [information_density_class_aware(t, cls) for t in baseline_texts]
    info_diff_pf = [information_density_class_aware(t, cls) for t in diff_texts]

    data["baseline_tokens"] = baseline_tokens
    data["diff_tokens"] = diff_tokens

    data["lexical_redundancy_baseline_avg"] = avg_red_base
    data["lexical_redundancy_diff_avg"] = avg_red_diff
    data["lexical_redundancy_baseline_all"] = red_list_base
    data["lexical_redundancy_diff_all"] = red_list_diff

    data["info_density_baseline"] = info_base
    data["info_density_diff"] = info_diff
    data["info_density_baseline_per_frame"] = info_base_pf
    data["info_density_diff_per_frame"] = info_diff_pf

    # remove those unnecessary fields
    for key in DROP_KEYS:
        if key in data:
            data.pop(key)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        f"[UPDATED] {path} | tokens({baseline_tokens}->{diff_tokens}), "
        f"lex_red({avg_red_base:.3f}->{avg_red_diff:.3f}), "
        f"info_density({info_base:.4f}->{info_diff:.4f})"
    )


def main():
    count = 0
    for root, _, files in os.walk(RESULTS_ROOT):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            json_path = os.path.join(root, fname)
            update_one_json(json_path)
            count += 1
    print(f"\n[DONE] processed {count} json files.")


if __name__ == "__main__":
    main()
