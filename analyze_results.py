"""
Aggregate metrics per action class.

For each class we compute:
- #videos
- avg baseline_tokens, avg diff_tokens, and relative token reduction
- avg lexical_redundancy_baseline_avg, avg lexical_redundancy_diff_avg
- avg info_density_baseline, avg info_density_diff

Results are printed to console and also written to analysis_summary.csv
"""

import os
import json
from collections import defaultdict

RESULTS_ROOT = "results"
CSV_PATH = "analysis_summary.csv"


def collect_results():
    # class_name -> list of per-video metric dicts
    by_class = defaultdict(list)

    for root, _, files in os.walk(RESULTS_ROOT):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            path = os.path.join(root, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            cls = data.get("class", "UNKNOWN")
            rec = {
                "video_id": data.get("video_id"),
                "baseline_tokens": data.get("baseline_tokens", 0),
                "diff_tokens": data.get("diff_tokens", 0),
                "lex_red_base": data.get("lexical_redundancy_baseline_avg", 0.0),
                "lex_red_diff": data.get("lexical_redundancy_diff_avg", 0.0),
                "info_base": data.get("info_density_baseline", 0.0),
                "info_diff": data.get("info_density_diff", 0.0),
            }
            by_class[cls].append(rec)

    return by_class


def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0


def analyze(by_class):
    print(
        f"{'Class':45s} | {'N':>3s} | "
        f"{'tok_base':>8s} | {'tok_diff':>8s} | {'tok_red%':>7s} | "
        f"{'lex_base':>8s} | {'lex_diff':>8s} | "
        f"{'info_base':>9s} | {'info_diff':>9s}"
    )
    print("-" * 120)

    rows_for_csv = []
    for cls, recs in sorted(by_class.items(), key=lambda x: x[0].lower()):
        n = len(recs)
        base_tokens = [r["baseline_tokens"] for r in recs]
        diff_tokens = [r["diff_tokens"] for r in recs]
        lex_base = [r["lex_red_base"] for r in recs]
        lex_diff = [r["lex_red_diff"] for r in recs]
        info_base = [r["info_base"] for r in recs]
        info_diff = [r["info_diff"] for r in recs]

        avg_tok_base = avg(base_tokens)
        avg_tok_diff = avg(diff_tokens)
        # let token reduction = 1 - diff/base
        if avg_tok_base > 0:
            tok_reduction = 1.0 - (avg_tok_diff / avg_tok_base)
        else:
            tok_reduction = 0.0

        avg_lex_base = avg(lex_base)
        avg_lex_diff = avg(lex_diff)
        avg_info_base = avg(info_base)
        avg_info_diff = avg(info_diff)

        print(
            f"{cls[:45]:45s} | {n:3d} | "
            f"{avg_tok_base:8.1f} | {avg_tok_diff:8.1f} | {tok_reduction*100:7.1f} | "
            f"{avg_lex_base:8.3f} | {avg_lex_diff:8.3f} | "
            f"{avg_info_base:9.4f} | {avg_info_diff:9.4f}"
        )

        rows_for_csv.append(
            {
                "class": cls,
                "num_videos": n,
                "avg_baseline_tokens": avg_tok_base,
                "avg_diff_tokens": avg_tok_diff,
                "avg_token_reduction": tok_reduction,
                "avg_lexical_redundancy_baseline": avg_lex_base,
                "avg_lexical_redundancy_diff": avg_lex_diff,
                "avg_info_density_baseline": avg_info_base,
                "avg_info_density_diff": avg_info_diff,
            }
        )

    return rows_for_csv


def write_csv(rows, path):
    if not rows:
        print("No rows to write.")
        return

    import csv

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[INFO] Summary written to {path}")


def main():
    by_class = collect_results()
    print(f"[INFO] Found {len(by_class)} classes in {RESULTS_ROOT}")
    rows = analyze(by_class)
    write_csv(rows, CSV_PATH)


if __name__ == "__main__":
    main()
