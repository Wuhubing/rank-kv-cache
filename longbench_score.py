"""LongBench sanity check: score sparse reference and dense generations per document.

Uses LongBench's own metrics (LongBench/metrics.py in THUDM/LongBench): qa_f1_score for
MultiFieldQA-en and rouge_score (ROUGE-L F from the rouge package) for GovReport, the
maximum over reference answers, reported x100. Generations are decoded from the manifests
with skip_special_tokens=True. Needs transformers (tokenizer) and rouge.
"""
import argparse
import csv
import json
import random
import re
import string
from collections import Counter
from pathlib import Path


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction, ground_truth):
    return f1_score(normalize_answer(prediction).split(), normalize_answer(ground_truth).split())


def rouge_score(prediction, ground_truth):
    from rouge import Rouge
    try:
        return Rouge().get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"]
    except Exception:  # LongBench returns 0 when rouge fails, e.g. on an empty prediction
        return 0.0


METRICS = {"multifieldqa_en": qa_f1_score, "gov_report": rouge_score}


def score(task, prediction, answers):
    return max(METRICS[task](prediction, a) for a in answers)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    manifests = [json.loads(p.read_text()) for p in sorted((args.run_dir / "manifests").glob("*.json"))]
    tokenizer = AutoTokenizer.from_pretrained(manifests[0]["model"],
                                              revision=manifests[0]["model_revision"])
    rows = []
    for m in manifests:
        row = dict(request_id=m["request_id"], task=m["task"], band=m["band"])
        for kind, key in (("sparse", "generated_token_ids"), ("dense", "dense_generated_token_ids")):
            if key in m:
                text = tokenizer.decode(m[key], skip_special_tokens=True)
                row[kind] = 100 * score(m["task"], text, m["answers"])
                row[f"{kind}_tokens"] = len(m[key])
        rows.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "longbench_by_doc.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    rng = random.Random(args.seed)
    for task in sorted({r["task"] for r in rows}):
        for band in ["all"] + sorted({r["band"] for r in rows}):
            group = [r for r in rows if r["task"] == task and band in ("all", r["band"])]
            entry = dict(task=task, band=band, documents=len(group))
            for kind in ("sparse", "dense"):
                values = [r[kind] for r in group if kind in r]
                entry[kind] = sum(values) / len(values) if values else None
            paired = [(r["sparse"], r["dense"]) for r in group if "sparse" in r and "dense" in r]
            if paired:
                diffs = []
                for _ in range(args.bootstrap):
                    s = [rng.choice(paired) for _ in paired]
                    diffs.append(sum(a - b for a, b in s) / len(s))
                diffs.sort()
                entry.update(sparse_minus_dense=sum(a - b for a, b in paired) / len(paired),
                             ci_low=diffs[int(.025 * len(diffs))],
                             ci_high=diffs[min(len(diffs) - 1, int(.975 * len(diffs)))])
            summary.append(entry)
    (args.out / "longbench_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for entry in summary:
        print(entry)


if __name__ == "__main__":
    main()
