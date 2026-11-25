"""Freeze the LongBench sample: 60 prompts split 20 dev / 40 held-out by source document.

Cells are task x length band, where length is the number of Qwen3 tokens in the full,
uncropped prompt. Each source document (identical context) contributes at most one
prompt, its first row in file order, so no document can appear in both splits. Within a
cell, candidates are shuffled with a seed derived from SEED, the task and the band; the
pilot document is moved to the front of its cell so it lands in dev. The first 15 form
the cell, the first 5 of those are dev and the rest held-out. dev_collection lists the
first two dev prompts per cell, in that order, for the GPU pilot.

Run once and commit the output. Never collect or inspect held-out prompts while tuning.
Needs transformers and huggingface_hub (tokenizer only, no PyTorch).
"""
import argparse
import json
import random
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REV = "c1899de289a04d12100db370d81485cdf75e47ca"
DATA_ID = "zai-org/LongBench"
DATA_REV = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
SEED = 6594010
PER_CELL, DEV_PER_CELL, COLLECT_PER_CELL = 15, 5, 2
BANDS = {"4-8k": (4096, 8192), "8-16k": (8192, 16384)}  # prompt tokens, half-open
PILOT = ("gov_report", "4bd6930b3b65caf89aa127d679a80294f31f9cfa7cdab703")
# GovReport keeps the pilot's wording. MultiFieldQA-en is LongBench's own template
# (LongBench/config/dataset2prompt.json in the THUDM/LongBench repository).
TEMPLATES = {
    "multifieldqa_en": ("Read the following text and answer briefly.\n\n{context}\n\nNow, answer "
                        "the following question based on the above text, only give me the "
                        "answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"),
    "gov_report": ("Government report excerpt:\n\n{context}\n\nWrite a one-paragraph summary of "
                   "this report.\nSummary:"),
}


def band_of(tokens):
    for name, (low, high) in BANDS.items():
        if low <= tokens < high:
            return name
    return None


def build(tokenizer, archive):
    cells = defaultdict(list)
    with ZipFile(archive) as z:
        for task, template in TEMPLATES.items():
            member, = [n for n in z.namelist() if Path(n).name == f"{task}.jsonl"]
            seen = set()
            with z.open(member) as data:
                for index, line in enumerate(data):
                    row = json.loads(line)
                    digest = sha256(row["context"].encode()).hexdigest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    prompt = template.format(context=row["context"], input=row["input"])
                    tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                    band = band_of(tokens)
                    if band:
                        cells[task, band].append(dict(
                            task=task, band=band, _id=row["_id"], row_index=index,
                            prompt_tokens=tokens, context_sha256=digest))
    prompts, collection = [], []
    for task in TEMPLATES:
        for band in BANDS:
            pool = sorted(cells[task, band], key=lambda r: r["_id"])
            random.Random(f"{SEED}/{task}/{band}").shuffle(pool)
            pool.sort(key=lambda r: (r["task"], r["_id"]) != PILOT)  # stable: pilot first
            if len(pool) < PER_CELL:
                raise RuntimeError(f"{task} {band}: only {len(pool)} documents")
            for order, item in enumerate(pool[:PER_CELL]):
                item.update(order=order, split="dev" if order < DEV_PER_CELL else "heldout",
                            request_id=f"{task}/{band}/{item['_id']}")
                prompts.append(item)
                if order < COLLECT_PER_CELL:
                    collection.append(item["request_id"])
    return dict(
        description=__doc__.split("\n\n")[0], seed=SEED, model=MODEL_ID, model_revision=MODEL_REV,
        dataset=DATA_ID, data_revision=DATA_REV, bands=BANDS, templates=TEMPLATES,
        per_cell=PER_CELL, dev_per_cell=DEV_PER_CELL,
        candidates={f"{t}/{b}": len(v) for (t, b), v in sorted(cells.items())},
        dev_collection=collection, prompts=prompts)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("splits/longbench_v1.json"))
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    archive = hf_hub_download(DATA_ID, "data.zip", repo_type="dataset", revision=DATA_REV)
    result = build(tokenizer, archive)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}: {len(result['prompts'])} prompts. Candidates per cell: "
          f"{result['candidates']}")
    print("Dev collection:", *result["dev_collection"], sep="\n  ")


if __name__ == "__main__":
    main()
