#!/bin/bash
# followup-v3 B4: re-collect the 14 0.6B dev documents with the dense attention mass of every page (data/dev_attn_all),
# then check that ranked / scores / generated tokens are bit-identical to data/dev.
set -e
cd /data/weibing/projects/rkv-v3
export TMPDIR=/data/weibing/tmp HF_HOME=/data/weibing/cache/huggingface
GPU=$(free-gpus 1)
[ -z "$GPU" ] && { echo "no free GPU"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPU
echo "[$(date -u -Is)] B4 collect on GPU $GPU $(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader)"
.venv/bin/python -m collect.long --split splits/longctx_v1.json --subset dev --out data/dev_attn_all --record-attn-all
echo "[$(date -u -Is)] identity check"
.venv/bin/python - <<'PY'
import glob, json, numpy as np, os
bad = 0
for pa in sorted(glob.glob("data/dev_attn_all/*.npz")):
    pb = "data/dev/" + os.path.basename(pa)
    za, zb = np.load(pa), np.load(pb)
    ok = np.array_equal(za["ranked"], zb["ranked"]) and np.array_equal(za["scores"], zb["scores"], equal_nan=True)
    ok = ok and json.load(open(pa[:-4] + ".json"))["generated"] == json.load(open(pb[:-4] + ".json"))["generated"]
    bad += not ok
    print(os.path.basename(pa), "identical" if ok else "DIFFERENT")
print(f"dev_attn_all vs dev: {bad} differ")
PY
echo "[$(date -u -Is)] B4 DONE"
