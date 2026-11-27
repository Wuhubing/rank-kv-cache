#!/bin/bash
set -e
cd /data/weibing/projects/rank-kv-cache
export TMPDIR=/data/weibing/tmp CUDA_VISIBLE_DEVICES=0
D=gov_report/0708332f2b27ef8843c261a207213212e80322ad79286259
.venv/bin/python -m pytest -q tests/test_collect_long.py
.venv/bin/python -m collect.long --split splits/longctx_v1.json --subset dev --docs $D --out data/s2/a --dense-reference
.venv/bin/python -m collect.long --split splits/longctx_v1.json --subset dev --docs $D --out data/s2/b
.venv/bin/python -m collect.long --split splits/longctx_v1.json --subset dev --docs $D --out data/s2/rec --record-attn-mass --record-true-attn-rank
echo S2DONE
