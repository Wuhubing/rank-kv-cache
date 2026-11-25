# B=96 长上下文单篇回放

选择冻结开发列表中的第一篇 GovReport（ID 前缀 `0708332f`）。输入 22,395 个 token，613 个 sparse decode 步，28 层，每步每层访问 K=64 个历史页。本次在 Mac CPU 上从原始 NPZ 回放。4 种策略 × 28 层共 112 行，每行的访问数、首次 miss 和总 miss 均与开发集归档相同。[原始 trace](https://drive.google.com/file/d/1cZ_2hAOMGzKz11OpG-pm8HE_9pGw-IsP/view) · [逐层实测与归档对照](by_layer.csv) · [输入哈希和设置](manifest.json)

| 策略 | 总 miss | 重复 miss | 总 miss 率 | 相对 LRU 的重复 miss 降幅 |
|---|---:|---:|---:|---:|
| LRU | 79,686 | 68,422 | 7.254% | — |
| 当前 Quest 分数 | 72,699 | 61,435 | 6.618% | 10.21% |
| 归一化 Quest 分数 EMA，α=0.5 | 71,864 | 60,600 | 6.542% | 11.43% |
| Bélády，知道未来的离线上界（最低 miss） | 51,303 | 40,039 | 4.670% | 41.48% |

每种策略都有 1,098,496 次访问和 11,264 次首次 miss。当前分数在驱逐时保留得分较高的候选页。EMA 每步为所有驻留页更新相对于本步第 64 名的归一化分数。LRU 到当前分数少了 6,987 次重复 miss；采用 EMA 又少了 835 次。EMA 在 28 层均优于 LRU，对当前分数为 24 层改善、2 层持平、2 层变差。

这是一篇已归档开发文档的复现。长 trace、B=96 和 score 策略属于扩展实验。原 proposal 的 10% 判据在短 trace、B=128、rank 策略上评价，其结论保持为未达到。跨文档稳定性需在全部开发文档检验，硬件耗时另测。

从分支根目录重跑：

```bash
python3 validation/longctx_document.py "$HOME/Downloads/gov_report__0708332f2b27ef8843c261a207213212e80322ad79286259.npz" --out docs/results/solo_longctx_pilot
```
