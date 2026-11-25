# 14 篇长上下文开发文档的原始 trace 复现

在 Mac CPU 上用全部 14 个冻结开发 NPZ（10 篇 GovReport、4 篇 QMSum）回放 LRU、Bélády、当前 Quest 分数和归一化分数 EMA（α=0.5）。每层 B=96、K=64、16 token/page。1,568 组逐层结果的访问数、首次 miss 和总 miss 均与仓库归档一致。

| 策略 | 重复 miss | 相对 LRU 的重复 miss 降幅 |
|---|---:|---:|
| LRU | 1,020,400 | — |
| 当前 Quest 分数 | 917,194 | 10.11% |
| EMA | 902,197 | 11.58% |
| Bélády | 596,789 | 41.51% |

当前分数省下 103,206 次重复 miss，占 EMA 相对 LRU 总收益的 87.3%。EMA 再省 14,997 次，相对当前分数下降 1.64%，95% CI 为 1.50%–1.79%。当前分数优于 LRU、EMA 优于当前分数，均为 14/14 篇。区间基于 10,000 次整篇文档配对 bootstrap，seed 0。每种策略共享 19,556,096 次访问和 158,010 次首次 miss。

本表对应长上下文开发集。原提案在短上下文 B=128 比较的是 rank 策略，实际下降 0.405%。两组数据和预算分别见项目大纲。硬件耗时需要实际 GPU 测量。

```bash
python3 validation/longctx_development.py /path/to/longctx_v1_dev --out docs/results/solo_longctx_dev
```

[开发 trace 文件夹](https://drive.google.com/drive/folders/1ZwuV8ksLYh7yZYEBxQKzJc0_uW2F-Npu) · [逐文档与输入哈希](by_document.csv) · [逐层核对](by_layer.csv) · [配对区间](paired.csv) · [结果图](savings.png)
