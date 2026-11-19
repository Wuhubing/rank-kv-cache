# Can Selection Rank Improve KV Cache Replacement?

A single-GPU study of HiSparse-style hierarchical KV caching (MIT 6.5940, Fall 2026, Project 10).

We test whether a page's sparse-attention top-K selection rank predicts reuse beyond
recency and frequency, using Qwen3-0.6B traces on LongBench (MultiFieldQA-en, GovReport).

## Team

Jiayu Li, Mingxi Wang, Weibing Wang, Xiaoyang Wu, Jimmy Yu
