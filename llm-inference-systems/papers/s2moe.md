# [2608.15018] S2-MoE: Enabling Efficient Self-Speculative Decoding for Mixture-of-Experts on Edge Devices

# [2608.15018] S2-MoE: Enabling Efficient Self-Speculative Decoding for Mixture-of-Experts on Edge Devices
URL: https://arxiv.org/abs/2608.15018
Published: 2026-08-15
Author: Huang, Haochen, Qiu, Shengxuan, Li, Meng

[2608.15018] S2-MoE: Enabling Efficient Self-Speculative Decoding for Mixture-of-Experts on Edge Devices

# Title: S2-MoE: Enabling Efficient Self-Speculative Decoding for Mixture-of-Experts on Edge Devices

Authors: Haochen Huang, Shengxuan Qiu, Meng Li

> Abstract: Deploying large language models (LLMs) for inference on edge devices is challenging due to severe memory and bandwidth constraints. While speculative decoding and Mixture-of-Experts (MoE) have been proposed to improve inference efficiency, naively combining them often incurs excessive verification overhead and poor expert reuse, limiting their effectiveness in memory-bound edge settings. In this work, we propose S2-MoE, an efficient self-speculative decoding framework for MoE inference on edge devices. S2-MoE reduces redundant verification through routing-aware adaptive speculative expansion, improves verification efficiency with reuse-aware expert gating, and aligns draft and target execution via shared context. Implemented in llama$.$cpp, S2-MoE achieves up to $5.3\times$ speedup (about $2.0\times$ on average) over standard autoregressive decoding across diverse MoE models and datasets on edge devices. Code is available at this https URL. 

| Comments: | 13 pages, 10 figures |
| --- | --- |
| Subjects: | Artificial Intelligence (cs.AI) |
| Cite as: | arXiv:2608.15018 [cs.AI] |
| | (or arXiv:2608.15018v2 [cs.AI] for this version) |
| | https://doi.org/10.48550/arXiv.2608.15018 Focus to learn more arXiv-issued DOI via DataCite |

## Submission history

 From: Haochen Huang [view email] [v1] Sat, 15 Aug 2026 04:08:04 UTC (518 KB) [v2] Wed, 19 Aug 2026 09:26:35 UTC (518 KB) 

## Access Paper:

HTML (experimental)

TeX Source