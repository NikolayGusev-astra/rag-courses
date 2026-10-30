# [2608.03893] Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse

# [2608.03893] Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse
URL: https://arxiv.org/abs/2608.03893
Published: 2026-08-04
Author: Heo, Taekyung, Shafipour, Rasoul, Zhao, Ritchie, Golub, Maximilian, Kamani, Mohammad Mahdi

[2608.03893] Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse

 [Submitted on 4 Aug 2026]

# Title: Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse

Authors: Taekyung Heo, Rasoul Shafipour, Ritchie Zhao, Maximilian Golub, Mohammad Mahdi Kamani, Ritika Borkar, Makesh Tarun Chandran, Pantea Zardoshti, Bita Darvish Rouhani

> Abstract: Production deployments often swap between different-sized models in a family for cost-quality cascading, mid-conversation switching, and routing, and each swap forces the receiver to repay the prefill from scratch. We propose cross-model KV cache transfer, where the receiver reuses the source's KV cache, skipping prefill. We find that cross-model KV has substantial linear structure across matched-KV pairs, where source and target share KV head count and per-head dimension. On Qwen3 14B->32B, one source layer explains 56% of variance in the target's keys and 32% in values, rising to 79% and 65% with multiple source layers. Building on this, we design a closed-form ridge mapper that operates per head and proceeds in three steps. First, for each target layer we select the top-k most predictive source layers and concatenate their KV as input. Second, we strip RoPE from the keys before mapping, so the fit is position-free and reusable across context lengths. Third, we fit ridge regression on a small calibration set of 500 FineWeb-Edu sequences of 1,024 tokens each. Surprisingly, across six pairs in three families, this linear mapper retains 73-98% of the receiver's standalone-prefill accuracy on four pairs, while two degrade sharply. A nonlinear MLP recovers up to +37 pp HellaSwag retention on the failures. The mapper runs 2.7-25x faster than re-prefill and remains stable across multi-turn handoff, making cross-model KV cache transfer practical. 

| Subjects: | Machine Learning (cs.LG) |
| --- | --- |
| Cite as: | arXiv:2608.03893 [cs.LG] |
| | (or arXiv:2608.03893v1 [cs.LG] for this version) |
| | https://doi.org/10.48550/arXiv.2608.03893 Focus to learn more arXiv-issued DOI via DataCite (pending registration) |

## Submission history

 From: Taekyung Heo [view email] [v1] Tue, 4 Aug 2026 16:26:47 UTC (1,193 KB) 

## Access Paper: