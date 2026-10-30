# [2602.07223] SpecAttn: Co-Designing Sparse Attention with Self-Speculative Decoding

# [2602.07223] SpecAttn: Co-Designing Sparse Attention with Self-Speculative Decoding
URL: https://arxiv.org/abs/2602.07223
Published: 2026-02-06
Author: Yue, Yikang, Xue, Yuqi, Huang, Jian

[2602.07223] SpecAttn: Co-Designing Sparse Attention with Self-Speculative Decoding

 [Submitted on 6 Feb 2026]

# Title: SpecAttn: Co-Designing Sparse Attention with Self-Speculative Decoding

Authors: Yikang Yue, Yuqi Xue, Jian Huang

> Abstract: Long-context large language model (LLM) inference has become the norm for today's AI applications. However, it is severely bottlenecked by the increasing memory demands of its KV cache. Previous works have shown that self-speculative decoding with sparse attention, where tokens are drafted using a subset of the KV cache and verified in parallel with full KV cache, speeds up inference in a lossless way. However, this approach relies on standalone KV selection algorithms to select the KV entries used for drafting and overlooks that the criticality of each KV entry is inherently computed during verification. In this paper, we propose SpecAttn, a self-speculative decoding method with verification-guided sparse attention. SpecAttn identifies critical KV entries as a byproduct of verification and only loads these entries when drafting subsequent tokens. This not only improves draft token acceptance rate but also incurs low KV selection overhead, thereby improving decoding throughput. SpecAttn achieves 2.81$\times$ higher throughput over vanilla auto-regressive decoding and 1.29$\times$ improvement over state-of-the-art sparsity-based self-speculative decoding methods. 

| Subjects: | Machine Learning (cs.LG) |
| --- | --- |
| Cite as: | arXiv:2602.07223 [cs.LG] |
| | (or arXiv:2602.07223v1 [cs.LG] for this version) |
| | https://doi.org/10.48550/arXiv.2602.07223 Focus to learn more arXiv-issued DOI via DataCite (pending registration) |

## Submission history

 From: Yikang Yue [view email] [v1] Fri, 6 Feb 2026 22:12:52 UTC (364 KB) 

## Access Paper: