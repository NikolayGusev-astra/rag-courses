# [2604.15039] Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter

# [2604.15039] Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter
URL: https://arxiv.org/abs/2604.15039
Published: 2026-04-16
Author: Qin, Ruoyu, He, Weiran, Wang, Yaoyu, Li, Zheming, Xu, Xinran

[2604.15039] Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter

 [Submitted on 16 Apr 2026]

# Title: Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter

Authors: Ruoyu Qin, Weiran He, Yaoyu Wang, Zheming Li, Xinran Xu, Yongwei Wu, Weimin Zheng, Mingxing Zhang

> Abstract: Prefill-decode (PD) disaggregation has become the standard architecture for large-scale LLM serving, but in practice its deployment boundary is still determined by KVCache transfer. In conventional dense-attention models, prefill generates huge KVCache traffics that keep prefill and decode tightly coupled within a single high-bandwidth network domain, limiting heterogeneous deployment and resource elasticity. Recent hybrid-attention architectures substantially reduce KVCache size, making cross-cluster KVCache transport increasingly plausible. However, smaller KVCache alone does not make heterogeneous cross-datacenter PD serving practical: real workloads remain bursty, request lengths are highly skewed, prefix caches are unevenly distributed, and inter-cluster bandwidth fluctuates. A naive design that fully externalizes prefill can therefore still suffer from congestion, unstable queueing, and poor utilization. We present Prefill-as-a-Service (PrfaaS), a cross-datacenter serving architecture that selectively offloads long-context prefill to standalone, compute-dense prefill clusters and transfers the resulting KVCache over commodity Ethernet to local PD clusters for decode. Rather than treating reduced KVCache as sufficient, PrfaaS combines model-side KV efficiency with system-side selective offloading, bandwidth-aware scheduling, and cache-aware request placement. This design removes the requirement that heterogeneous accelerators share the same low-latency RDMA fabric, enabling independent scaling of prefill and decode capacity across loosely coupled clusters. In a case study using an internal 1T-parameter hybrid model, a PrfaaS-augmented heterogeneous deployment achieves 54% and 32% higher serving throughput than homogeneous PD and naive heterogeneous baselines, respectively, while consuming only modest cross-datacenter bandwidth. 

| Comments: | 16 pages, 5 figures, 6 tables |
| --- | --- |
| Subjects: | Distributed, Parallel, and Cluster Computing (cs.DC) |
| Cite as: | arXiv:2604.15039 [cs.DC] |
| | (or arXiv:2604.15039v1 [cs.DC] for this version) |
| | https://doi.org/10.48550/arXiv.2604.15039 Focus to learn more arXiv-issued DOI via DataCite (pending registration) |

## Submission history

 From: Ruoyu Qin [view email] [v1] Thu, 16 Apr 2026 14:07:41 UTC (244 KB) 

## Access Paper:

TeX Source