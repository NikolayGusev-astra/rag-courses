# [2410.01104] Softmax is not Enough (for Sharp Size Generalisation)

# [2410.01104] Softmax is not Enough (for Sharp Size Generalisation)
URL: https://arxiv.org/abs/2410.01104
Published: 2024-10-01
Author: Veličković, Petar, Perivolaropoulos, Christos, Barbero, Federico, Pascanu, Razvan

[2410.01104] Softmax is not Enough (for Sharp Size Generalisation)

 [Submitted on 1 Oct 2024 (v1), last revised 30 May 2025 (this version, v3)]

# Title: Softmax is not Enough (for Sharp Size Generalisation)

Authors: Petar Veličković, Christos Perivolaropoulos, Federico Barbero, Razvan Pascanu

> Abstract: A key property of reasoning systems is the ability to make sharp decisions on their input data. For contemporary AI systems, a key carrier of sharp behaviour is the softmax function, with its capability to perform differentiable query-key lookups. It is a common belief that the predictive power of networks leveraging softmax arises from "circuits" which sharply perform certain kinds of computations consistently across many diverse inputs. However, for these circuits to be robust, they would need to generalise well to arbitrary valid inputs. In this paper, we dispel this myth: even for tasks as simple as finding the maximum key, any learned circuitry must disperse as the number of items grows at test time. We attribute this to a fundamental limitation of the softmax function to robustly approximate sharp functions with increasing problem size, prove this phenomenon theoretically, and propose adaptive temperature as an ad-hoc technique for improving the sharpness of softmax at inference time. 

| Comments: | To appear at ICML 2025. 22 pages, 9 figures |
| --- | --- |
| Subjects: | Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Information Theory (cs.IT) |
| Cite as: | arXiv:2410.01104 [cs.LG] |
| | (or arXiv:2410.01104v3 [cs.LG] for this version) |
| | https://doi.org/10.48550/arXiv.2410.01104 Focus to learn more arXiv-issued DOI via DataCite |

## Submission history

 From: Petar Veličković [view email] [v1] Tue, 1 Oct 2024 22:22:35 UTC (6,983 KB) [v2] Mon, 7 Oct 2024 13:13:41 UTC (6,792 KB) [v3] Fri, 30 May 2025 23:46:05 UTC (7,003 KB) 

## Access Paper: