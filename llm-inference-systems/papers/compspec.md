# Component-Aware Self-Speculative Decoding in Hybrid Language Models

# Component-Aware Self-Speculative Decoding in Hybrid Language Models
URL: https://arxiv.org/abs/2605.01106

Component-Aware Self-Speculative Decoding in Hybrid Language Models

arXiv is now an independent nonprofit! Learn more×

# Component-Aware Self-Speculative Decoding in Hybrid Language Models

Hector Borobia Email: hecboar@doctor.upv.es Note: Corresponding author. Affiliation: VRAIN – Valencian Research Institute for Artificial Intelligence, Universitat Politècnica de València, Valencia, Spain Elies Seguí-Mas Email: eseguim@upvnet.upv.es Affiliation: Department of Economics and Social Sciences, Universitat Politècnica de València, Valencia, Spain Guillermina Tormo-Carbó Email: gtormo@omp.upv.es Affiliation: Department of Business Organisation, Universitat Politècnica de València, Valencia, Spain

###### Abstract

Speculative decoding accelerates autoregressive inference by drafting candidate tokens with a fast model and verifying them in parallel with the target. Self-speculative methods avoid the need for an external drafter but have been studied exclusively in homogeneous Transformer architectures. We introduce component-aware self-speculative decoding, the first method to exploit the internal architectural heterogeneity of hybrid language models, isolating the SSM/linear-attention subgraph as a zero-cost internal draft. We evaluate this on two architecturally distinct hybrid families: Falcon-H1 (parallel: Mamba-2 + attention per layer) and Qwen3.5 (sequential: interleaved linear and attention layers), with a pure Transformer control (Qwen2.5). Parallel hybrids achieve acceptance rates of $\alpha=0.68$ at draft length $k=2$ under greedy decoding, while sequential hybrids yield only $\alpha=0.038$ —an $18\times$ gap attributable to how each architecture integrates its components. The property is scale-invariant: Falcon-H1 at 3B reproduces the rates observed at 0.5B. We further show that perplexity degradation from a companion ablation study predicts speculative viability without running speculative decoding: a $3.15\times$ ratio (Falcon) maps to $\alpha=0.37$ at $k=4$ , while $81.96\times$ (Qwen) maps to $\alpha=0.019$ . For sequential hybrids, generic LayerSkip achieves $12\times$ higher acceptance rates than the component-aware strategy. The composition pattern of hybrid models—not merely the presence of alternative components—determines whether component-level self-speculation is viable.

###### Keywords:

Speculative Decoding , Hybrid Language Models , State Space Models , Inference Efficiency , Self-Speculative Decoding , Architectural Analysis

## 1 Introduction

Autoregressive decoding remains the primary bottleneck for inference latency in large language models (LLMs). Each token generation requires a full forward pass through the model, making generation time proportional to sequence length regardless of hardware parallelism. Speculative decoding [1, 2] addresses this by using a fast draft model to propose multiple candidate tokens, which the target model then verifies in a single parallel forward pass. When draft tokens are accepted, mu