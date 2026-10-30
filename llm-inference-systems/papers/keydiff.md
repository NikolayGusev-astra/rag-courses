# KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments

# KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments
URL: https://arxiv.org/abs/2504.15364
Published: 2026-01-20
Author: Dalton Jones

KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments

# KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments

 Junyoung Park † † thanks: Corresponding author Dalton Jones Matthew J Morse Raghavv Goel Mingu Lee Chris Lott Affiliation: Qualcomm AI Research Email: {junpark,daltjone,mattmors,raghgoel,mingul,clott}@qti.qualcomm.com 

###### Abstract

We demonstrate that geometrically distinctive keys during LLM inference tend to have high attention scores. Based on the phenomenon we propose KeyDiff, a training-free KV cache eviction method based solely on key similarity. Unlike other KV cache eviction methods, KeyDiff can process arbitrarily long prompts within strict resource constraints and efficiently generate responses. We provide a theoretical basis for KeyDiff by relating key diversity with attention scores. These results imply KeyDiff can efficiently identify the most important tokens to retain. Notably KeyDiff does not rely on attention scores, allowing the use of optimized attention mechanisms like FlashAttention. Under a strict memory allowance, we demonstrate the effectiveness of KeyDiff for the Llama and Qwen model families by observing a performance gap of less than 0.04% with 8K cache budget ($∼\sim$ 23% KV cache reduction) from the non-evicting baseline on LongBench for Llama 3.1-8B and Llama 3.2-3B. We also observe near baseline performance for Deepseek-R1-Distill-Llama-8B on the Math500 reasoning benchmark and decrease end-to-end inference latency by up to 30% compared to the other token-eviction methods.

### 1 Introduction

Key-Value (KV) caching is a standard technique to accelerate large language model (LLM) inference that reuses key and value states (KVs) from previously processed tokens, enabling efficient autoregressive generation. This is crucial for long-context applications such as document summarization, code generation, question answering [7, 25, 31, 10], retrieval augmented generation [19] and reasoning [34, 17, 41]. However, the memory footprint of the stored KV cache grows linearly with input length, which becomes a bottleneck in memory-constrained environments.

This challenge is particularly acute for LLM inference on edge device, where compute, memory, and power resources are limited [3, 22, 32, 36]. While KV cache eviction policies have been proposed to bound memory overhead by removing unimportant KVs (often measured by attention scores) [35, 24, 43], they typically process the entire prompt at once and violate memory constraints during intermediate computation.

To enforce strict memory bounds throughout the prompt prefill and token generation inference phases, we adopt a block-wise inference strategy: the input prompt is divided into smaller blocks which are processed sequentially by the model, similar to [18, 1, 37]. After processing each block, we evict some of the cached KVs by accordin