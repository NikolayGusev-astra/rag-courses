# Swarm Telepathy: Prefill-Free KV-Cache Transfer Across Families

# Swarm Telepathy: Prefill-Free KV-Cache Transfer Across Families
URL: https://arxiv.org/abs/2610.15095
Published: 2026-10-10
Author: Abdulrahman Alabdulkareem, Irene Tenison, Yang Zhang, Yada Zhu, Boris Katz, Dan Gutfreund

# Swarm Telepathy: Prefill-Free KV-Cache Transfer Across Families

arXiv (Cornell University). Published: 2026-10-10. Preprint. 0 citations.

## Abstract

LLM multi-agent systems split work across agents that communicate through text. An agent summarizes its findings, another agent receives the text. This constrains communication to discrete tokens and costs redundant compute, limiting bandwidth and capabilities of multi-agent systems. Recent works demonstrate KV-cache transfer between LLMs but they either require the LLMs to be within the same family or cannot function across context, thus requiring the receiver to prefill text. We present SwaT (Swarm Telepathy), the first cross-family KV-cache transfer method that allows prefill-free communication in multi-family multi-agent systems. In an agent-to-agent communication setting, combining SwaT with recent advances in KV-cache compaction beats text-to-text communication in both accuracy and speed, with a mean of $40\%$ higher accuracy and a mean of $4 imes$ faster wall-clock time. Experiments show SwaT achieves high performance on 17 benchmarks across 12 LLMs and 6 families. Additionally, given a sender Vision-Language Model (VLM), SwaT is able to transfer the vision capabilities to a text-only receiver LLM exceeding text-to-text communication despite SwaT being trained on text-only data. SwaT keeps both sender and receiver models frozen without fine-tuning and only requires $<50M$ tokens and $<1$ GPU-day to train.

## Authors

- Abdulrahman Alabdulkareem (MIT Computer Science and Artificial Intelligence Laboratory): h-index 1; 2 citations
- Irene Tenison (MIT Computer Science and Artificial Intelligence Laboratory): h-index 5; 61 citations
- Yang Zhang (MIT-IBM Computing Research Lab): h-index 0; 0 citations
- Yada Zhu (MIT-IBM Computing Research Lab): h-index 0; 0 citations
- Boris Katz (MIT Computer Science and Artificial Intelligence Laboratory): h-index 0; 0 citations
- Dan Gutfreund (MIT-IBM Computing Research Lab): h-index 18; 1,310 citations

## Topics

- Topic Modeling
- Natural Language Processing Techniques
- Multimodal Machine Learning Applications