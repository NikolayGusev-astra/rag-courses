# [2309.17453] Efficient Streaming Language Models with Attention Sinks

# [2309.17453] Efficient Streaming Language Models with Attention Sinks
URL: https://arxiv.org/abs/2309.17453
Published: 2023-09-29

[2309.17453] Efficient Streaming Language Models with Attention Sinks
Skip to main content

archive

Search arXiv
Press Enter to search · Advanced search
# Computer Science > Computation and Language
**arXiv:2309.17453** (cs)
[Submitted on 29 Sep 2023 (v1), last revised 7 Apr 2024 (this version, v4)]
# Title:Efficient Streaming Language Models with Attention Sinks
Authors:Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis
View a PDF of the paper titled Efficient Streaming Language Models with Attention Sinks, by Guangxuan Xiao and 4 other authors
View PDF
HTML (experimental)
>
> Abstract:
> Deploying Large Language Models (LLMs) in streaming applications such as multi-round dialogue, where long interactions are expected, is urgently needed but poses two major challenges. Firstly, during the decoding stage, caching previous tokens' Key and Value states (KV) consumes extensive memory. Secondly, popular LLMs cannot generalize to longer texts than the training sequence length. Window attention, where only the most recent KVs are cached, is a natural approach -- but we show that it fails when the text length surpasses the cache size. We observe an interesting phenomenon, namely attention sink, that keeping the KV of initial tokens will largely recover the performance of window attention. In this paper, we first demonstrate that the emergence of attention sink is due to the strong attention scores towards initial tokens as a "sink" even if they are not semantically important. Based on the above analysis, we introduce StreamingLLM, an efficient framework that enables LLMs trained with a finite length attention window to generalize to infinite sequence lengths without any fine-tuning. We show that StreamingLLM can enable Llama-2, MPT, Falcon, and Pythia to perform stable and efficient language modeling with up to 4 million tokens and more. In addition, we discover that adding a placeholder token as a dedicated attention sink during pre-training can further improve streaming deployment. In streaming settings, StreamingLLM outperforms the sliding window recomputation baseline by up to 22.2x speedup. Code and datasets are provided at
> this https URL
> .
|Comments:|ICLR 2024|
|Subjects:|Computation and Language (cs.CL); Artificial Intelligence (cs.AI)|
|Cite as:|arXiv:2309.17453 [cs.CL]|
| |(or arXiv:2309.17453v4 [cs.CL] for this version)
|
| |https://doi.org/10.48550/arXiv.2309.17453
Focus to learn more
arXiv-issued DOI via DataCite
|
## Submission history
From: Guangxuan Xiao [view email]
**[v1]**
Fri, 29 Sep 2023 17:59:56 UTC (9,548 KB)
**[v2]**
Tue, 21 Nov 2023 05:04:49 UTC (13,230 KB)
**[v3]**
Tue, 12 Dec 2023 00:33:07 UTC (13,230 KB)
**[v4]**
Sun, 7 Apr 2024 00:56:53 UTC (13,231 KB)
Full-text links:
## Access Paper:
View a PDF of the paper titled Efficient Streaming Language Models with Attention Sinks, by Guangxuan Xiao and 4 other authors
* View PDF
* HTML (experimental)
* TeX Source

license icon
view license

### Current browse contex