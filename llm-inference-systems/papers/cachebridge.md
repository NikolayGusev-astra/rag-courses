# [2609.00891] CacheBridge: Efficient Cross-Model KV Cache Transfer

# [2609.00891] CacheBridge: Efficient Cross-Model KV Cache Transfer
URL: https://arxiv.org/abs/2609.00891
Published: 2026-09-01

[2609.00891] CacheBridge: Efficient Cross-Model KV Cache Transfer
Skip to main content

archive

Search arXiv
Press Enter to search · Advanced search
# Computer Science > Artificial Intelligence
**arXiv:2609.00891** (cs)
[Submitted on 1 Sep 2026]
# Title:CacheBridge: Efficient Cross-Model KV Cache Transfer
Authors:Xingyu Qu, Siyuan Lu, Zhiyu Chen, Sheng Wang, Tao Lin
View a PDF of the paper titled CacheBridge: Efficient Cross-Model KV Cache Transfer, by Xingyu Qu and 4 other authors
View PDF
HTML (experimental)
>
> Abstract:
> Sharing context between LLMs in a multi-model system requires the receiving model to prefill the shared prefix because KV caches are model-specific. Recent closed-form cross-model KV transfer, hereafter Full-Head Mapping, avoids this replay by fitting a training-free affine mapper from source to target caches. However, its full-head design maps each target KV head from every source KV head in the selected layers, making transfer quality sensitive to architectural differences and causing mapper storage and application cost to grow with layer support. To this end, we introduce CacheBridge, which co-designs architecture-indexed mapper support, attention-aligned calibration, and bounded mapper construction while retaining a closed-form affine interface for online deployment. CacheBridge restricts each target head to a matched source head, weights reconstruction errors by causal attention sensitivity, and uses a fused GPU kernel to construct weighted sufficient statistics without materializing full observation tensors. Across three transfer directions, CacheBridge recovers the two Ministral 3 transfer directions where Full-Head Mapping loses substantial accuracy while preserving 99.83\% mean target retention on Qwen3. On Qwen3
> 14
> B
> →
> 32
> B
> , it reduces mapper storage by
> 8
> ×
> , accelerates application by up to
> 3.0
> ×
> , matches \fullhead with one tenth of the calibration data, and reduces 500-sequence construction from 92.63 to 8.63 seconds (
> 10.7
> ×
> ).
|Subjects:|Artificial Intelligence (cs.AI)|
|Cite as:|arXiv:2609.00891 [cs.AI]|
| |(or arXiv:2609.00891v1 [cs.AI] for this version)
|
| |https://doi.org/10.48550/arXiv.2609.00891
Focus to learn more
arXiv-issued DOI via DataCite (pending registration)
|
## Submission history
From: Siyuan Lu [view email]
**[v1]**
Tue, 1 Sep 2026 08:20:35 UTC (239 KB)
Full-text links:
## Access Paper:
View a PDF of the paper titled CacheBridge: Efficient Cross-Model KV Cache Transfer, by Xingyu Qu and 4 other authors
* View PDF
* HTML (experimental)
* TeX Source

license icon
view license

### Current browse context:
cs.AI
< prev
 | 
next >
new
|
recent
| 2026-09
Change to browse by:
cs
### References & Citations
* NASA ADS
* Google Scholar
* Semantic Scholar
export BibTeX citation
Loading...
## BibTeX formatted citation
×
loading...
Data provided by:
### Bookmark

BibSonomy

Reddit

Bibliographic Tools
# Bibliographic and Citation Tools
Bibliographic Explorer Toggle
Bibliog