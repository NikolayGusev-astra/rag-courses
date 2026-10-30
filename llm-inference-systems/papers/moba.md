# MoBA: Mixture of Block Attention for Long-Context LLMs

# MoBA: Mixture of Block Attention for Long-Context LLMs
URL: https://arxiv.org/abs/2502.13189
Published: 2025-02-18
Author: Enzhe Lu Affiliation: Moonshot AI

MoBA: Mixture of Block Attention for Long-Context LLMs

# MoBA: Mixture of Block Attention for Long-Context LLMs

 Enzhe Lu Affiliation: Moonshot AI Zhejun Jiang Affiliation: Moonshot AI Jingyuan Liu Affiliation: Moonshot AI Yulun Du Affiliation: Moonshot AI Tao Jiang Affiliation: Moonshot AI Chao Hong Affiliation: Moonshot AI Shaowei Liu Affiliation: Moonshot AI Weiran He Affiliation: Moonshot AI Enming Yuan Affiliation: Moonshot AI Yuzhi Wang Affiliation: Moonshot AI Zhiqi Huang Affiliation: Moonshot AI Huan Yuan Affiliation: Moonshot AI Suting Xu Affiliation: Moonshot AI Xinran Xu Affiliation: Moonshot AI Guokun Lai Affiliation: Moonshot AI Yanru Chen Affiliation: Moonshot AI Huabin Zheng Affiliation: Moonshot AI Junjie Yan Affiliation: Moonshot AI Jianlin Su Affiliation: Moonshot AI Yuxin Wu Affiliation: Moonshot AI Neo Y. Zhang Affiliation: Moonshot AI Zhilin Yang Affiliation: Moonshot AI Xinyu Zhou Affiliation: Moonshot AI Mingxing Zhang Affiliation: Tsinghua University Jiezhong Qiu † † thanks: * zhang_mingxing@mail.tsinghua.edu.cn † † thanks: ‡ Co-corresponding authors. Xinyu Zhou (zhouxinyu@moonshot.cn), Jiezhong Qiu (jiezhongqiu@outlook.com) Affiliation: Zhejiang Lab/Zhejiang University 

###### Abstract

Scaling the effective context length is essential for advancing large language models (LLMs) toward artificial general intelligence (AGI). However, the quadratic increase in computational complexity inherent in traditional attention mechanisms presents a prohibitive overhead. Existing approaches either impose strongly biased structures, such as sink or window attention which are task-specific, or radically modify the attention mechanism into linear approximations, whose performance in complex reasoning tasks remains inadequately explored.

In this work, we propose a solution that adheres to the “less structure” principle, allowing the model to determine where to attend autonomously, rather than introducing predefined biases. We introduce Mixture of Block Attention (MoBA), an innovative approach that applies the principles of Mixture of Experts (MoE) to the attention mechanism. This novel architecture demonstrates superior performance on long-context tasks while offering a key advantage: the ability to seamlessly transition between full and sparse attention, enhancing efficiency without the risk of compromising performance. MoBA has already been deployed to support Kimi’s long-context requests and demonstrates significant advancements in efficient attention computation for LLMs. Our code is available at https://github.com/MoonshotAI/MoBA.

## 1 Introduction

The pursuit of artificial general intelligence (AGI) has driven the development of large language models (LLMs) to unprecedented scales, with the promise of handling complex tasks that mimic human cognition. A pivotal capability for achieving AGI is the ability to process, understand, and generate long sequences, which is essential for a wide range of applications, from historical da