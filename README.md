# LLM Materials
记录平时阅读的有关大模型的一些材料

## LLM Training

### Techniques
- [Arxiv 2024] [LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/abs/2402.12354). Applying different learning rate to adapter matrices A and B in LoRA. [Code](https://github.com/nikhil-ghosh-berkeley/loraplus)
- [Arxiv 2021] [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/abs/2106.09685).  Training large model with minor trainable parameters by using low rank adapters. [Code](https://github.com/microsoft/LoRA)

### Datasets
- [some open hug datasets.](https://kili-technology.com/large-language-models-llms/9-open-sourced-datasets-for-training-large-language-models) Mainly used for pretrain LLM.
- [stanford alpaca dataset.](https://github.com/tatsu-lab/stanford_alpaca) 52k data for instruction finetuning LLM.
- [alpaca cleaned finetune dataset.](https://github.com/gururise/AlpacaDataCleaned) a cleaned and curated version of stanford alpaca dataset.

## LLM Evaluation 
- [open llm leaderboard.](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) Evaluating LLM model with six different task.
- [tinyBenchmarks](https://huggingface.co/tinyBenchmarks), [paper](https://arxiv.org/abs/2402.14992). A small version of open llm leaderboard. This benchmark only contains 100 samples of each task.
- [lm-evaluation-harness.](https://github.com/EleutherAI/lm-evaluation-harness) A evaluation tool, which can evaluate LLM on different tasks with sample commands.


## Model Compression
### Survey
- [Arxiv 2023] [A Survey on Model Compression for Large Language Models](https://arxiv.org/abs/2308.07633).

### Low-rank  Decomposition
- [Arxiv 2024] [Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy](https://arxiv.org/abs/2310.01334) grouping experts in a layer, merging them into a single expert, and then decomposing the merged expert with Singular Value Decomposition (SVD). [Code](https://github.com/UNITES-Lab/MC-SMoE)
- [Arxiv 2024] [SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression](https://arxiv.org/abs/2403.07378). Proposing a truncation-aware data whitening strategy to ensure a direct mapping between singular values and compression loss and adopting a layer-wise closed-form model parameter update strategy to compensate for accuracy degradation caused by SVD truncation. [Code](https://github.com/AIoT-MLSys-Lab/SVD-LLM)
- [Arxiv 2023] [ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models](https://arxiv.org/abs/2312.05821). Adding activation to the SVD process to mitigate the outliters problem of activation. [Code](https://github.com/hahnyuan/ASVD4LLM)
- [Arxiv 2023] [LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation](https://arxiv.org/abs/2306.11222) This method combines the advantages of both low-rank approximations and pruning, using the low-rank svd to decompose the (W-S) matrix, where W is weight matrix and S is the pruning matrix. [Code](https://github.com/yxli2123/LoSparse)

### Pruning
- [Arxiv 2024] [Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy](https://arxiv.org/abs/2310.01334) grouping experts in a layer, merging them into a single expert, and then decomposing the merged expert with Singular Value Decomposition (SVD). [Code](https://github.com/UNITES-Lab/MC-SMoE)
- [Arxiv 2023] [LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation](https://arxiv.org/abs/2306.11222) This method combines the advantages of both low-rank approximations and pruning, using the low-rank svd to decompose the (W-S) matrix, where W is weight matrix and S is the pruning matrix. [Code](https://github.com/yxli2123/LoSparse)


## ML System
### Training System
- [ATC 2023] [SmartMoE: Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization](https://www.usenix.org/conference/atc23/presentation/zhai). Dynamic runtime selection of optimal parallel strategy is enabled by efficient searching algorithm. [Code](https://github.com/zms1999/SmartMoE)
- [ATC 2023] [Accelerating Distributed MoE Training and Inference with Lina](https://www.usenix.org/conference/atc23/presentation/li-jiamin). First,  ystematically analyzing all-to-all overhead in distributed MoE and present the main causes for it to be the bottleneck in training and inference, respectively. Second, designing and building Lina to address the all-to-all bottleneck head-on. Lina opportunistically prioritizes all-to-all over the concurrent allreduce whenever feasible using tensor partitioning, so all-to-all and training step time is improved.
- [IPDPS 2023] [MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism](https://ieeexplore.ieee.org/document/10177396). a high-performance library that accelerates MoE training with adaptive and memory-efficient pipeline parallelism. Inspired by that the MoE training procedure can be divided into multiple independent sub-stages, designing adaptive pipeline parallelism with an online algorithm to configure the granularity of the pipelining. [Code](https://github.com/whuzhangzheng/MPipeMoE)
- [Arxiv 2023] [SE-MoE: A Scalable and Efficient Mixture-of-Experts Distributed Training and Inference System](https://arxiv.org/abs/2205.10034). SE-MoE proposes Elastic MoE training with 2D prefetch and Fusion communication over Hierarchical storage, so as to enjoy efficient parallelisms in various types. For scalable inference in a single node, especially when the model size is larger than GPU memory, SE-MoE forms the CPU-GPU memory jointly into a ring of sections to load the model, and executes the computation tasks across the memory sections in a round-robin manner for efficient inference.  [Code](https://github.com/PaddlePaddle/Paddle)
- [SIGCOMM 2023] [Janus: A Unified Distributed Training Framework for Sparse Mixture-of-Experts Models](https://dl.acm.org/doi/pdf/10.1145/3603269.3604869). All-to-All communication originates from expert-centric paradigm: keeping experts in-place and exchanging intermediate data to feed experts. Proposing the novel data-centric paradigm: keeping data in-place and moving experts between GPUs. 
- [Arxiv 2023] [TUTEL: ADAPTIVE MIXTURE-OF-EXPERTS AT SCALE](https://arxiv.org/abs/2206.03382). Flex designs an identical layout for distributing MoE model parameters and input data, which can be leveraged by all possible parallelism or pipelining methods without any mathematical inequivalence or tensor migration overhead. [Code](https://github.com/microsoft/tutel)
- [PPoPP 2022] [FasterMoE: Modeling and Optimizing Training of Large-Scale Dynamic Pre-Trained Models](https://dl.acm.org/doi/10.1145/3503221.3508418). Propose a performance model that can both accurately predict the latency of different operations of a specific training task, and intuitively analyze its end-to-end performance via a novel roofline-like model. Then, guided by this model, inventing a dynamic shadowing approach to cope with load imbalance, and a smart fine-grained schedule that splits different operations and executes them concurrently. [Code](https://github.com/thu-pacman/FasterMoE)
- [OSDI 2020] [A Unified Architecture for Accelerating Distributed DNN Training in Heterogeneous GPU/CPU Clusters](https://www.usenix.org/conference/osdi20/presentation/jiang) [Code](https://github.com/bytedance/byteps)
- [Arxiv 2022] [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://arxiv.org/abs/2201.05596). An end-to-end MoE training and inference system. [Code](https://github.com/microsoft/DeepSpeed)
### Inference System
- [Arxiv 2024] [MOE-INFINITY: Activation-Aware Expert Offloading for Efficient MoE Serving](https://arxiv.org/abs/2401.14361).  Designing a cost-efficient mixture-of-expert (MoE) serving system that realizes activation-aware expert offloading, including sequence-level tracing, activation-aware expert prefetching and caching techniques. [Code](https://github.com/TorchMoE/MoE-Infinity)
