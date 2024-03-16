# LLM Materials
记录平时阅读的有关大模型的一些材料

## LLM Training

### Techniques

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
- [Arxiv 2024] [SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression](https://arxiv.org/abs/2403.07378). Proposing a truncation-aware data whitening strategy to ensure a direct mapping between singular values and compression loss and adopting a layer-wise closed-form model parameter update strategy to compensate for accuracy degradation caused by SVD truncation. [Code](https://github.com/AIoT-MLSys-Lab/SVD-LLM)
- [Arxiv 2023] [ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models](https://arxiv.org/abs/2312.05821). Adding activation to the SVD process to mitigate the outliters problem of activation. [Code](https://github.com/hahnyuan/ASVD4LLM)
