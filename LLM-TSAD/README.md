# README for Supplementary Material

This document provides an overview of the supplementary materials submitted in support of our paper:

**Paper Title**: *Delving into Large Language Models for Effective Time-Series Anomaly Detection* 


# Environment Setup
* This repository is built upon [AnomLLM](https://github.com/rose-stl-lab/anomllm), and therefore **must be set up using the AnomLLM environment**.  
Please follow the installation instructions provided in the AnomLLM repository before proceeding.

* Dataset Download
  * AnomLLM: Downloaad "anomllm.zip" in the provied link in README of https://github.com/rose-stl-lab/anomllm
  * TSB-AD-U: Download "Datasets" directory in https://github.com/TheDatumOrg/TSB-AD/tree/main/Datasets

* For Qwen and InternVL models, we use [LMDeploy](https://lmdeploy.readthedocs.io/en/latest/#).

* File Structure
```
LLM-TSAD/
│
├── ...
├── credentials.yml               # For online API
├── data
     ├── synthetic                # For AnomLLM benchmark
├── TSB-AD
     ├── Datasets                 # For TSB-AD-U benchmark
└── README.md                     # This file
```

# Run Our Method

## Experiemntal Results on AnomLLM Benchmark

1. Run online api (For convenience, we have saved the results of the previous run in the ./results/ directory. Therefore, you can proceed directly to step 2.)
```
python src/LLM-TSAD-AnomLLM_api.py --model gemini-1.5-flash --data trend --variant 0shot-text-vision
```

2. Aggregate evaluation results
```
python ./src/result_agg_by_model.py --model gemini-1.5-flash --benchmark anomllm
```

## Experiemntal Results on TSB-AD-U Benchmark

1. Run online api (For convenience, we have saved the results of the previous run in the ./results/ directory. Therefore, you can proceed directly to step 2.)
```
python src/LLM-TSAD-TSB_api.py --model gemini-1.5-flash --datadir ./TSB-AD/Datasets
```

2. Aggregate evaluation results
```
python ./src/result_agg_by_model.py --model gemini-1.5-flash  --benchmark tsb-ad-u
```
