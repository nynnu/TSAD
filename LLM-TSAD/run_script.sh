#!/bin/bash

# 실행할 데이터셋 리스트
data_list=("freq" "trend" "range" "point")

# 사용할 모델 리스트
model_list=("gemini-1.5-flash" "gpt-4o")

# variant 리스트
variant_list=("0shot-text-vision")

# 루프 돌면서 모든 조합 실행
for data in "${data_list[@]}"; do
  for model in "${model_list[@]}"; do
    for variant in "${variant_list[@]}"; do
      echo "Running: data=${data}, model=${model}, variant=${variant}"
      python src/LLM-TSAD-AnomLLM_api.py --data "$data" --model "$model" --variant "$variant"
    done
  done
done
