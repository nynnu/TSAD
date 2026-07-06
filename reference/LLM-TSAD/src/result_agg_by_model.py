import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import re
import json
import math
import glob
import argparse
    
    
def average_dict_values(dict_list):
    sums = {}
    counts = {}
    for d in dict_list:
        for key, value in d.items():
            sums[key] = sums.get(key, 0) + value
            counts[key] = counts.get(key, 0) + 1

    averages = {key: sums[key] / counts[key] for key in sums}
    return averages


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process online API anomaly detection.')
    parser.add_argument('--variant', type=str, default='0shot-text-vision', help='Variant type')
    parser.add_argument('--model', type=str, default='gemini-1.5-flash', help='Model name')
    #'OpenGVLab/InternVL2-Llama3-76B' #'Qwen/Qwen-VL-Chat' #'OpenGVLab/InternVL2-Llama3-76B'# 'gpt-4o'# gemini-1.5-flash
    parser.add_argument('--benchmark', type=str, default='anomllm', help='benchmark name')
    # anomllm, tsb-ad-u
    return parser.parse_args()

args = parse_arguments()

agg_per_data = False
result_paths = []
if args.benchmark == 'anomllm':
    for data_name in ['trend', 'freq', 'point', 'range']:
        result_paths.append(f'./results/synthetic/{data_name}/{args.model}/{args.variant}.jsonl')
elif args.benchmark == 'tsb-ad-u':
    agg_per_data = True
    result_paths.append(f'./results/tsb-ad-u/{args.model}/{args.variant}.jsonl')
        
print(result_paths)


all_results = []
for path in result_paths:
    result_df = pd.read_json(path, lines=True)
    result_df['cate'] = result_df['custom_id'].apply(lambda x: x.split('_')[0])
    print(path, result_df.shape)     
    
    if agg_per_data:
        for cate in result_df.cate.value_counts().index.tolist():
            a =  average_dict_values(result_df[result_df.cate == cate]['metrics'].tolist())
            log_str = [cate]
            for k, v in a.items():
                log_str.append(f'{v*100:.2f}')
            print('\t'.join(log_str))
            all_results.append(a)
    else:
        a =  average_dict_values(result_df['metrics'].tolist())
        log_str = []
        for k, v in a.items():
            log_str.append(f'{v*100:.2f}')
        print('\t'.join(log_str))

        all_results.append(a)
    print()
    
    
a = average_dict_values(all_results)
log_str = []
for k, v in a.items():
    log_str.append(f'{v*100:.2f}')
print('\t'.join(log_str))
    