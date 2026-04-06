import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import re
import json
import math
import glob
import torch
import base64
from io import BytesIO
import matplotlib.image as mpimg

from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from datetime import datetime, timedelta
from PIL import Image

from prompt import time_series_to_image
from utils import view_base64_image, display_messages, collect_results, compute_metrics, interval_to_vector, plot_series_and_predictions, plot_series_and_predictions_with_timestamp, vector_to_interval
from data.synthetic import SyntheticDataset

from neurips_our.AnoAgent import AnoAgent

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process online API anomaly detection.')
    parser.add_argument('--model', type=str, default='gemini-1.5-flash', help='Model name')
    parser.add_argument('--datadir', type=str, default='/home/jovyan/project/TSB-AD/Datasets/', help='TSB-AD Data Directory')
    parser.add_argument('--index', type=str, default='number', help='Index type')
    
    return parser.parse_args()


def find_intervals(series):
    intervals = []
    start = None

    for i, value in enumerate(series):
        # 1의 시작 지점 발견 (현재 값이 1이고, 이전에 시작 지점이 정해지지 않았을 경우)
        if value == 1 and start is None:
            start = i
        # 1의 연속 구간이 끝나는 지점: 현재 값이 0인데, 1의 구간이 진행 중인 경우
        if value == 0 and start is not None:
            intervals.append([start, i])
            start = None

    # 시퀀스가 1로 끝난 경우 마지막 구간 처리
    if start is not None:
        intervals.append([start, len(series)])
    
    return intervals

def make_eval_datasets(df):
    series = torch.tensor(df[['Data']].values)

    ano_sections = find_intervals(df.Label)
    ano_sections = torch.tensor(ano_sections).unsqueeze(0)

    return ano_sections, series


def build_tsb_ad_u_short_dataset(datadir='/home/jovyan/project/TSB-AD/Datasets/'):
    filename_df = pd.read_csv(os.path.join(datadir, 'File_List/TSB-AD-U-Eva.csv'))
    filename_df.file_name = filename_df.file_name.apply(lambda x: os.path.join(datadir, 'TSB-AD-U', x))
    filename_df['cate'] = filename_df.file_name.apply(lambda x: x.split('/')[-1].split('_')[1])
    filename_df.info()

    top8_short_data_names = ['NEK', 'TAO', 'MSL', 'Power', 'Daphnet', 'YAHOO', 'SED', 'TODS']

    # Filter and sort by category order
    eval_file_list_df = filename_df[filename_df.cate.isin(top8_short_data_names)].copy()
    eval_file_list_df['cate'] = pd.Categorical(eval_file_list_df['cate'], categories=top8_short_data_names, ordered=True)
    eval_file_list_df = eval_file_list_df.sort_values('cate')
    eval_file_list_df.info()

    eval_dataset = []
    for _, row in eval_file_list_df.iterrows():
        path = row['file_name']
        cate = row['cate']
        df = pd.read_csv(path)
        print(df.shape)
        ano_sections, series = make_eval_datasets(df)
        eval_dataset.append((cate, ano_sections, series))  # 카테고리 정보를 추가

    print(len(eval_dataset))

    return eval_dataset
    
    
def online_AD_with_retries(
    model_name: str,
    num_retries: int = 4,
):
    import json
    import time
    import pickle
    import os
    from loguru import logger
    
    args = parse_arguments()
    results = {}
    
        # Configure logger
    data_name = 'tsb-ad-u'
    variant = '0shot-text-vision'
    log_fn = f"logs/{data_name}/{model_name}/" + variant + ".log"
    logger.add(log_fn, format="{time} {level} {message}", level="INFO")
    results_dir = f'results/{data_name}/{model_name}/'
    jsonl_fn = os.path.join(results_dir, variant + '.jsonl')
    os.makedirs(results_dir, exist_ok=True)

    eval_dataset = build_tsb_ad_u_short_dataset(args.datadir)
    
    index_type = args.index
    use_image = True if 'vision' in variant.lower() else False
    use_deseasonality = True
    print(f'indextype:{index_type} use_image:{use_image} use_deseason:{use_deseasonality}')

    model = AnoAgent(data_name=data_name, llm_model=model_name, max_ts_len=2000, index_type=index_type, min_acf_period=24, value_scale=10)

    # Load existing results if jsonl file exists
    if os.path.exists(jsonl_fn):
        open(jsonl_fn, 'w').close()

    # Loop over image files
    for i in range(1, len(eval_dataset) + 1):
        cate = eval_dataset[i - 1][0]
        custom_id = f"{cate}_{model_name}_{variant}_{str(i).zfill(5)}"
        
        # Skip already processed files
        if custom_id in results:
            continue
        
        # Perform anomaly detection with exponential backoff
        for attempt in range(num_retries):
            try:
                
                
                series = eval_dataset[i - 1][2]
                ano_locs = np.array(eval_dataset[i - 1][1]) 
                
                
                pred_vector = model.inference(series[:, 0], anomaly_ratio=0.05, use_deseasonal=use_deseasonality, use_image=use_image, context=None)
                pred_vector = pred_vector.reshape(-1, 1).astype(bool).astype(int)

                gt = interval_to_vector(ano_locs[0], end=pred_vector.shape[0])
                sample_metrics = compute_metrics(gt, pred_vector)
                
                pred_intervals = vector_to_interval(pred_vector)
                
                # Write the result to jsonl
                with open(jsonl_fn, 'a') as f:
                    json.dump({'custom_id': custom_id, 'pred_intervals':pred_intervals, 'metrics':sample_metrics}, f)
                    f.write('\n')
                # If successful, break the retry loop
                break
            except Exception as e:
                if "503" in str(e):  # Server not up yet, sleep until the server is up again
                    while True:
                        logger.debug("503 error, sleep 30 seconds")
                        time.sleep(30)
                        try:
                            response = send_openai_request(request, model_name)
                            break
                        except Exception as e:
                            if "503" not in str(e):
                                break
                else:
                    logger.error(e)
                    # If an exception occurs, wait and then retry
                    wait_time = 2 ** (attempt + 3)
                    logger.debug(f"Attempt {attempt + 1} failed. Waiting for {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue
        else:
            logger.error(f"Failed to process {custom_id} after {num_retries} attempts")

            
def main():
    args = parse_arguments()
    online_AD_with_retries(
        model_name=args.model,
    )


if __name__ == '__main__':
    main()
