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
    parser.add_argument('--variant', type=str, default='0shot-text-vision', help='Variant type')
    parser.add_argument('--model', type=str, default='gemini-1.5-flash', help='Model name')
    #'OpenGVLab/InternVL2-Llama3-76B' #'Qwen/Qwen-VL-Chat' #'OpenGVLab/InternVL2-Llama3-76B'# 'gpt-4o'# gemini-1.5-flash
    parser.add_argument('--data', type=str, default='point', help='Data name')
    return parser.parse_args()


def online_AD_with_retries(
    model_name: str,
    data_name: str,
    variant: str = "standard",
    num_retries: int = 4,
):
    import json
    import time
    import pickle
    import os
    from loguru import logger
    from data.synthetic import SyntheticDataset

    # Initialize dictionary to store results
    results = {}

    # Configure logger
    log_fn = f"logs/synthetic/{data_name}/{model_name}/" + variant + ".log"
    logger.add(log_fn, format="{time} {level} {message}", level="INFO")
    results_dir = f'results/synthetic/{data_name}/{model_name}/'
    data_dir = f'data/synthetic/{data_name}/eval/'
    train_dir = f'data/synthetic/{data_name}/train/'
    jsonl_fn = os.path.join(results_dir, variant + '.jsonl')
    os.makedirs(results_dir, exist_ok=True)

    eval_dataset = SyntheticDataset(data_dir)
    eval_dataset.load()

    train_dataset = SyntheticDataset(train_dir)
    train_dataset.load()
    
    if variant.lower() not in ['0shot-text', '0shot-text-vision']:
        raise Exception(f'Not supported varaint: {variant}')
    
    index_type = 'number' 
    use_image = True if 'vision' in variant.lower() else False
    use_deseasonality = True
    print(f'indextype:{index_type} use_image:{use_image} use_deseason:{use_deseasonality}')

    model = AnoAgent(data_name=data_name, llm_model=model_name, max_ts_len=2000, index_type=index_type, min_acf_period=24, value_scale=10)

    # Load existing results if jsonl file exists
    if os.path.exists(jsonl_fn):
        with open(jsonl_fn, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                results[entry['custom_id']] = entry["response"]

    # Loop over image files
    for i in range(1, len(eval_dataset) + 1):
        custom_id = f"{data_name}_{model_name}_{variant}_{str(i).zfill(5)}"
        
        # Skip already processed files
        if custom_id in results:
            continue
        
        # Perform anomaly detection with exponential backoff
        for attempt in range(num_retries):
            try:
                
                series = eval_dataset.series[i - 1]
                ano_locs = np.array(eval_dataset.anom[i - 1]) 
                
                pred_vector, request, response = model.sample_inference(series[:, 0], anomaly_ratio=0.005, use_deseasonal=use_deseasonality, use_image=use_image, context=None, return_all=True)
                pred_vector = pred_vector.reshape(-1, 1).astype(bool).astype(int)

                gt = interval_to_vector(ano_locs[0], end=pred_vector.shape[0])
                sample_metrics = compute_metrics(gt, pred_vector)
                
                pred_intervals = vector_to_interval(pred_vector)
                
                # Write the result to jsonl
                with open(jsonl_fn, 'a') as f:
                    json.dump({'custom_id': custom_id, 'request': request, 'response': response, 'pred_intervals':pred_intervals, 'metrics':sample_metrics}, f)
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
        data_name=args.data,
        variant=args.variant,
    )


if __name__ == '__main__':
    main()
