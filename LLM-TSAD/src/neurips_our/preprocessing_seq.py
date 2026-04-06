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

from prompt import time_series_to_image
from utils import view_base64_image, display_messages, collect_results, compute_metrics, interval_to_vector, plot_series_and_predictions, plot_series_and_predictions_with_timestamp


def scale_time_series(ts, scale_value, data_min_max=None):
    ts = np.array(ts)

    if data_min_max == None:
        min_val = np.min(ts)
        max_val = np.max(ts)
    else:
        min_val = data_min_max[0]
        max_val = data_min_max[1]
    

    if max_val == min_val:
        return np.full(ts.shape, scale_value / 2)

    ts_norm = (ts - min_val) / (max_val - min_val)
    ts_scaled = ts_norm * scale_value
    return ts_scaled


def preprocessing(raw_input, patch_len, scale, decompose=True, data_min_max=None):
    input_len = len(raw_input)
    timestamp_series = map_to_timestamps(N=input_len, P=patch_len)
    scaled_ts = scale_time_series(raw_input, scale, data_min_max)

    input_df = pd.DataFrame({"date": timestamp_series, "value": scaled_ts})
    input_df = input_df.set_index("date")
    
    if decompose:
        decompose_result = seasonal_decompose(input_df, model="additive", period=patch_len)
        decomposed_df = pd.DataFrame(input_df.value - decompose_result.seasonal, index=decompose_result.trend.index, columns=['value'])
    else:
        decomposed_df = None
    
    return input_df, decomposed_df


def find_period_autocorr(seq, min_lag=2):
    seq = np.array(seq)
    seq_detrended = seq - np.mean(seq)
    n = len(seq_detrended)
    
    autocorr = np.correlate(seq_detrended, seq_detrended, mode='full')[n-1:]
    
    peaks, _ = find_peaks(autocorr[min_lag:])
    if len(peaks) == 0:
        return 0

    peaks = peaks + min_lag
    best_peak = peaks[np.argmax(autocorr[peaks])]
    
    if np.max(autocorr[peaks]) < autocorr[0]//5:
        return 0
    return best_peak

def find_period_fft(seq):
    seq = np.array(seq)
    n = len(seq)
    seq_detrended = seq - np.mean(seq)
    
    fft_vals = np.fft.rfft(seq_detrended)
    power = np.abs(fft_vals) ** 2
    
    power[0] = 0
    
    peak_idx = np.argmax(power)
    if peak_idx == 0:
        return None
    period = n / peak_idx
    return period

def map_to_timestamps(N, P, start=None):
    if start is None:
        start_time = datetime(2024, 4, 1, 0, 0, 0)
        start = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    
    interval = timedelta(seconds=86400 / P)
    
    timestamps = [(start + i * interval).replace(microsecond=0) for i in range(N)]
    return timestamps


def seq2image(seq_df, gt_locs=None, predictions=None, index_type='number', data_min_max=None, figsize=(10, 1.5)):
    deseasonal_series = seq_df.values
    deseasonal_series.shape
    d_min = deseasonal_series.min()
    d_max = deseasonal_series.max()
    if data_min_max is not None:
        d_min = data_min_max[0] if d_min > data_min_max[0] else d_min
        d_max = data_min_max[1] if d_max < data_min_max[1] else d_max
    
    if index_type == 'number':
        fig = plot_series_and_predictions(
            series=deseasonal_series,
            single_series_figsize = figsize,
            gt_anomaly_intervals=gt_locs,
            gt_ylim = (d_min, d_max),
            anomalies=predictions
        )
    elif index_type == 'timestamp':
        fig = plot_series_and_predictions_with_timestamp(
            series=deseasonal_series, 
            single_series_figsize = figsize,
            gt_anomaly_intervals=gt_locs,
            gt_ylim = (d_min, d_max),
            anomalies=predictions,
            timestamps=seq_df.index.tolist()
        )
    else:
        fig = plot_series_and_predictions(
            series=deseasonal_series, 
            single_series_figsize = figsize,
            gt_anomaly_intervals=gt_locs,
            gt_ylim = (d_min, d_max),
            anomalies=predictions
        )
    
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return img_base64