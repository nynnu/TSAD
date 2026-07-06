import pandas as pd
import numpy as np
import re
import json


def make_anomllm_prompt(target_df, use_vision, n_anomalies=5, max_digits=2, reduce_token=False):
    hist_time = target_df.index.strftime("%Y-%m-%d %H:%M:%S").values
    hist_value = target_df.values[:, -1]
    
    history = " ".join(
        f"{y:.{max_digits}f}"
        for x, y in zip(hist_time, hist_value)
    )

    if use_vision:
        prompt = f"""
Assume there are up to {n_anomalies} anomalies.

Detect ranges of anomalies in this time series, in terms of the x-axis coordinate.
List one by one, in JSON format. 
If there are no anomalies, answer with an empty list [].
"""
    else:
        prompt = f"""{history}

Assume there are up to {n_anomalies} anomalies.

Detect ranges of anomalies in this time series, in terms of the x-axis coordinate.
List one by one, in JSON format. 
If there are no anomalies, answer with an empty list [].
"""
    output_prompt ="""Output template:
[{"start": ..., "end": ...}, {"start": ..., "end": ...}...]
"""
    return prompt + output_prompt


def make_simple_prompt(target_df, use_vision, n_anomalies=5, max_digits=2, reduce_token=False):
    hist_time = target_df.index.strftime("%Y-%m-%d %H:%M:%S").values
    hist_value = target_df.values[:, -1]
    
    if reduce_token:
        history = "\n".join(
            f"{x}, {y:.{max_digits}f}"
        for x, y in zip(hist_time, hist_value)
        )
    else:
        history = "\n".join(
            f"({x}, {y:.{max_digits}f})"
        for x, y in zip(hist_time, hist_value)
        )

    if use_vision:
        prompt = f"""
I will provide you with time-series data recorded at hourly intervals, along with a plotted time-series image.

Here is time-series data in (timestamp, value) format:
<history>
{history}
</history>

Assume there are up to {n_anomalies} anomalies.

Detect ranges of anomalies in this time series, in terms of the timestamp of time-series data, considering the plotted image.
List one by one, in JSON format. 
If there are no anomalies, answer with an empty list []. Do not say anything other than the answer.
"""
    else:
        prompt = f"""
I will provide you with time-series data recorded at hourly intervals.

Here is time-series data in (timestamp, value) format:
<history>
{history}
</history>

Assume there are up to {n_anomalies} anomalies.

Detect ranges of anomalies in this time series, in terms of the timestamp of time-series data.
List one by one, in JSON format. 
If there are no anomalies, answer with an empty list []. Do not say anything other than the answer.
"""
    
    output_prompt ="""Output template:
[{"start timestamp": ..., "end timestamp": ...}, {"start timestamp": ..., "end timestamp": ...}...]
"""
    return prompt + output_prompt


def make_simple_wo_text_seq_prompt(target_df, use_vision, n_anomalies=5, max_digits=2, reduce_token=False):
    hist_time = target_df.index.strftime("%Y-%m-%d %H:%M:%S").values
    hist_value = target_df.values[:, -1]
    
    history = "\n".join(
        f"({index},)"
        for index, (x, y) in enumerate(zip(hist_time, hist_value))
    )

    if use_vision:
        prompt = f"""
I will provide you with time-series value data recorded at hourly intervals, along with a plotted time-series image.

Here is time-series data in (index, ) format:
<history>
{history}
</history>

Assume there are up to {n_anomalies} anomalies.

Detect ranges of anomalies in this time series, considering the plotted image.
The index of the time series starts from 0 to {len(hist_value)}. 
List one by one, in JSON format. 
If there are no anomalies, answer with an empty list []. Do not say anything other than the answer.
"""
    else:
        prompt = f"""
I will provide you with time-series value data recorded at hourly intervals.

Here is time-series data in (, value) format:
<history>
{history}
</history>

Assume there are up to {n_anomalies} anomalies.

Detect ranges of anomalies in this time series.
The index of the time series starts from 0 to {len(hist_value)}. 
List one by one, in JSON format. 
If there are no anomalies, answer with an empty list []. Do not say anything other than the answer.
"""        
    
    output_prompt ="""Output template:
[{"start": ..., "end": ...}, {"start": ..., "end": ...}...]
"""
    return prompt + output_prompt



def make_simple_wo_index_prompt(target_df, use_vision, n_anomalies=5, max_digits=2, reduce_token=False):
    hist_time = target_df.index.strftime("%Y-%m-%d %H:%M:%S").values
    hist_value = target_df.values[:, -1]
    
    history = "\n".join(
        f"(, {y:.{max_digits}f})"
    for x, y in zip(hist_time, hist_value)
    )

    if use_vision:
        prompt = f"""
I will provide you with time-series value data recorded at hourly intervals, along with a plotted time-series image.

Here is time-series data in (, value) format:
<history>
{history}
</history>

Assume there are up to {n_anomalies} anomalies.

Detect ranges of anomalies in this time series, considering the plotted image.
The index of the time series starts from 0 to {len(hist_value)}. 
List one by one, in JSON format. 
If there are no anomalies, answer with an empty list []. Do not say anything other than the answer.
"""
    else:
        prompt = f"""
I will provide you with time-series value data recorded at hourly intervals.

Here is time-series data in (, value) format:
<history>
{history}
</history>

Assume there are up to {n_anomalies} anomalies.

Detect ranges of anomalies in this time series.
The index of the time series starts from 0 to {len(hist_value)}. 
List one by one, in JSON format. 
If there are no anomalies, answer with an empty list []. Do not say anything other than the answer.
"""        
    
    output_prompt ="""Output template:
[{"start": ..., "end": ...}, {"start": ..., "end": ...}...]
"""
    return prompt + output_prompt


def make_simple_num_index_prompt(target_df, use_vision, n_anomalies=5, max_digits=2, reduce_token=False):
    hist_time = target_df.index.strftime("%Y-%m-%d %H:%M:%S").values
    hist_value = target_df.values[:, -1]
    
    if reduce_token:
        history = "\n".join(
            f"{index}, {y:.{max_digits}f}"
            for index, (x, y) in enumerate(zip(hist_time, hist_value))
        )
    else:
        history = "\n".join(
            f"({index}, {y:.{max_digits}f})"
        for index, (x, y) in enumerate(zip(hist_time, hist_value))
        )

    if use_vision:
        prompt = f"""
I will provide you with time-series value data recorded at hourly intervals, along with a plotted time-series image.

Here is time-series data in (index, value) format:
<history>
{history}
</history>

Assume there are up to {n_anomalies} anomalies.

Detect ranges of anomalies in this time series, considering the plotted image.
The index of the time series starts from 0 to {len(hist_value)}. 
List one by one, in JSON format. 
If there are no anomalies, answer with an empty list []. Do not say anything other than the answer.
"""
    else:
        prompt = f"""
I will provide you with time-series value data recorded at hourly intervals.

Here is time-series data in (index, value) format:
<history>
{history}
</history>

Assume there are up to {n_anomalies} anomalies.

Detect ranges of anomalies in this time series.
The index of the time series starts from 0 to {len(hist_value)}. 
List one by one, in JSON format. 
If there are no anomalies, answer with an empty list []. Do not say anything other than the answer.
"""        
    
    output_prompt ="""Output template:
[{"start": ..., "end": ...}, {"start": ..., "end": ...}...]
"""
    return prompt + output_prompt




def extract_timestamp_dicts(data: str) -> list:
    json_str = re.sub(r"```json", "", data)
    json_str = re.sub(r"```", "", json_str).strip()

    object_strs = re.findall(r'\{[^}]+\}', json_str)
    result = []
    for obj_str in object_strs:
        try:
            item = json.loads(obj_str)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error (skip): {e} - object: {obj_str}")
            continue

        start_ts_str = item.get("start timestamp")
        end_ts_str = item.get("end timestamp")
        try:
            start_ts = pd.Timestamp(start_ts_str) if start_ts_str else None
        except Exception as e:
            print(f"start timestamp: {start_ts_str} - error: {e}")
            start_ts = None

        try:
            end_ts = pd.Timestamp(end_ts_str) if end_ts_str else None
        except Exception as e:
            print(f"end timestamp: {end_ts_str} - error: {e}")
            end_ts = None

        if start_ts is not None and end_ts is not None:
            transformed_item = {
                "start_timestamp": start_ts,
                "end_timestamp": end_ts
            }
            result.append(transformed_item)
    
    return result