import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import re
import json
import math
import glob
import base64
import torch

from io import BytesIO
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta

from utils import parse_output

from neurips_our.preprocessing_seq import find_period_autocorr, map_to_timestamps, preprocessing, seq2image
from neurips_our.prompts import make_simple_prompt, make_simple_wo_text_seq_prompt, make_simple_wo_index_prompt, make_simple_num_index_prompt, extract_timestamp_dicts, make_anomllm_prompt

from openai_api import send_openai_request

class AnoAgent:
    def __init__(self, data_name, llm_model, index_type=None, max_ts_len=1000, min_acf_period=24, value_scale=10):
        self.data_name = data_name
        self.llm_model = llm_model
        self.max_ts_len = max_ts_len
        self.min_acf_period = min_acf_period
        self.value_scale = value_scale
        self.index_type = index_type
        self.reduce_token = False
        self.max_digits = 2
        
        if self.llm_model in ['gpt-4o', 'gpt-4o-mini']:
            self.send_request = send_openai_request
            self.make_request = self.make_openai_request
        elif self.llm_model in ['gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-flash-latest', 'gemini-2.5-flash-preview-04-17']:
            self.send_request = send_openai_request
            self.make_request = self.make_openai_request
        elif self.llm_model in ['OpenGVLab/InternVL2-Llama3-76B', 
                                'Qwen/Qwen-VL-Chat', 
                                'Qwen/Qwen2.5-VL-3B-Instruct', 
                                'Qwen/Qwen2.5-VL-72B-Instruct']:
            self.send_request = send_openai_request
            self.make_request = self.make_openai_request
        else:
            raise Exception(f'Unsupported LLM: {self.llm_model}')
            
        if data_name in ['trend', 'freq', 'point', 'range']:
            self.data_min_max = (-1, 1)
        else:
            self.data_min_max = None
            
    def send_qwen_request(self, messages, model_name=None):
        text = self.qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.qwen_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.qwen_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
            
    def make_openai_request(self, user_prompt, image_url=None):
        if image_url is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_url}"}
                        },
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        }
                    ],
                }
            ]
        
        return {
            "messages": messages,
            "temperature": 0.4,
            "stop": ["’’’’", " – –", "<|endoftext|>", "<|eot_id|>"]
        }
    
    def make_qwen_request(self, user_prompt, image_url=None):
        if image_url is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                        {
                            "type": "image",
                            "image": f"data:image/jpeg;base64,{image_url}"
                        },
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        }
                    ],
                }
            ]
        
        return messages
    
    
    def sample_get_prompt_and_response(self, sample_ts, anomaly_ratio=0.01, use_deseasonal=False, use_image=False, context=None):
        acf_period = find_period_autocorr(sample_ts)
        acf_period = self.min_acf_period if acf_period < self.min_acf_period else acf_period
        acf_period = self.min_acf_period if acf_period > len(sample_ts)//2 else acf_period

#         print(self.data_min_max)
        input_seq_df, decomposed_df = preprocessing(sample_ts, acf_period, self.value_scale, data_min_max=self.data_min_max)
        sample_ts_len = len(input_seq_df)
        n_anomaly = int(sample_ts_len * anomaly_ratio)
        n_anomaly = 3 if n_anomaly < 3 else n_anomaly
        
        prompt_seq_df = decomposed_df.copy() if use_deseasonal else input_seq_df.copy()
        
        if self.data_min_max == None:
            figsize = (20, 3)
            seqimg_base64 = seq2image(prompt_seq_df, index_type=self.index_type, figsize=figsize) if use_image else None
        else:
            seqimg_base64 = seq2image(prompt_seq_df, index_type=self.index_type, data_min_max=(0, self.value_scale)) if use_image else None
        
        use_vision = seqimg_base64 is not None
         
        if self.index_type == 'timestamp':
            text_prompt = make_simple_prompt(prompt_seq_df, use_vision, n_anomalies=n_anomaly,  max_digits=self.max_digits, reduce_token=self.reduce_token)
            request = self.make_request(text_prompt, seqimg_base64)
            response = self.send_request(request, self.llm_model)
            try:
                pred_vector = self.extract_range_with_timestamp(input_seq_df, response)
            except Exception as e:
                print(e)
                pred_vector = np.zeros(len(prompt_seq_df))
        elif self.index_type == 'number':
            text_prompt = make_simple_num_index_prompt(prompt_seq_df, use_vision, n_anomalies=n_anomaly,  max_digits=self.max_digits, reduce_token=self.reduce_token)
            # print(text_prompt)
            request = self.make_request(text_prompt, seqimg_base64)
            response = self.send_request(request, self.llm_model)
            # print('hi', response)
            try:
                pred_vector = self.extract_range_with_index(input_seq_df, response)
            except Exception as e:
                print(e)
                pred_vector = np.zeros(len(prompt_seq_df))
        
        elif self.index_type == 'wo-index':
            text_prompt = make_simple_wo_index_prompt(prompt_seq_df, use_vision, n_anomalies=n_anomaly,  max_digits=self.max_digits, reduce_token=self.reduce_token)
            request = self.make_request(text_prompt, seqimg_base64)
            response = self.send_request(request, self.llm_model)
            try:
                pred_vector = self.extract_range_with_index(input_seq_df, response)
            except Exception as e:
                print(e)
                pred_vector = np.zeros(len(prompt_seq_df))
        
        elif self.index_type == 'wo-text-seq':
            text_prompt = make_simple_wo_text_seq_prompt(prompt_seq_df, use_vision, n_anomalies=n_anomaly,  max_digits=self.max_digits, reduce_token=self.reduce_token)
            request = self.make_request(text_prompt, seqimg_base64)
            response = self.send_request(request, self.llm_model)
            try:
                pred_vector = self.extract_range_with_index(input_seq_df, response)
            except Exception as e:
                print(e)
                pred_vector = np.zeros(len(prompt_seq_df))
                
        elif self.index_type == 'anomllm':
            text_prompt = make_anomllm_prompt(prompt_seq_df, use_vision, n_anomalies=n_anomaly,  max_digits=self.max_digits, reduce_token=self.reduce_token)
            request = self.make_request(text_prompt, seqimg_base64)
            response = self.send_request(request, self.llm_model)
            try:
                pred_vector = self.extract_range_with_index(input_seq_df, response)
            except Exception as e:
                print(e)
                pred_vector = np.zeros(len(prompt_seq_df))
        else:
            raise Exception(f'Unsupported index type: {self.index_type}')
            
        return pred_vector, request, response

        
    
    def sample_inference(self, sample_ts, anomaly_ratio=0.01, use_deseasonal=False, use_image=False, context=None, return_all=False):
        pred_vector, request, response = self.sample_get_prompt_and_response(sample_ts, anomaly_ratio=anomaly_ratio, use_deseasonal=use_deseasonal, use_image=use_image, context=context)
        if return_all:
            return pred_vector, request, response
        else:
            return pred_vector
    
    
    def inference(self, ts, anomaly_ratio=0.01, use_deseasonal=False, use_image=False, context=None):
        ts_len = len(ts)
        pred = np.zeros(ts_len)
        if ts_len > self.max_ts_len:
            for st in range(0, ts_len, self.max_ts_len):
                sample_ed = st + self.max_ts_len
                if sample_ed > ts_len:
                    sample_st = ts_len - self.max_ts_len
                    sample_ed = ts_len
                else:
                    sample_st = st
                print(sample_st, sample_ed)
                sample_ts = ts[sample_st:sample_ed]
                pred_vector = self.sample_inference(sample_ts, anomaly_ratio, use_deseasonal, use_image, context, return_all=False)
                pred[sample_st:sample_ed] = pred[sample_st:sample_ed] + pred_vector
        else:
            pred_vector = self.sample_inference(ts, anomaly_ratio, use_deseasonal, use_image, context)
            pred = pred_vector
            
        return pred
    
    
    def batch_inference(self):
        pass
    
    def extract_range_with_timestamp(self, input_df, response):
        target_dates = extract_timestamp_dicts(response)
        temp_df = input_df.reset_index()
        pred_vector = np.zeros(len(input_df))
        for region in target_dates:
            region_vec = ((temp_df.date >= region['start_timestamp']) & (temp_df.date <= region['end_timestamp'])).values.astype(int)
            pred_vector = pred_vector + region_vec
        
        return pred_vector
    
    def extract_range_with_index(self, input_df, response):
        intervals = parse_output(response)
        temp_df = input_df.reset_index()
        pred_vector = np.zeros(len(input_df))
        for ad in intervals:
            st = ad['start']
            ed = ad['end'] + 1 if ad['start'] == ad['end'] else ad['end']
            pred_vector[st:ed] = 1
            
        return pred_vector