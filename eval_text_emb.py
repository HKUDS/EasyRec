import os
import torch
import random
import pickle
import logging
import argparse
import numpy as np
import transformers

seed=2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from model import *
from utility.logger import *
from utility.metric import *
from utility.trainer import *
from datetime import datetime
from utility.load_data import *
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from transformers import AutoConfig, AutoModel, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='easyrec-roberta-large', help='Model name')
parser.add_argument('--cuda', type=str, default='0')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

eval_dataset_list = ['sports', 'steam', 'yelp']
diverse_profile_num = 3
eval_dataset = {}
for _dataset in eval_dataset_list:
    eval_dataset[_dataset] = EvalEmbedderAllRankTestDataset(
        data_path='./data/',
        dataset_name=_dataset,
    )

model_name = args.model
print(model_name)
overall_results = {}
evaluator = Metric(k=[10, 20])

for eval_dataset_name, _eval_dataset in eval_dataset.items():
    dataset_name = eval_dataset_name
    eval_dataloader = data.DataLoader(_eval_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    # read text embeddings
    with open(f'data/{dataset_name}/text_emb/user_{model_name}.pkl', 'rb') as f:
        user_embeds = pickle.load(f)
        user_embeds = torch.tensor(user_embeds)
    with open(f'data/{dataset_name}/text_emb/item_{model_name}.pkl', 'rb') as f:
        item_embeds = pickle.load(f)
        item_embeds = torch.tensor(item_embeds)
    
    eval_result = evaluator.eval_w_embeds(user_embeds, item_embeds, eval_dataloader)
    for i in range(len(evaluator.k)):
        _k = evaluator.k[i]
        if f'{dataset_name}_recall@{_k}' not in overall_results:
            overall_results[f'{dataset_name}_recall@{_k}'] = []
        overall_results[f'{dataset_name}_recall@{_k}'].append(eval_result['recall'][i])
    for i in range(len(evaluator.k)):
        _k = evaluator.k[i]
        if f'{dataset_name}_ndcg@{_k}' not in overall_results:
            overall_results[f'{dataset_name}_ndcg@{_k}'] = []
        overall_results[f'{dataset_name}_ndcg@{_k}'].append(eval_result['ndcg'][i])
        
    # diverse profile
    for diverse_no in range(diverse_profile_num):
        with open(f'data/{dataset_name}/text_emb/diverse_profile/user_{model_name}_{diverse_no}.pkl', 'rb') as f:
            user_embeds = pickle.load(f)
            user_embeds = torch.tensor(user_embeds)
        with open(f'data/{dataset_name}/text_emb/diverse_profile//item_{model_name}_{diverse_no}.pkl', 'rb') as f:
            item_embeds = pickle.load(f)
            item_embeds = torch.tensor(item_embeds)
        
        eval_result = evaluator.eval_w_embeds(user_embeds, item_embeds, eval_dataloader)
        for i in range(len(evaluator.k)):
            _k = evaluator.k[i]
            if f'{dataset_name}_recall@{_k}' not in overall_results:
                overall_results[f'{dataset_name}_recall@{_k}'] = []
            overall_results[f'{dataset_name}_recall@{_k}'].append(eval_result['recall'][i])
        for i in range(len(evaluator.k)):
            _k = evaluator.k[i]
            if f'{dataset_name}_ndcg@{_k}' not in overall_results:
                overall_results[f'{dataset_name}_ndcg@{_k}'] = []
            overall_results[f'{dataset_name}_ndcg@{_k}'].append(eval_result['ndcg'][i])

for _key in overall_results:
    overall_results[_key] = sum(overall_results[_key]) / len(overall_results[_key])
message = ''
message += '['
for metric in overall_results:
    message += '{}: {:.4f} '.format(metric, overall_results[metric])
message += ']'

print(message)