import os
import math
import json
import torch
import pickle
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from scipy.spatial.distance import cosine

from model import *
from utility.logger import *
from utility.metric import *
from utility.trainer import *
from datetime import datetime
from utility.load_data import *
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from transformers import AutoConfig, AutoModel, AutoTokenizer

save = True
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='hkuds/easyrec-roberta-large', help='Model name')
parser.add_argument('--cuda', type=str, default='0')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

model_name_or_path = args.model
print(model_name_or_path)
config = AutoConfig.from_pretrained(model_name_or_path)
model = Easyrec.from_pretrained(
    model_name_or_path,
    from_tf=bool(".ckpt" in model_name_or_path),
    config=config,             
).cuda()
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path,
    use_fast=False,
)

eval_datas = ['sports', 'steam', 'yelp']
diverse_profile_num = 3

for _dataset in eval_datas:
    save_path = f'./data/{_dataset}/text_emb'
    os.makedirs(save_path, exist_ok=True)
    
    # original profiles
    user_profile, item_profile = {}, {}
    user_profile_list, item_profile_list = [], []   
    with open(f'./data/{_dataset}/user_profile.json', 'r') as f:
        for _line in f.readlines():
            _data = json.loads(_line)
            user_profile[_data['user_id']] = _data['profile']
    with open(f'./data/{_dataset}/item_profile.json', 'r') as f:
        for _line in f.readlines():
            _data = json.loads(_line)
            item_profile[_data['item_id']] = _data['profile']
    
    for i in range(len(user_profile)):
        user_profile_list.append(user_profile[i])
    for i in range(len(item_profile)):
        item_profile_list.append(item_profile[i])
    
    profiles = user_profile_list + item_profile_list
    batch_size = 128
    n_batchs = math.ceil(len(profiles) / batch_size)
    text_emb = []
    for i in tqdm(range(n_batchs), desc=f'{_dataset}'):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_profile = profiles[start: end]
        inputs = tokenizer(batch_profile, padding=True, truncation=True, max_length=512, return_tensors="pt")
        for tem in inputs:
            inputs[tem] = inputs[tem].cuda()
        with torch.inference_mode():
            embeddings = model.encode(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        embeddings = F.normalize(embeddings.pooler_output.detach().float(), dim=-1)
        text_emb.append(embeddings.cpu())
    text_emb = torch.concat(text_emb, dim=0)
    user_emb = text_emb[: len(user_profile)].numpy()
    item_emb = text_emb[len(user_profile): ].numpy()
    
    if save:
        with open(f'{save_path}/user_{model_name_or_path.split("/")[-1]}.pkl', 'wb') as f:
            pickle.dump(user_emb, f)
        with open(f'{save_path}/item_{model_name_or_path.split("/")[-1]}.pkl', 'wb') as f:
            pickle.dump(item_emb, f)
    
    # diversified profiles
    os.makedirs(f'{save_path}/diverse_profile', exist_ok=True)
    for diverse_no in range(diverse_profile_num):
        user_profile, item_profile = {}, {}
        user_profile_list, item_profile_list = [], [] 
        with open(f'./data/{_dataset}/diverse_profile/diverse_user_profile_{diverse_no}.json', 'r') as f:
            for _line in f.readlines():
                _data = json.loads(_line)
                user_profile[_data['user_id']] = _data['profile']
        with open(f'./data/{_dataset}/diverse_profile/diverse_item_profile_{diverse_no}.json', 'r') as f:
            for _line in f.readlines():
                _data = json.loads(_line)
                item_profile[_data['item_id']] = _data['profile']
        
        for i in range(len(user_profile)):
            user_profile_list.append(user_profile[i])
        for i in range(len(item_profile)):
            item_profile_list.append(item_profile[i])
        
        profiles = user_profile_list + item_profile_list
        batch_size = 128
        n_batchs = math.ceil(len(profiles) / batch_size)
        text_emb = []
        for i in tqdm(range(n_batchs), desc=f'diverse_{_dataset}_{diverse_no}'):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch_profile = profiles[start: end]
            inputs = tokenizer(batch_profile, padding=True, truncation=True, max_length=512, return_tensors="pt")
            for tem in inputs:
                inputs[tem] = inputs[tem].cuda()
            with torch.inference_mode():
                embeddings = model.encode(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            embeddings = F.normalize(embeddings.pooler_output.detach().float(), dim=-1)
            text_emb.append(embeddings.cpu())
        text_emb = torch.concat(text_emb, dim=0)
        user_emb = text_emb[: len(user_profile)].numpy()
        item_emb = text_emb[len(user_profile): ].numpy()
        
        if save:
            with open(f'{save_path}/diverse_profile/user_{model_name_or_path.split("/")[-1]}_{diverse_no}.pkl', 'wb') as f:
                pickle.dump(user_emb, f)
            with open(f'{save_path}/diverse_profile/item_{model_name_or_path.split("/")[-1]}_{diverse_no}.pkl', 'wb') as f:
                pickle.dump(item_emb, f)