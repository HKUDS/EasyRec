import os
import json
import math
import torch
from tqdm import tqdm
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, AutoModel

# Initializing a RoBERTa configuration
configuration = RobertaConfig(num_hidden_layers=6, layer_norm_eps=1e-5, max_position_embeddings=514, type_vocab_size=1)
# Initializing a model (with random weights) from the configuration
small_model = RobertaModel(configuration).cuda()
# Accessing the model configuration
configuration = small_model.config
# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('./baseline_embedders/roberta-base')
# Load the Base model
base_model = AutoModel.from_pretrained("./baseline_embedders/roberta-base").cuda()
# Copy the weight of Base model
small_model.load_state_dict(base_model.state_dict(), strict=False)

test_work = True

if test_work:
    eval_datas = ['sports', 'steam', 'yelp']
    diverse_profile_num = 3

    for _dataset in eval_datas:
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
            inputs = tokenizer(batch_profile, padding=True, truncation=True, return_tensors="pt")
            for tem in inputs:
                inputs[tem] = inputs[tem].cuda()
            with torch.no_grad():
                embeddings = small_model(**inputs, output_hidden_states=True, return_dict=True).last_hidden_state[:, 0]
            embeddings = F.normalize(embeddings, dim=-1)
            text_emb.append(embeddings.cpu())
        text_emb = torch.concat(text_emb, dim=0)
        user_emb = text_emb[: len(user_profile)].numpy()
        item_emb = text_emb[len(user_profile): ].numpy()
        
        # diversified profiles
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
                inputs = tokenizer(batch_profile, padding=True, truncation=True, return_tensors="pt")
                for tem in inputs:
                    inputs[tem] = inputs[tem].cuda()
                with torch.no_grad():
                    embeddings = small_model(**inputs, output_hidden_states=True, return_dict=True).last_hidden_state[:, 0]
                embeddings = F.normalize(embeddings, dim=-1)
                text_emb.append(embeddings.cpu())
            text_emb = torch.concat(text_emb, dim=0)
            user_emb = text_emb[: len(user_profile)].numpy()
            item_emb = text_emb[len(user_profile): ].numpy()
        
        # item raw meta profile
        if _dataset in ['sports']:
            item_profile = {}
            item_profile_list = []
            with open(f'./data/{_dataset}/diverse_profile/item_raw_meta_profile.json', 'r') as f:
                for _line in f.readlines():
                    _data = json.loads(_line)
                    item_profile[_data['item_id']] = _data['profile']
            
            for i in range(len(item_profile)):
                item_profile_list.append(item_profile[i])
            
            profiles = item_profile_list
            batch_size = 128
            n_batchs = math.ceil(len(profiles) / batch_size)
            text_emb = []
            for i in tqdm(range(n_batchs), desc=f'item_{_dataset}_rawmeta'):
                start = i * batch_size
                end = (i + 1) * batch_size
                batch_profile = profiles[start: end]
                inputs = tokenizer(batch_profile, padding=True, truncation=True, return_tensors="pt")
                for tem in inputs:
                    inputs[tem] = inputs[tem].cuda()
                with torch.no_grad():
                    embeddings = small_model(**inputs, output_hidden_states=True, return_dict=True).last_hidden_state[:, 0]
                embeddings = F.normalize(embeddings, dim=-1)
                text_emb.append(embeddings.cpu())
            text_emb = torch.concat(text_emb, dim=0)
            item_emb = text_emb

# Save the model
small_model.save_pretrained('./baseline_embedders/roberta-small')
# Save the configuration
configuration.save_pretrained('./baseline_embedders/roberta-small')
# Save the tokenizer
tokenizer.save_pretrained('./baseline_embedders/roberta-small')