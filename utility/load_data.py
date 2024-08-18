import os
import json
import torch
import pickle
import random
import logging
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import PreTrainedTokenizer
from typing import List, Optional, Tuple, Dict
from torch.utils.data import Dataset, DataLoader

IGNORE_INDEX=-100

class LazyPretrainEmbedderDataset(Dataset):
    """Dataset for Supervised Fine-tuning. (Representation Learning)"""
    def __init__(self, data_path:str, dataset:str, tokenizer:PreTrainedTokenizer, max_seq_length:int, used_diverse_profile_num:int, add_item_raw_meta:bool, **kawrgs, ):
        self.data_path = data_path
        dataset = dataset.split('-')
        self.max_seq_length = max_seq_length
        self.n_user, self.n_item = {}, {}
        self.trn_mat, self.dok_trn_mat = {}, {}
        self.user_profile, self.item_profile = {}, {}
        self.u_items = {}
        trn_users, which_data = [], []
        for _dataset in dataset:
            self.u_items[_dataset] = {}
            # training and validation data
            with open(f'{self.data_path}/{_dataset}/trn_mat.pkl', 'rb') as f:
                self.trn_mat[_dataset] = pickle.load(f)
                self.n_user[_dataset] = self.trn_mat[_dataset].shape[0]
                self.n_item[_dataset] = self.trn_mat[_dataset].shape[1]
                self.dok_trn_mat[_dataset] = self.trn_mat[_dataset].todok()
            
            # prepare sample list
            trn_users += list(range(self.n_user[_dataset]))
            which_data += [_dataset] * self.n_user[_dataset]
            
            # record interactions
            _row, _col = self.trn_mat[_dataset].row, self.trn_mat[_dataset].col
            for i in range(len(_row)):
                uid = _row[i]
                iid = _col[i]
                if uid not in self.u_items[_dataset]:
                    self.u_items[_dataset][uid] = [iid]
                else:
                    self.u_items[_dataset][uid].append(iid)
            
            # text description of users/items
            with open(f'{self.data_path}/{_dataset}/user_profile.json', 'r') as f:
                self.user_profile[_dataset] = {}
                for _line in f.readlines():
                    _profile = json.loads(_line)
                    self.user_profile[_dataset][_profile['user_id']] = [_profile['profile']]
            with open(f'{self.data_path}/{_dataset}/item_profile.json', 'r') as f:
                self.item_profile[_dataset] = {}
                for _line in f.readlines():
                    _profile = json.loads(_line)
                    self.item_profile[_dataset][_profile['item_id']] = [_profile['profile']]
            
            # diverse description of users/items
            for i in range(used_diverse_profile_num):
                with open(f'{self.data_path}/{_dataset}/diverse_profile/diverse_user_profile_{i}.json', 'r') as f:
                    for _line in f.readlines():
                        _profile = json.loads(_line)
                        self.user_profile[_dataset][_profile['user_id']].append(_profile['profile'])
                with open(f'{self.data_path}/{_dataset}/diverse_profile/diverse_item_profile_{i}.json', 'r') as f:
                    for _line in f.readlines():
                        _profile = json.loads(_line)
                        self.item_profile[_dataset][_profile['item_id']].append(_profile['profile'])
            
            # add metadata-based text description
            if add_item_raw_meta:
                with open(f'{self.data_path}/{_dataset}/diverse_profile/item_raw_meta_profile.json', 'r') as f:
                    for _line in f.readlines():
                        _profile = json.loads(_line)
                        self.item_profile[_dataset][_profile['item_id']].append(_profile['profile'])
                        
        
        # shuffle
        combined = list(zip(trn_users, which_data))
        random.shuffle(combined)
        self.trn_users, self.which_data = zip(*combined)
        
        # tokenizer
        self.tokenizer = tokenizer
    
    def _get_sentence_input(self, idx):
        u = self.trn_users[idx]
        _dataset = self.which_data[idx]
        i_pos = random.choice(self.u_items[_dataset][u])
        while True:
            i_neg = np.random.randint(self.n_item[_dataset])
            if (u, i_neg) not in self.dok_trn_mat[_dataset]:
                break
        # print(_dataset, u, i_pos, i_neg)
        u_profile = random.choice(self.user_profile[_dataset][u])
        i_pos_profile = random.choice(self.item_profile[_dataset][i_pos])
        i_neg_profile = random.choice(self.item_profile[_dataset][i_neg])
        
        return {
            "user_profile": u_profile,
            "positive_item_profile": i_pos_profile,
            "negative_item_profile": i_neg_profile
        }
        
    def _preprocess(self, sources):
        # tokenize user profile
        user_profile = sources['user_profile']
        user_input_ids = self.tokenizer(
            user_profile, 
            return_tensors='pt', 
            padding="longest",
            max_length=self.max_seq_length,
            truncation=True,
        ).input_ids
        
        # tokenize positive item profile
        pos_item_profile = sources['positive_item_profile']
        pos_item_input_ids = self.tokenizer(
            pos_item_profile, 
            return_tensors='pt', 
            padding="longest",
            max_length=self.max_seq_length,
            truncation=True,
        ).input_ids
        
        # tokenize negative item profile
        neg_item_profile = sources['negative_item_profile']
        neg_item_input_ids = self.tokenizer(
            neg_item_profile, 
            return_tensors='pt', 
            padding="longest",
            max_length=self.max_seq_length,
            truncation=True,
        ).input_ids
        
        return {
            'user_input_ids': user_input_ids[0],
            'pos_item_input_ids': pos_item_input_ids[0],
            'neg_item_input_ids': neg_item_input_ids[0],
        }   

    def __len__(self):
        return len(self.trn_users)

    def __getitem__(self, idx):
        # to sentences
        sources = self._get_sentence_input(idx)
        # tokenize
        data_dict = self._preprocess(sources)
        return data_dict


class PretrainEmbedderAllRankTestDataset(Dataset):
    def __init__(self, data_path:str, dataset_name:str, diverse_profile_no=None, **kawrgs, ):
        self.data_path = data_path
        self.dataset_name = dataset_name
        with open(f'{self.data_path}/{self.dataset_name}/trn_mat.pkl', 'rb') as f:
            self.trn_mat = pickle.load(f)
            self.n_user = self.trn_mat.shape[0]
            self.n_item = self.trn_mat.shape[1]
            self.csrmat = (self.trn_mat.tocsr() != 0) * 1.0
        # For validation, load valiation data
        with open(f'{self.data_path}/{self.dataset_name}/val_mat.pkl', 'rb') as f:
            self.val_mat = pickle.load(f)
        user_pos_lists = [list() for uid in range(self.n_user)]
        test_users = set()
        for i in range(len(self.val_mat.data)):
            uid = self.val_mat.row[i]
            iid = self.val_mat.col[i]
            user_pos_lists[uid].append(iid)
            test_users.add(uid)
        self.test_users = np.array(list(test_users))
        self.user_pos_lists = user_pos_lists
        
        self.user_profile, self.item_profile = {}, {}
        self.user_profile_list, self.item_profile_list = [], []
        
        if diverse_profile_no == None:
            with open(f'{self.data_path}/{self.dataset_name}/user_profile.json', 'r') as f:
                for _line in f.readlines():
                    _profile = json.loads(_line)
                    self.user_profile[_profile['user_id']] = _profile['profile']
            with open(f'{self.data_path}/{self.dataset_name}/item_profile.json', 'r') as f:
                for _line in f.readlines():
                    _profile = json.loads(_line)
                    self.item_profile[_profile['item_id']] = _profile['profile']
        else:
            with open(f'{self.data_path}/{self.dataset_name}/diverse_profile/diverse_user_profile_{diverse_profile_no}.json', 'r') as f:
                for _line in f.readlines():
                    _profile = json.loads(_line)
                    self.user_profile[_profile['user_id']] = _profile['profile']
            with open(f'{self.data_path}/{self.dataset_name}/diverse_profile/diverse_item_profile_{diverse_profile_no}.json', 'r') as f:
                for _line in f.readlines():
                    _profile = json.loads(_line)
                    self.item_profile[_profile['item_id']] = _profile['profile']
        
        for uid in range(self.n_user):
            self.user_profile_list.append(self.user_profile[uid])
        for iid in range(self.n_item):
            self.item_profile_list.append(self.item_profile[iid])
            
        if diverse_profile_no is not None:
            self.dataset_name = f'{self.dataset_name}_diverse_{diverse_profile_no}'
    
    def __len__(self):
        return len(self.test_users)
    
    def __getitem__(self, idx):
        pck_user = self.test_users[idx]
        pck_mask = self.csrmat[pck_user].toarray()
        pck_mask = np.reshape(pck_mask, [-1])
        return pck_user, pck_mask


class EvalEmbedderAllRankTestDataset(Dataset):
    def __init__(self, data_path:str, dataset_name:str, diverse_profile_no=None, **kawrgs, ):
        self.data_path = data_path
        self.dataset_name = dataset_name
        with open(f'{self.data_path}/{self.dataset_name}/trn_mat.pkl', 'rb') as f:
            self.trn_mat = pickle.load(f)
            self.n_user = self.trn_mat.shape[0]
            self.n_item = self.trn_mat.shape[1]
            self.csrmat = (self.trn_mat.tocsr() != 0) * 1.0
        # For testing, load test data
        with open(f'{self.data_path}/{self.dataset_name}/tst_mat.pkl', 'rb') as f:
            self.tst_mat = pickle.load(f)
        user_pos_lists = [list() for uid in range(self.n_user)]
        test_users = set()
        for i in range(len(self.tst_mat.data)):
            uid = self.tst_mat.row[i]
            iid = self.tst_mat.col[i]
            user_pos_lists[uid].append(iid)
            test_users.add(uid)
        self.test_users = np.array(list(test_users))
        self.user_pos_lists = user_pos_lists
        
        self.user_profile, self.item_profile = {}, {}
        self.user_profile_list, self.item_profile_list = [], []
        
        if diverse_profile_no == None:
            with open(f'{self.data_path}/{self.dataset_name}/user_profile.json', 'r') as f:
                for _line in f.readlines():
                    _profile = json.loads(_line)
                    self.user_profile[_profile['user_id']] = _profile['profile']
            with open(f'{self.data_path}/{self.dataset_name}/item_profile.json', 'r') as f:
                for _line in f.readlines():
                    _profile = json.loads(_line)
                    self.item_profile[_profile['item_id']] = _profile['profile']
        else:
            with open(f'{self.data_path}/{self.dataset_name}/diverse_profile/diverse_user_profile_{diverse_profile_no}.json', 'r') as f:
                for _line in f.readlines():
                    _profile = json.loads(_line)
                    self.user_profile[_profile['user_id']] = _profile['profile']
            with open(f'{self.data_path}/{self.dataset_name}/diverse_profile/diverse_item_profile_{diverse_profile_no}.json', 'r') as f:
                for _line in f.readlines():
                    _profile = json.loads(_line)
                    self.item_profile[_profile['item_id']] = _profile['profile']
        
        for uid in range(self.n_user):
            self.user_profile_list.append(self.user_profile[uid])
        for iid in range(self.n_item):
            self.item_profile_list.append(self.item_profile[iid])
            
        if diverse_profile_no is not None:
            self.dataset_name = f'{self.dataset_name}_diverse_{diverse_profile_no}'
    
    def __len__(self):
        return len(self.test_users)
    
    def __getitem__(self, idx):
        pck_user = self.test_users[idx]
        pck_mask = self.csrmat[pck_user].toarray()
        pck_mask = np.reshape(pck_mask, [-1])
        return pck_user, pck_mask
    
    
@dataclass
class DataCollatorForPretrainEmbedderDataset(object):
    """Collate examples for supervised fine-tuning. (Representation Learning)"""  
    
    mlm_probability: float
    tokenizer: PreTrainedTokenizer
    
    def _mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        inputs = inputs.clone()
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = IGNORE_INDEX  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def __call__(self, instances):
        user_input_ids, pos_item_input_ids, neg_item_input_ids, mlm_input_ids = [], [], [], []
        for instance in instances:
            user_input_ids.append(instance['user_input_ids'])
            pos_item_input_ids.append(instance['pos_item_input_ids'])
            neg_item_input_ids.append(instance['neg_item_input_ids'])
            mlm_input_ids.append(instance['user_input_ids'].clone())
            mlm_input_ids.append(instance['pos_item_input_ids'].clone())
            mlm_input_ids.append(instance['neg_item_input_ids'].clone())
            
        user_input_ids = torch.nn.utils.rnn.pad_sequence(
            user_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        pos_item_input_ids = torch.nn.utils.rnn.pad_sequence(
            pos_item_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        neg_item_input_ids = torch.nn.utils.rnn.pad_sequence(
            neg_item_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        mlm_input_ids = torch.nn.utils.rnn.pad_sequence(
            mlm_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        
        mlm_input_ids, mlm_labels = self._mask_tokens(mlm_input_ids)
        
        batch = dict(
            user_input_ids=user_input_ids,
            user_attention_mask=user_input_ids.ne(self.tokenizer.pad_token_id),
            pos_item_input_ids=pos_item_input_ids,
            pos_item_attention_mask=pos_item_input_ids.ne(self.tokenizer.pad_token_id),
            neg_item_input_ids=neg_item_input_ids,
            neg_item_attention_mask=neg_item_input_ids.ne(self.tokenizer.pad_token_id),
            mlm_input_ids=mlm_input_ids,
            mlm_attention_mask=mlm_input_ids.ne(self.tokenizer.pad_token_id),
            mlm_labels=mlm_labels,
        )
        return batch


def make_pretrain_embedder_supervised_data_module(tokenizer:PreTrainedTokenizer, data_args):
    """Make dataset and collator for supervised fine-tuning. (Representation Learning)"""
    train_dataset = LazyPretrainEmbedderDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        dataset=data_args.trn_dataset,
        max_seq_length=data_args.max_seq_length,
        used_diverse_profile_num=data_args.used_diverse_profile_num,
        add_item_raw_meta=data_args.add_item_raw_meta,
    )
    
    eval_dataset = {}
    for _dataset in data_args.val_dataset.split('-'):
        eval_dataset[f'{_dataset}'] = PretrainEmbedderAllRankTestDataset(
            data_path=data_args.data_path,
            dataset_name=_dataset,
        )
        for i in range(data_args.total_diverse_profile_num):
            eval_dataset[f'{_dataset}_diverse_{i}'] = PretrainEmbedderAllRankTestDataset(
                data_path=data_args.data_path,
                dataset_name=_dataset,
                diverse_profile_no=i,
            )
    
    data_collator = DataCollatorForPretrainEmbedderDataset(mlm_probability=data_args.mlm_probability, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)