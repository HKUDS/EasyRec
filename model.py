import math
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import scipy.sparse as sp
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

from utility.loss_utils import *

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


"""
EasyRec
"""
def dot_product_scores(q_vectors, ctx_vectors):
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Easyrec(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    
    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        try:
            self.model_args = model_kargs["model_args"]
            self.roberta = RobertaModel(config, add_pooling_layer=False)
            if self.model_args.pooler_type == "cls":
                self.mlp = MLPLayer(config)
            if self.model_args.do_mlm:
                self.lm_head = RobertaLMHead(config)
            """
            Contrastive learning class init function.
            """
            self.pooler_type = self.model_args.pooler_type
            self.pooler = Pooler(self.pooler_type)
            self.sim = Similarity(temp=self.model_args.temp)
            self.init_weights()
        except:
            self.roberta = RobertaModel(config, add_pooling_layer=False)
            self.mlp = MLPLayer(config)
            self.lm_head = RobertaLMHead(config)
            self.pooler_type = 'cls'
            self.pooler = Pooler(self.pooler_type)
            self.init_weights()
    
    def forward(self,
        user_input_ids=None,
        user_attention_mask=None,
        pos_item_input_ids=None,
        pos_item_attention_mask=None,
        neg_item_input_ids=None,
        neg_item_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mlm_input_ids=None,
        mlm_attention_mask=None,
        mlm_labels=None,
    ):
        """
        Contrastive learning forward function.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = user_input_ids.size(0)
        
        # Get user embeddings
        user_outputs = self.roberta(
            input_ids=user_input_ids,
            attention_mask=user_attention_mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Get positive item embeddings
        pos_item_outputs = self.roberta(
            input_ids=pos_item_input_ids,
            attention_mask=pos_item_attention_mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Get negative item embeddings
        neg_item_outputs = self.roberta(
            input_ids=neg_item_input_ids,
            attention_mask=neg_item_attention_mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # MLM auxiliary objective
        if mlm_input_ids is not None:
            mlm_outputs = self.roberta(
                input_ids=mlm_input_ids,
                attention_mask=mlm_attention_mask,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        # Pooling
        user_pooler_output = self.pooler(user_attention_mask, user_outputs)
        pos_item_pooler_output = self.pooler(pos_item_attention_mask, pos_item_outputs)
        neg_item_pooler_output = self.pooler(neg_item_attention_mask, neg_item_outputs)
        
        # If using "cls", we add an extra MLP layer
        # (same as BERT's original implementation) over the representation.
        if self.pooler_type == "cls":
            user_pooler_output = self.mlp(user_pooler_output)
            pos_item_pooler_output = self.mlp(pos_item_pooler_output)
            neg_item_pooler_output = self.mlp(neg_item_pooler_output)
        
        # Gather all item embeddings if using distributed training
        if dist.is_initialized() and self.training:
            # Dummy vectors for allgather
            user_list = [torch.zeros_like(user_pooler_output) for _ in range(dist.get_world_size())]
            pos_item_list = [torch.zeros_like(pos_item_pooler_output) for _ in range(dist.get_world_size())]
            neg_item_list = [torch.zeros_like(neg_item_pooler_output) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=user_list, tensor=user_pooler_output.contiguous())
            dist.all_gather(tensor_list=pos_item_list, tensor=pos_item_pooler_output.contiguous())
            dist.all_gather(tensor_list=neg_item_list, tensor=neg_item_pooler_output.contiguous())
            
            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            user_list[dist.get_rank()] = user_pooler_output
            pos_item_list[dist.get_rank()] = pos_item_pooler_output
            neg_item_list[dist.get_rank()] = neg_item_pooler_output
            
            # Get full batch embeddings
            user_pooler_output = torch.cat(user_list, dim=0)
            pos_item_pooler_output = torch.cat(pos_item_list, dim=0)
            neg_item_pooler_output = torch.cat(neg_item_list, dim=0)
        
        cos_sim = self.sim(user_pooler_output.unsqueeze(1), pos_item_pooler_output.unsqueeze(0))
        neg_sim = self.sim(user_pooler_output.unsqueeze(1), neg_item_pooler_output.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, neg_sim], 1)
        
        labels = torch.arange(cos_sim.size(0)).long().to(self.device)
        loss_fct = nn.CrossEntropyLoss()
        
        loss = loss_fct(cos_sim, labels)
        
        # Calculate loss for MLM
        if mlm_outputs is not None and mlm_labels is not None and self.model_args.do_mlm:
            mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
            prediction_scores = self.lm_head(mlm_outputs.last_hidden_state)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
            loss = loss + self.model_args.mlm_weight * masked_lm_loss
        
        if not return_dict:
            raise NotImplementedError
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=cos_sim,
        )
    
    def encode(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,    
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooler_output = self.pooler(attention_mask, outputs)
        if self.pooler_type == "cls":
            pooler_output = self.mlp(pooler_output)
        if not return_dict:
            return (outputs[0], pooler_output) + outputs[2:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )
    
    def inference(self,
        user_profile_list,
        item_profile_list,
        dataset_name,
        tokenizer,
        infer_batch_size=128
    ):
        n_user = len(user_profile_list)
        profiles = user_profile_list + item_profile_list
        n_batch = math.ceil(len(profiles) / infer_batch_size)
        text_embeds = []
        for i in tqdm(range(n_batch), desc=f'Encoding Text {dataset_name}'):
            batch_profiles = profiles[i * infer_batch_size: (i + 1) * infer_batch_size]
            inputs = tokenizer(batch_profiles, padding=True, truncation=True, max_length=512, return_tensors="pt")
            for k in inputs:
                inputs[k] = inputs[k].to(self.device)
            with torch.inference_mode():
                embeds = self.encode(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask
                )
            text_embeds.append(embeds.pooler_output.detach().cpu())
        text_embeds = torch.concat(text_embeds, dim=0).cuda()
        user_embeds = F.normalize(text_embeds[: n_user], dim=-1)
        item_embeds = F.normalize(text_embeds[n_user: ], dim=-1)
        return user_embeds, item_embeds
