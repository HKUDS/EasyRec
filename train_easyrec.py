import os
import torch
import random
import logging
import numpy as np
import transformers

seed=2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from model import Easyrec
from utility.logger import *
from utility.metric import *
from utility.trainer import *
from datetime import datetime
from utility.load_data import *
from transformers import AutoConfig
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    
    # SimCSE's arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    )
    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

@dataclass
class DataArguments:
    data_path: str = field(default='data/', metadata={"help": "Path to the training data."})
    trn_dataset: str = field(default='arts-games-movies-home-electronics-tools', metadata={"help": "Training data."})
    val_dataset: str = field(default='arts-games-movies-home-electronics-tools', metadata={"help": "Validation data."})
    used_diverse_profile_num: int = field(default=3)
    total_diverse_profile_num: int = field(default=3)
    add_item_raw_meta: bool = field(default=True, metadata={"help": "Whether to use raw item meta information or not."})
    
    # SimCSE's arguments
    max_seq_length: Optional[int] = field(
        default=64,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )

def main():
    global local_rank
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    print("training_args.output_dir", training_args.output_dir)
    
    ## logger
    logger = EasyrecEmbedderTrainingLogger(
        model_args=model_args, 
        data_args=data_args, 
        training_args=training_args,
    )
    logger.log(model_args)
    logger.log(data_args)
    logger.log(training_args)
    
    ## load model
    if 'roberta' in model_args.model_name_or_path:
        config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        }
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
        model = Easyrec.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            model_args=model_args,
            torch_dtype=torch.bfloat16,              
        )
    else:
        raise NotImplementedError
    
    ## tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        padding_side="right",
        use_fast=False,
    )
    
    ## data module
    data_module = make_pretrain_embedder_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    ## trainer
    trainer = EasyrecEmbedderTrainer(model=model,
                                      tokenizer=tokenizer,
                                      args=training_args,
                                      **data_module)
    metric = Metric(metrics=['recall'], k=[20])
    trainer.add_evaluator(metric)
    trainer.add_logger(logger)
    ## training
    trainer.train()

if __name__ == "__main__":
    main()