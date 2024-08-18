import collections
import inspect
import math
import sys
import os
import re
import json
import shutil
import time
import warnings
from pathlib import Path
import importlib.util
from packaging import version
import torch.utils.data as data
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import logging
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
)
from transformers.file_utils import (
    is_apex_available,
    is_datasets_available,
)

from transformers.utils import logging
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

from transformers.optimization import Adafactor, AdamW, get_scheduler
import copy

import numpy as np
from datetime import datetime
from filelock import FileLock

logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

class EasyrecEmbedderTrainer(Trainer):

    def add_evaluator(self, evaluator):
        self.evaluator = evaluator
        
    def add_logger(self, logger):
        self.logger = logger
    
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self.model.eval()
    
        # handle multipe eval datasets
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            # self.log(metrics)
            self.logger.log_eval(metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        
        # encoding users/items text embeddings
        user_embeds, item_embeds = self.model.inference(
            user_profile_list=eval_dataset.user_profile_list,
            item_profile_list=eval_dataset.item_profile_list,
            dataset_name=eval_dataset.dataset_name,
            tokenizer=self.tokenizer,
        )
        
        # process evaluation
        eval_dataloader = data.DataLoader(eval_dataset, batch_size=256, shuffle=False, num_workers=0)
        eval_result = self.evaluator.eval_w_embeds(user_embeds, item_embeds, eval_dataloader)
        metrics = {}
        for i in range(len(self.evaluator.k)):
            _k = self.evaluator.k[i]
            metrics[f'{metric_key_prefix}_recall@{_k}'] = eval_result['recall'][i]
        return metrics

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Compared to original implementation, we change the saving policy to
        only save the best-validation checkpoints.
        """
        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            metric_value = []
            for _key in metrics:
                if _key.endswith(metric_to_check):
                    metric_value.append(metrics[_key])
            mean_metric_value = np.mean(metric_value)

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(mean_metric_value, self.state.best_metric)
            ):
                self.logger.log(f"mean_metric_value: {mean_metric_value}")
                self.logger.log(f"Better Results! Now Saving...")
                output_dir = self.args.output_dir
                self.state.best_metric = mean_metric_value
                self.state.best_model_checkpoint = output_dir

                self.save_model(output_dir, _internal_call=True)

                if not self.args.save_only_model:
                    # Save optimizer and scheduler
                    self._save_optimizer_and_scheduler(output_dir)
                    # Save RNG state
                    self._save_rng_state(output_dir)
                    
                # Save the Trainer state
                if self.args.should_save:
                    self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

                if self.args.push_to_hub:
                    self._push_from_checkpoint(output_dir)
        else:
            # Save model checkpoint
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self.save_model(output_dir, _internal_call=True)

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Save the Trainer state
            if self.args.should_save:
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                # Solely rely on numerical checkpoint id for rotation.
                # mtime is not reliable especially on some fuse fs in cloud environments.
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)