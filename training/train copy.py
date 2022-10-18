#!/usr/bin/env python
# coding=utf-8
# Copyright BigScience, The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning T0 in PyTorch, optionally few-shot.

This script is adapted from
https://github.com/huggingface/transformers/blob/master/examples/pytorch/multiple-choice/run_swag_no_trainer.py
as well as
https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization_no_trainer.py
"""

import argparse
from lib2to3.pgen2 import token
import logging
import os
import random
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import pandas as pd
import numpy as np
import math
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import datasets
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    default_data_collator,
    DataCollatorForSeq2Seq,
    AdamW,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import PaddingStrategy


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tuning T0 in PyTorch, optionally few-shot."
    )
    parser.add_argument(
        "-d",
        "--dataset_name",
        type=str,
        default=None,
        required=True,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "-en",
        "--eval_name",
        type=str,
        default=None,
        required=True,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Where to store the results CSV and (TODO) optionally the final model.",
    )
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        required=True,
        help=(
            "Path to pretrained model or model identifier from huggingface.co/models. "
            "The list of T0 variants can be found on `https://huggingface.co/bigscience/T0_3B`"
        ),
    )
    parser.add_argument(
        "-pa",
        "--parallelize",
        action="store_true",
        help=(
            "If passed, will call `model.parallelize` which splits the model on all GPUs available (model parallelism). "
            "Note that this feature is still experimental in HF Transformers."
        ),
    )
    parser.add_argument(
        "-eb",
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader. Will be multiplied by the number of answer choices.",
    )
    parser.add_argument(
        "-tb",
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "-ns",
        "--num_shots",
        type=int,
        default=None,
        help="Number of training training for few-shot learning. Default is None, which uses the entire train set.",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "-ep",
        "--num_train_epochs",
        type=int,
        default=10,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "-fe",
        "--freeze_encoder",
        action="store_true",
        help="If enabled the encoder layers will be excluded from model training.",
    )
    parser.add_argument(
        "-gc",
        "--gradient_checkpoint",
        action="store_true",
        help="If enabled model will train with gradient checkpointing, reducing GPU memory usage",
    )
    parser.add_argument(
        "-ms",
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "-ga",
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "-ie",
        "--input_eos",
        action="store_true",
        help=(
            "T0 was trained without EOS in its input sequences, which is the default in this script."
            "However, T5 was pretrained with EOS in its input sequences. See README for more info."
        ),
    )
    parser.add_argument(
        "-db",
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "-wb",
        "--wandb_proj",
        type=str,
        default=None,
        help="Project name for Weights & Biases. By default, W&B is disabled.",
    )
    parser.add_argument(
        "-sd",
        "--seed",
        type=int,
        default=42,
        help="Especially important for few-shot example sampling.",
    )
    parser.add_argument(
        "-tk",
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "-il",
        "--max_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "-tl",
        "--target_max_length",
        type=int,
        default=256,
        help="Target max length. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "-pml",
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "-st",
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for the AdamW optimizer.",
    )
    parser.add_argument(
        "-ls",
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "-ws",
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    args = parser.parse_args()

    return args


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
            Note that it's very NOT recommended to use fp16 to do any time of inference with T0 as the predictions will vastly differ from the predictions using fp32.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [
                {k: v[i] for k, v in feature.items() if k != "targets"}
                for i in range(num_choices)
            ]
            for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Pad the labels because it's not padded automatically
        max_label_length = max([len(elem["labels"]) for elem in flattened_features])
        batch["labels"] = [
            l + [self.tokenizer.pad_token_id] * (max_label_length - len(l))
            for l in [elem["labels"] for elem in flattened_features]
        ]
        batch["labels_attention_mask"] = [
            m + [0] * (max_label_length - len(m))
            for m in [elem["labels_attention_mask"] for elem in flattened_features]
        ]

        # Convert to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}

        batch["targets"] = torch.tensor([f.pop("targets") for f in features])
        return batch


def main():
    args = parse_args()
    set_seed(args.seed)

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Handle the output directory creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # In distributed evaluation, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        data_files = {"train": args.dataset_name, "test": args.eval_name}
        raw_train_dataset = load_dataset("data", data_files=data_files, split="train")
        # raw_test_dataset = load_dataset("data", data_files=data_files, split="test")
        raw_eval_dataset = load_dataset("data", data_files=data_files, split="test")
    else:
        raise ValueError("Please specify `args.dataset_name`.")

    column_names = raw_eval_dataset.column_names

    # Load pretrained model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # # # get all item_no and add as tokens
    items = pd.read_parquet("data/item_no_100.parquet.gzip")["item_no"].values.tolist()
    tokenizer.add_tokens(items)

    # then resize embeddings
    model.resize_token_embeddings(len(tokenizer))

    # fix initial values of new matrices
    def resample(model, layer, n_new):

        new_tensor = list(model.named_parameters())[layer][1].detach().cpu()

        for i in range(len(new_tensor[-1,:])):
            val, bin = np.histogram(new_tensor[:-n_new, i], 10000)
            pdf = val / sum(val)
            cdf = np.cumsum(pdf)
            b = (bin[1:] + bin[:-1]) / 2
            new_tensor[-n_new:, i] = torch.tensor(
                np.interp(np.random.random(n_new), cdf, b)
            )
        data = list(model.named_parameters())[layer][1].data
        data[:, :] = new_tensor

    resample(model, 0, len(items))
    resample(model, -1, len(items))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    def tokenize_train(examples):
        input_texts = examples["input"]
        target_texts = examples["target"]

        model_inputs = tokenizer(
            input_texts,
            padding=padding,
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=args.input_eos,
        )

        with tokenizer.as_target_tokenizer():
            tokenized_targets = tokenizer(
                target_texts,
                padding=padding,
                max_length=args.target_max_length,
                truncation=True,
                add_special_tokens=False,
            )
            model_inputs["labels"] = [
                [(t if t != tokenizer.pad_token_id else -100) for t in targets]
                for targets in tokenized_targets["input_ids"]
            ]
        return model_inputs

    def preprocess_eval(examples):
        input_texts = examples["input"]
        target_texts = examples["target"]
        answer_choices_texts = examples["options"]

        tokenized_inputs = tokenizer(
            input_texts,
            padding=padding,
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = [
            tokenizer(
                ans_choi,
                padding=padding,
                max_length=args.target_max_length,
                truncation=True,
            )
            for ans_choi in answer_choices_texts
        ]

        features = {
            k: [
                [elem for _ in range(len(tokenized_targets[idx]["input_ids"]))]
                for idx, elem in enumerate(v)
            ]
            for k, v in tokenized_inputs.items()
        }
        bs = len(examples[column_names[0]])
        features["labels"] = [tokenized_targets[idx]["input_ids"] for idx in range(bs)]
        features["labels_attention_mask"] = [
            tokenized_targets[idx]["attention_mask"] for idx in range(bs)
        ]
        features["targets"] = [
            answer_choices_texts[idx].index(t) for idx, t in enumerate(target_texts)
        ]

        return features

    with accelerator.main_process_first():
        train_dataset = raw_train_dataset.map(tokenize_train, batched=True)
        train_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        test_dataset = raw_train_dataset.map(tokenize_train, batched=True)
        test_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        #    train_dataset.save_to_disk("data/tokenized")

    # DataLoaders creation:
    train_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=train_collator,
        batch_size=args.per_device_train_batch_size,
    )

    test_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=test_collator,
        batch_size=args.per_device_train_batch_size,
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    if args.parallelize:
        num_gpus = torch.cuda.device_count()
        assert num_gpus > 1, "You need at least 2 GPUs to use `model.parallelize()`."
        model.parallelize()
        (
            train_dataloader,
            test_dataloader,
        ) = accelerator.prepare(train_dataloader, test_dataloader)
    else:
        model, train_dataloader, test_dataloader = accelerator.prepare(
            model, train_dataloader, test_dataloader
        )

    total_batch_size = (
        args.per_device_train_batch_size
        * (accelerator.num_processes)
        * args.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num training = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")


    if args.gradient_checkpoint:
        model.gradient_checkpointing_enable()

    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if name.startswith("encoder") or name.startswith("decoder"):
                param.requires_grad = False
            if name.startswith("shared") or name.startswith("lm_head"):
                grad_mask = torch.ones_like(param)
                grad_mask[: (len(tokenizer) - len(items)), :] = 0
                param.register_hook(lambda grad: grad * grad_mask)

    # train model using pytorch-lightning API
    plmodel = plModelClass(model)
    checkpoint_callback = ModelCheckpoint(every_n_epochs=100, dirpath=args.output_dir, filename='model_epoch_{epoch:02d}', save_last=True)
    trainer = pl.Trainer(accelerator="gpu", min_epochs=args.num_train_epochs, devices=-1, auto_select_gpus=True, accumulate_grad_batches=args.gradient_accumulation_steps, strategy="deepspeed_stage_2", callbacks=[checkpoint_callback])
    trainer.fit(model=plmodel, train_dataloaders=train_dataloader)


    os.mkdir(args.output_dir + "/final_model/")
    model.save_pretrained(args.output_dir + "/final_model/")
    tokenizer.save_pretrained(args.output_dir + "/final_model/")


    # define the LightningModule
    class plModelClass(pl.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.learning_rate = args.learning_rate

        def training_step(self, batch, batch_idx):
            # training_step defines the train loop.
            # it is independent of forward
            outputs = self.model(**batch)
            loss = outputs.loss
            # Logging to TensorBoard by default
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

            return optimizer


if __name__ == "__main__":
    main()
