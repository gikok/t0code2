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
from dataclasses import dataclass
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim import AdamW

import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    set_seed,
)

from helpers.data_processing import resample, clean_train

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
        required=False,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "-ckpt",
        "--ckpt_directory",
        type=str,
        default=None,
        required=False,
        help="The name of checkpoint from which to resume the training.",
    )
    parser.add_argument(
        "-cf",
        "--ckpt_freq",
        type=int,
        default=1,
        required=False,
        help="Number of epochs between each checkpoint",
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
        "-fm",
        "--freeze_model",
        action="store_true",
        help="If enabled the encoder/decoder layers will be excluded from model training.",
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



def main():
    args = parse_args()
    set_seed(args.seed)

    # Handle the output directory creation
    os.makedirs(args.output_dir, exist_ok=True)

    # In distributed evaluation, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # data_files = {"train": , "test": args.eval_name}
        raw_train_dataset = load_dataset(
            "data", data_files=args.dataset_name, split="train"
        )
        # raw_test_dataset = load_dataset("data", data_files=data_files, split="test")
        # raw_eval_dataset = load_dataset("data", data_files=data_files, split="test")
    else:
        raise ValueError("Please specify `args.dataset_name`.")


    # Load pretrained model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # # # get all item_no and add as tokens
    items = pd.read_parquet("data/seven_names.parquet.gzip")["item_no"].values.tolist()
    items += ["item_no"]
    tokenizer.add_tokens(items)

    # then resize embeddings
    model.resize_token_embeddings(len(tokenizer))

    # fix initial values of new matrices
    resample(model, 0, len(items))
    resample(model, -1, len(items))

    # Preprocessing the datasets.
    train_dataset = clean_train(raw_train_dataset, tokenizer, args)

    # DataLoaders creation:
    train_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        persistent_workers=True,
        collate_fn=train_collator,
        num_workers=1,
        batch_size=args.per_device_train_batch_size,
    )

    # only train the new parts of the model, freeze non-vocab layers and put grad=0 for pre-existing tokens
    if args.freeze_model:
        for name, param in model.named_parameters():
            if name.startswith("encoder") or name.startswith("decoder"):
                param.requires_grad = False
            if name.startswith("shared") or name.startswith("lm_head"):
                grad_mask = torch.ones_like(param)
                grad_mask[: (len(tokenizer) - len(items)), :] = 0
                param.register_hook(
                    lambda grad: grad * grad_mask.to(f"cuda:{grad.get_device()}")
                )

    # train model using pytorch-lightning API
    plmodel = plModelClass(model, args)
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=args.ckpt_freq,
        dirpath=args.output_dir,
        filename="model_epoch_{epoch:02d}",
        save_last=True,
    )
    trainer = pl.Trainer(
        logger=True,
        accelerator="gpu",
        min_epochs=args.num_train_epochs,
        max_epochs=2 * args.num_train_epochs,
        devices=-1,
        auto_select_gpus=True,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        strategy="deepspeed_stage_2",
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model=plmodel,
        train_dataloaders=train_dataloader,
        ckpt_path=args.ckpt_directory,
    )
    if not os.path.exists(args.output_dir + "/final_model/"):
        os.mkdir(args.output_dir + "/final_model/")
    model.save_pretrained(args.output_dir + "/final_model/")
    tokenizer.save_pretrained(args.output_dir + "/final_model/")


# define the LightningModule
class plModelClass(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
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

        optimizer = AdamW(
            [
                list(self.model.named_parameters())[0][1],
                list(self.model.named_parameters())[-1][1],
            ],
            lr=self.learning_rate,
            weight_decay=0,
        )

        return optimizer


if __name__ == "__main__":
    main()
