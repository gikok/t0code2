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

import datasets
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AutoConfig,
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
from promptsource.templates import DatasetTemplates


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
        data_files = {"train": args.dataset_name, "test": args.dataset_name}
        raw_train_dataset = load_dataset(
            "data", data_files=data_files, split="train"
        )
        raw_eval_dataset = load_dataset(
            "data", data_files=data_files, split="test"
        )
    else:
        raise ValueError(
            "Please specify `args.dataset_name`."
        )

    # Trim a number of evaluation training
    if args.debug:
        raw_train_dataset = raw_train_dataset.select(
            range(min(100, len(raw_train_dataset)))
        )
        raw_eval_dataset = raw_eval_dataset.select(
            range(min(100, len(raw_eval_dataset)))
        )

    column_names = raw_eval_dataset.column_names

    # Load pretrained model and tokenizer

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # get all item_no and add as tokens
    items = pd.read_parquet('../data/item_no_6k.gzip')['item_no'].values.tolist()
    tokenizer.add_tokens(items)

    # then resize embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    def tokenize_train(examples):
        input_texts = examples['input']
        target_texts = examples['target']

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
        input_texts = examples['input']
        target_texts = examples['target']
        answer_choices_texts = examples['options']

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
                padding=True,
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

        features["labels"] = [tokenized_targets[idx]["input_ids"] for idx in range(bs)]
        features["labels_attention_mask"] = [
            tokenized_targets[idx]["attention_mask"] for idx in range(bs)
        ]
        features["targets"] = [
            answer_choices_texts[idx].index(t)
            for idx, t in enumerate(target_texts)
        ]

        return features

    with accelerator.main_process_first():
        eval_dataset = raw_eval_dataset.map(
            preprocess_eval, batched=True, remove_columns=column_names
        )

        if args.num_shots is not None:
            sample_indices = random.sample(
                range(0, len(raw_train_dataset)), k=args.num_shots
            )
            raw_train_dataset = raw_train_dataset.select(sample_indices)
        train_dataset = raw_train_dataset.map(
            tokenize_train, batched=True, remove_columns=column_names
        )

    # Log a few random training:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.debug(f"Sample {index} of the training set: {train_dataset[index]}.")
    for index in random.sample(range(len(eval_dataset)), 3):
        logger.debug(f"Sample {index} of the evaluation set: {eval_dataset[index]}.")

    # DataLoaders creation:
    train_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=train_collator,
        batch_size=args.per_device_train_batch_size,
    )

    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        eval_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        eval_collator = DataCollatorForMultipleChoice(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=eval_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

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

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    if args.parallelize:
        num_gpus = torch.cuda.device_count()
        assert num_gpus > 1, "You need at least 2 GPUs to use `model.parallelize()`."
        model.parallelize()
        optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            optimizer, train_dataloader, eval_dataloader
        )
    else:
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

    # Metrics
    metric = load_metric("accuracy")

    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
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
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    global_steps = 0

    if args.wandb_proj and accelerator.is_main_process:
        import wandb

        extra_metadata = {
            "template_jinja": template.jinja,
            "template_answer_choices": template.answer_choices,
            "template_reflects_original_task": template.metadata.original_task,
            "template_choices_in_prompt": template.metadata.choices_in_prompt,
            "template_comment": template.reference,
        }
        run_config = vars(args)
        run_config.update(extra_metadata)
        wandb.init(
            project=args.wandb_proj,
            config=run_config,
            # name=f'S{len(train_set)} {args.template_name} R{args.seed}',  # uncomment to customize each run's name
            # reinit=True,  # uncomment if running multiple runs in one script
        )

    # freeze encoder updates if specified
    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if name.startswith("encoder"):
                param.requires_grad = False
    
    # how often trained model should be saved
    r = int(args.max_train_steps / 10)
    if args.gradient_checkpoint:
        model.gradient_checkpointing_enable()
    model_counter = 0
    for epoch in range(1, args.num_train_epochs + 1):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                global_steps += 1
                loss = loss.item()
                if accelerator.is_main_process:
                    tqdm.write(f"epoch = {epoch}, step = {global_steps}, loss = {loss}")
                if args.wandb_proj and accelerator.is_main_process:
                    wandb.log({"loss": loss}, step=global_steps)
                    
                    # log inference every 10 steps
                    if global_steps % 10 == 0:
                        inp = ["what is a cow?", "query is: steel fork for babies. what are the top 3 results?"]
                        inputs = tokenizer.batch_encode_plus(inp, return_tensors='pt', padding=True)
                        inputs = inputs.to("cuda:0")
                        with torch.no_grad():
                            outputs = model.generate(inputs['input_ids'])
                            result = [tokenizer.decode(outputs[0], skip_special_tokens=True), tokenizer.decode(outputs[1], skip_special_tokens=True)]
                            wandb.log({'standard inference': result[0]}, step=global_steps)
                            wandb.log({'trained inference': result[1]}, step=global_steps)

            # save model checkpoint ever r steps
            if global_steps % r == 0:
                model_counter += 1
                mc = str(model_counter).zfill(3)
                print("SAVING INTERMEDIATE MODEL")
                os.mkdir(args.output_dir+f'_'+mc)
                model.save_pretrained(args.output_dir+f'_'+mc)
                tokenizer.save_pretrained(args.output_dir+f'_'+mc)
            if global_steps >= args.max_train_steps:
                break

        # Evaluate every epoch
        total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes
        logger.info("***** Running evaluation *****")
        logger.info(f"  Num training = {len(eval_dataset)}")
        logger.info(
            f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}"
        )
        logger.info(
            f"  Total eval batch size (w. parallel, distributed) = {total_batch_size}"
        )
        # Only show the progress bar once on each machine.  # NOTE commented out to avoid nested pbar mess
        # progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)

    #     model.eval()
    #     checker = 0
    #     for batch in eval_dataloader:
    #         checker +=1
    #         print('eval iteration:',checker)
    #         model_inputs = {
    #             k: batch[k]
    #             for k in ["input_ids", "attention_mask", "labels"]
    #         }
    #         with torch.no_grad():
    #             logits = model(**model_inputs).logits
    #         masked_log_probs = batch["labels_attention_mask"].unsqueeze(-1) * torch.log_softmax(logits, dim=-1)
    #         seq_token_log_probs = torch.gather(masked_log_probs, -1, batch["labels"].unsqueeze(-1))
    #         seq_log_prob = seq_token_log_probs.squeeze(dim=-1).sum(dim=-1)
    #         seq_log_prob = seq_log_prob.view(batch["targets"].size(0), -1) #TODO(Victor): this reshapes works based on the assumption that all training have the same number of choices. the pre-processing doesn't make this assumption.
    #         predictions = seq_log_prob.argmax(dim=-1)

    #         metric.add_batch(
    #             predictions=accelerator.gather(predictions),
    #             references=accelerator.gather(batch["targets"]),
    #         )

    #         # progress_bar.update(1)

    #     eval_metric = metric.compute()
    #     score = eval_metric["accuracy"]  # TODO support other metrics; currently hardcoded at load_metric() anyway
    #     accelerator.print(f"Accuracy: {score}")
    #     result_table.append({
    #         "dataset_name": args.dataset_name,
    #         "dataset_config_name": args.dataset_config_name,
    #         "template_name": args.template_name,
    #         "epoch": epoch,
    #         "step": global_steps,
    #         "metric": 'accuracy',
    #         "score": score,
    #     })
    #     if args.wandb_proj and accelerator.is_main_process:
    #         wandb.log({"accuracy": score}, step=global_steps)
    # # End training loop
    os.mkdir(args.output_dir+'/final_model/')
    model.save_pretrained(args.output_dir+'/final_model/')
    tokenizer.save_pretrained(args.output_dir+'/final_model/')

    # if accelerator.is_main_process:
    #     if args.output_dir is not None:
    #         with open(os.path.join(args.output_dir, "results.csv"), "w") as f:
    #             writer = csv.DictWriter(f, fieldnames=result_table[0].keys())
    #             writer.writeheader()
    #             writer.writerows(result_table)

    # if args.wandb_proj:
    #     wandb.finish()






def add_or(s):
    li = s.rsplit(', ')
    start = ', '.join(li[:-1])
    end =  ' or ' + li[-1]
    return start+end
    

def create_option_string(df, column, inds):
    options = ""
    for i in inds:
        options +=  f"'{df[column].iloc[i]}', "
        if i == inds[-1]:
            options = options[:-2]
    options = add_or(options)
        
    return options

def get_other_indices(index, size, num):
    """
    given a number {index}, returns {num} integers in range({size})
    """
    
    proceed = 0
    while proceed == 0:
        inds = np.random.choice(range(size), num)
        # check if any of the new indices is same as initial
        if (inds == index).any()==False:
            proceed = 1
        else:
            proceed = 0
    
    return inds


def no_to_name(index):
    
    # get random number of options
    num = np.random.randint(1, 4)
    
    # get random item index
    inds = get_other_indices(index, len(df), num)
    inds = np.append(inds, index)
    np.random.shuffle(inds)
    
    # get answer options
    options = create_option_string(df, 'name', inds)
    
    # create the input string
    inp = f"if item_no is {df['item_no'].iloc[index]}, which of the following is the correct name: " + options + '?'
    
    
    # set the target
    target = df['name'].iloc[inds[np.argmin(inds-index)]]
    
    return inp.lower(), target.lower()

def name_to_no(index):
    
    # get random number of options
    num = np.random.randint(1, 4)
    
    # get random item index
    inds = get_other_indices(index, len(df), num)
    inds = np.append(inds, index)
    np.random.shuffle(inds)
    
    # get answer options
    options = create_option_string(df, 'item_no', inds)
    
    # get the input string
    inp = f"if name is {df['name'].iloc[index]}, what item_no does it refer to? " + options + '?' 
    
    # get the target
    target = df['item_no'].iloc[inds[np.argmin(inds-index)]]
    
    return inp.lower(), target.lower()

def is_description(index):
    
    is_true = np.random.randint(2)
    
    if is_true:
        desc = df['benefits'].iloc[index]
        target = "yes"
    else:
        tempdf = df[df['benefits']!=df['benefits'].iloc[index]]
        desc = np.random.choice(tempdf['benefits'])
        target = "no"
        
    # make sure ends with period    
    desc = desc if desc.endswith(".") else desc + "."
    
    # create input
    inp = desc + f" is the previous sentence a description of item_no {df['item_no'].iloc[index]}. yes or no?"
    
    return inp.lower(), target

def is_summary(index):
    
    is_true = np.random.randint(2)
    
    if is_true:
        desc = df['key_w'].iloc[index]
        target = "yes"
    else:
        tempdf = df[df['key_w']!=df['key_w'].iloc[index]]
        desc = np.random.choice(tempdf['key_w'])
        target = "no"
        
    # make sure ends with period    
    desc = desc if desc.endswith(".") else desc + "."
    
    # create input
    inp = desc + f" is the previous sentence a summary of item_no {df['item_no'].iloc[index]}. yes or no?"
    
    return inp.lower(), target

def true_query(query_df, item_no):
    
    is_true = np.random.randint(2)
    
    if is_true:
        query = query_df['clean_query'].sample().iloc[0]
        target = "yes"
    else:
        query = query_df['clean_query'].sample().iloc[0]
        target = "no"

    
    # create input
    inp = "query:'" + query + f"'\ndoes the query above return item_no {item_no} as a result. yes or no?"
    
    return inp.lower(), target

def query_rank(query_df, item_no):

    # select random row
    row = query_df.sample()
    
    # get query string
    query = row['clean_query'].iloc[0]
    
    # get rank of item_no
    cols = np.array(row.columns)
    inds = (row[row==item_no].isnull()==False).values[0]
    
    # get first true value, that is not 0 (queries can be for item_no) and
    # sometimes same item_no has multiple ranks
    if len(inds)>1:
        inds[0] = False
        rank = cols[np.argmax(inds)][-1]
    
    # in case rank is 10
    rank = '10' if rank=='0' else rank
    
    # create possible options
    n_options = np.random.randint(1, 5)
    
    options = list(np.arange(1, 11))
    options.remove(int(rank))
    
    options = np.append(np.random.choice(options, n_options).astype(str), rank)
    np.random.shuffle(options)
    
    # create input
    inp = "query:'" + query + f"'\nthe query above returns item_no {item_no} as a result. what is its rank? " + add_or(', '.join(options))
    
    target = rank
    
    return inp.lower(), rank




if __name__ == "__main__":
    main()
