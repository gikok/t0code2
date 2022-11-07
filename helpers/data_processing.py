import numpy as np
import pandas as pd
from itertools import chain

import torch
from tqdm.auto import tqdm


def resample(model, tokenizer, items, layer, n_new, init_type="random"):

    new_tensor = list(model.named_parameters())[layer][1].detach().cpu()

    if init_type == "random":
        for i in range(len(new_tensor[-1, :])):
            val, bin = np.histogram(new_tensor[:-n_new, i], 10000)
            pdf = val / sum(val)
            cdf = np.cumsum(pdf)
            b = (bin[1:] + bin[:-1]) / 2
            new_tensor[-n_new:, i] = torch.tensor(
                np.interp(np.random.random(n_new), cdf, b)
            )
    if init_type == "targeted":
        search_names = pd.read_parquet("data/item_no_to_query.parquet.gzip")
        for i in range(n_new):
            ranks = search_names[search_names["item_no"] == items[i]][l]
            ranks = ranks[ranks != "PLACEHOLDER"].dropna(axis=1)
            rank_tokens = [tokenizer.encode(r)[:-1] for r in ranks.values.tolist()[0]]
            all_tokens = list(chain.from_iterable(rank_tokens))
            tensor_list = [new_tensor[t, :] * 1 for t in all_tokens]
            temp_tensor = torch.stack(tensor_list)
            new_tensor[-n_new + i, :] = torch.mean(temp_tensor, axis=0)
    data = list(model.named_parameters())[layer][1].data
    data[:, :] = new_tensor


def clean_train(raw_train_dataset, tokenizer, args):
    def tokenize_train(examples):
        input_texts = examples["input"]
        target_texts = examples["target"]
        padding = "max_length" if args.pad_to_max_length else False

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

    train_dataset = raw_train_dataset.map(tokenize_train, batched=True)
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    return train_dataset
