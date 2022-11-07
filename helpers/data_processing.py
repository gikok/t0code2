import numpy as np

import torch
from tqdm.auto import tqdm


def resample(model, layer, n_new, init_type='random'):

    new_tensor = list(model.named_parameters())[layer][1].detach().cpu()

    if init_type=='random':
        for i in range(len(new_tensor[-1, :])):
            val, bin = np.histogram(new_tensor[:-n_new, i], 10000)
            pdf = val / sum(val)
            cdf = np.cumsum(pdf)
            b = (bin[1:] + bin[:-1]) / 2
            new_tensor[-n_new:, i] = torch.tensor(
                np.interp(np.random.random(n_new), cdf, b)
            )
    if init_type=='targeted':
        print("test")
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

