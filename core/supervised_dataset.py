import copy
import logging
from dataclasses import dataclass
from typing import Dict, Sequence
import json
import torch
import transformers
from torch.utils.data import Dataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|pad|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
DEFAULT_UNK_TOKEN = "<|unk|>"


def fmt_prompt(prompt):
    return f"[Instructions]:\n{prompt}\n\n[Response]:"


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


def load_json(files, limit):
    data = []

    for file in files:
        f = open(file, mode="r")
        if "jsonl" in file:
            _data = f.read()
            _data = _data.splitlines()
            _data = [json.loads(js) for js in _data]
        else:
            _data = json.load(f)

        f.close()

        data += _data
        logging.info(f"loaded data file: {file}, size: {len(data)}")

    if limit > 0:
        return data[:limit]

    return data


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_path: list[str],
        limit=-1,
    ):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = load_json(data_path, limit)

        logging.warning("Formatting inputs...")

        sources = [
            f"{fmt_prompt(example['instruction'])}"
            for example in list_data_dict
        ]

        targets = [
            f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
