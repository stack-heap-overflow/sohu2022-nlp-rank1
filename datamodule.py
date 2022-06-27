import json
import os
import pickle
import random

# import roformer
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import transformers
from tqdm import tqdm


class BasicDataset(Dataset):
    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

class DataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = self.valid_dataset = self.test_dataset = None
        if self.hparams.model_type == "xlnet":
            self.tokenizer = transformers.XLNetTokenizerFast.from_pretrained(self.hparams.model_name)
            self.mask_symbol = "<mask>"
        elif self.hparams.model_type == "roformer":
            self.tokenizer = transformers.BertTokenizerFast.from_pretrained(self.hparams.model_name)
            self.mask_symbol = "[MASK][PAD]"
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.hparams.model_name)
            self.mask_symbol = "[MASK]，"

    def _setup(self, data):
        output = []
        for item in tqdm(data):
            output_item = {}
            text = item["content"]
            if not text or not item["entity"]:
                continue
            prompt = "".join([f"{entity}{self.mask_symbol}" for entity in item["entity"]])
            inputs = self.tokenizer.__call__(text=text, text_pair=prompt, add_special_tokens=True, max_length=self.hparams.max_length, truncation="only_first")
            inputs["is_masked"] = [int(i == self.tokenizer.mask_token_id) for i in inputs["input_ids"]]
            inputs["first_mask"] = [int(i == 0) for i in inputs["token_type_ids"]]
            output_item["inputs"] = inputs
            if isinstance(item["entity"], dict):
                labels = list(map(lambda x: x + 2, item["entity"].values()))
                output_item["labels"] = labels
            output.append(output_item)
        return output

    def prepare_data(self) -> None:
        load = lambda file: list(map(json.loads, open(file, "r+", encoding="utf-8").readlines()))
        self.train_cache_file = self.hparams.train_data_path.replace(".txt", f"_{self.hparams.model_type}.pkl")
        self.test_cache_file = self.hparams.test_data_path.replace(".txt", f"_{self.hparams.model_type}.pkl")
        if not os.path.exists(self.train_cache_file):
            train_data = self._setup(load(self.hparams.train_data_path))
            pickle.dump(train_data, open(self.train_cache_file, "wb"))
        if not os.path.exists(self.test_cache_file):
            test_data = self._setup(load(self.hparams.test_data_path))
            pickle.dump(test_data, open(self.test_cache_file, "wb"))
        if self.hparams.pseudo_data_path:
            pass

    def setup(self, stage=None):
        load_pkl = lambda file: pickle.load(open(file, "rb"))
        if stage in ["fit", "validate"]:
            if self.train_dataset is None or self.valid_dataset is None:
                train_data = load_pkl(self.train_cache_file)
                if self.hparams.shuffle_valid:
                    random.shuffle(train_data)
                expanded_train_data = train_data[self.hparams.valid_size:]
                if self.hparams.pseudo_data_path:
                    pass
                self.train_dataset = BasicDataset(train_data[self.hparams.valid_size:])
                self.valid_dataset = BasicDataset(train_data[:self.hparams.valid_size])
        elif stage in ["test", "predict"]:
            if self.test_dataset is None:
                self.test_dataset = BasicDataset(load_pkl(self.test_cache_file))
        else:
            raise NotImplementedError

    def collate_fn(self, batch):
        output = {"inputs": {key: [] for key in batch[0]["inputs"]}, "labels": []}
        for item in batch:
            for key in item["inputs"]:
                output["inputs"][key].append(torch.tensor(item["inputs"][key]))
            output["labels"].extend(item["labels"])
        for key in output["inputs"]:
            output["inputs"][key] = torch.nn.utils.rnn.pad_sequence(output["inputs"][key], batch_first=True, padding_value=0)
        output["labels"] = torch.tensor(output["labels"])
        return output

    def test_collate_fn(self, batch):
        output = {"inputs": {key: [] for key in batch[0]["inputs"]}}
        for item in batch:
            for key in item["inputs"]:
                output["inputs"][key].append(torch.tensor(item["inputs"][key]))
        for key in output["inputs"]:
            output["inputs"][key] = torch.nn.utils.rnn.pad_sequence(output["inputs"][key], batch_first=True, padding_value=0)
        return output

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.train_batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.hparams.test_batch_size, shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.test_batch_size, shuffle=False, collate_fn=self.test_collate_fn)