import torch
import numpy as np
from datasets import load_dataset
from torch.utils import data
from tqdm.auto import tqdm

def embed_dataset(model, tokenizer, sequences, cfg):
    model.eval()
    inputs_embedding = []
    with torch.no_grad():
        for sample in tqdm(sequences):
            ids = tokenizer(sample,
                            add_special_tokens=True,
                            padding=False,
                            return_token_type_ids=False,
                            return_tensors='pt').input_ids.to(cfg.device)
            output = model(ids)
            if cfg.cls:
                try:
                    emb = output.pooler_output.detach().cpu().numpy()
                except:
                    emb = output.hidden_states[-1][:, 0. :].detach().cpu().numpy()
            elif cfg.average:
                try:
                    emb = torch.mean(output.last_hidden_state, dim=1).detach().cpu().numpy()
                except:
                    emb = torch.mean(output.hidden_states[-1], dim=1).detach().cpu().numpy()
            else:
                try:
                    emb = output.last_hidden_state.detach().cpu().numpy()
                except:
                    emb = output.hidden_states[-1].detach().cpu().numpy()
            inputs_embedding.append(emb)
        model.train()
    return inputs_embedding


class FineTuneDatasetEmbeds(data.Dataset):
    def __init__(self, embeddings, labels, cfg):
        self.embeddings = embeddings
        self.labels = labels
        self.task_type = cfg.task_type
        self.num_classes = cfg.num_labels
        self.peft = cfg.peft

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        if self.task_type == 'binary' and not self.peft:
            label_encoded = torch.tensor([1-label, label], dtype=torch.float)
        elif self.task_type == 'multiclass':
            label_encoded = torch.zeros(self.num_classes, dtype=torch.float)
            label_encoded[label] = 1.0
        else:
            label_encoded = torch.tensor(label, dtype=torch.float)
        return {'embeddings': torch.tensor(embedding).squeeze(0), 'labels': label_encoded}
    def __len__(self):
        return len(self.labels)


class FineTuneDatasetCollator(data.Dataset):
    def __init__(self, ds, cfg):
        self.ds = ds
        self.task_type = cfg.task_type
        self.num_classes = cfg.num_labels
        self.peft = cfg.peft

    def __len__(self):
        return self.ds.num_rows

    def __getitem__(self, idx):
        label = self.ds['labels'][idx]
        if self.task_type == 'binary' and not self.peft:
            label_encoded = torch.tensor([1-label, label], dtype=torch.float)
        elif self.task_type == 'multiclass':
            label_encoded = torch.zeros(self.num_classes, dtype=torch.float)
            label_encoded[label] = 1.0
        else:
            label_encoded = torch.tensor(label, dtype=torch.float)
        return self.ds['seqs'][idx], label_encoded


def collate_fn(tokenizer):
    def _collate_fn(batch):
        seqs = [ex[0] for ex in batch]
        labels = torch.stack([ex[1] for ex in batch])
        input_ids = tokenizer(
            seqs,
            padding='longest',
            return_tensors='pt',
            truncation=True,
            max_length=cfg.max_length,
            return_token_type_ids=False,
        )
        input_ids.update({'labels': labels})
        return input_ids
    return _collate_fn


def get_data(cfg):
    dataset = load_dataset(cfg.data_path)
    train_set = dataset['train']
    valid_set = dataset['valid']
    test_set = dataset['test']
    if cfg.trim_len:
        train_set = train_set.filter(lambda x: len(x['seqs'].replace(' ', '')) <= cfg.max_length)
        valid_set = valid_set.filter(lambda x: len(x['seqs'].replace(' ', '')) <= cfg.max_length)
        test_set = test_set.filter(lambda x: len(x['seqs'].replace(' ', '')) <= cfg.max_length)
    if cfg.task_type == 'multilabel':
        import ast
        train_set = train_set.map(lambda example: {'labels': ast.literal_eval(example['labels'])})
        valid_set = valid_set.map(lambda example: {'labels': ast.literal_eval(example['labels'])})
        test_set = test_set.map(lambda example: {'labels': ast.literal_eval(example['labels'])})
    try:
        num_labels = len(train_set['labels'][0])
    except:
        num_labels = len(np.unique(train_set['labels']))
    if cfg.task_type == 'regression':
        cfg.traing_labels_mean = sum(train_set['labels']) / len(train_set['labels'])
        num_labels = 1
    cfg.num_labels = num_labels
    return train_set, valid_set, test_set


def get_seqs(dataset, seq_col='seqs', label_col='labels'):
    return dataset[seq_col], dataset[label_col]