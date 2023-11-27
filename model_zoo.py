import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from convberts import *

def get_loss_fct(task_type):
    if task_type == 'multiclass':
        loss_fct = nn.CrossEntropyLoss()
    elif task_type == 'multilabel' or task_type == 'binary':
        loss_fct = nn.BCEWithLogitsLoss()
    elif task_type == 'regression':
        loss_fct = nn.MSELoss()
    else:
        print('Specified wrong classification type')
    return loss_fct


class LinearClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)
        self.num_layers = cfg.num_layers
        self.input_layer = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for i in range(cfg.num_layers):
            self.hidden_layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
        self.output_layer = nn.Linear(cfg.hidden_dim, cfg.num_labels)
        self.loss_fct = get_loss_fct(cfg.task_type)
        self.task_type = cfg.task_type
        self.num_labels = cfg.num_labels

    def forward(self, embeddings, labels=None):
        embeddings = self.dropout(self.gelu(self.input_layer(embeddings)))
        for i in range(self.num_layers):
            embeddings = self.dropout(self.gelu(self.hidden_layers[i](embeddings)))
        logits = self.output_layer(embeddings)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )


class ClassificationHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.num_labels)

    def forward(self, x, **kwargs):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class LinearHeadWithBackbone(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone
        self.classification_head = ClassificationHead(cfg)
        self.loss_fct = get_loss_fct(cfg.task_type)
        self.num_labels = cfg.num_labels

    def forward(self, input_ids, attention_mask, labels=None):
        pooler = self.backbone(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        logits = self.classification_head(pooler)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )


class ConvBert(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.task_type == 'binary':
            self.model = ConvBertForBinaryClassification(cfg)
        elif cfg.task_type == 'multiclass':
            self.model = ConvBertForMultiClassClassification(cfg)
        elif cfg.task_type == 'multilabel':
            self.model = ConvBertForMultiLabelClassification(cfg)
        elif cfg.task_type == 'regression':
            self.model = ConvBertForRegression(cfg)
        else:
            print('You did not pass a correct task type:\n binary , multiclass , multilabel , regression')

    def forward(self, embeddings, labels=None):
        out = self.model(embeddings, labels)
        return out


class ConvBertWithBackbone(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone
        input_dim = self.backbone.config.hidden_size
        self.convbert = ConvBert(cfg)

    def forward(self, input_ids, attention_mask, labels=None):
        embeddings = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).hidden_states[-1]
        out = self.convbert(embeddings, labels)
        return out