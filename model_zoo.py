import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from convberts import *


def get_loss_fct(task_type):
    if task_type == 'multiclass' or task_type == 'binary':
        loss_fct = nn.CrossEntropyLoss()
    elif task_type == 'multilabel':
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


class T5EncoderClassificationHead(nn.Module):
    """From https://gist.github.com/sam-writer/723baf81c501d9d24c6955f201d86bbb"""
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class T5EncoderForSequenceClassification:
    """Use an in-memory T5Encoder to do sequence classification"""
    def __init__(self, t5_encoder, config):
        self.num_labels = config.num_labels
        self.config = config

        self.encoder = t5_encoder  # already initialized model
        # either we are in eval mode, and the following code should do nothing
        # or we are training, but we only want to fine tune the classifier head
        # we do not want to fine-tune the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier = T5EncoderClassificationHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # TODO fix logic that takes cfg to determine task_type
            #
            #
            #
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )