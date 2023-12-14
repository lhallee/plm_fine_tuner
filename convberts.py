import transformers.models.convbert as c_bert
import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput


"""
Base model that consists of ConvBert layer.

Args:
    input_dim: Dimension of the input embeddings.
    nhead: Integer specifying the number of heads for the `ConvBert` model.
    hidden_dim: Integer specifying the hidden dimension for the `ConvBert` model.
    nlayers: Integer specifying the number of layers for the `ConvBert` model.
    kernel_size: Integer specifying the filter size for the `ConvBert` model. Default: 7
    dropout: Float specifying the dropout rate for the `ConvBert` model. Default: 0.2
    pooling: String specifying the global pooling function. Accepts "avg" or "max". Default: "max"
"""

def init_weights(self):
    initrange = 0.1
    self.decoder.bias.data.zero_()
    self.decoder.weight.data.uniform_(-initrange, initrange)


class GlobalMaxPooling1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_max_pool1d = partial(torch.max, dim=1)

    def forward(self, x):
        out, _ = self.global_max_pool1d(x)
        return out


class GlobalAvgPooling1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_avg_pool1d = partial(torch.mean, dim=1)

    def forward(self, x):
        out = self.global_avg_pool1d(x)
        return out


class BaseModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        nhead: int,
        hidden_dim: int,
        num_hidden_layers: int = 1,
        num_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
        pooling: str = None,
    ):

        super().__init__()

        self.model_type = "Transformer"
        encoder_layers_Config = c_bert.ConvBertConfig(
            hidden_size=input_dim,
            num_attention_heads=nhead,
            intermediate_size=hidden_dim,
            conv_kernel_size=kernel_size,
            num_hidden_layers=num_hidden_layers,
            hidden_dropout_prob=dropout,
        )

        self.transformer_encoder = nn.ModuleList(
            [c_bert.ConvBertLayer(encoder_layers_Config) for _ in range(num_layers)]
        )

        if pooling is not None:
            if pooling in {"avg", "mean"}:
                self.pooling = GlobalAvgPooling1D()
            elif pooling == "max":
                self.pooling = GlobalMaxPooling1D()
            else:
                raise ValueError(
                    f"Expected pooling to be [`avg`, `max`]. Recieved: {pooling}"
                )

    def convbert_forward(self, x):
        for convbert_layer in self.transformer_encoder:
            x = convbert_layer(x)[0]
        return x


class ConvBertForBinaryClassification(BaseModule):
    def __init__(self, cfg):
        if cfg.pooling is None:
            raise ValueError(
                '`pooling` cannot be `None` in a binary classification task. Expected ["avg", "max"].'
            )

        super(ConvBertForBinaryClassification, self).__init__(
            input_dim=cfg.input_dim,
            nhead=cfg.nhead,
            hidden_dim=cfg.hidden_dim // 2,
            num_hidden_layers=cfg.num_layers,
            num_layers=cfg.num_layers,
            kernel_size=cfg.kernel,
            dropout=cfg.dropout,
            pooling=cfg.pooling,
        )

        self.decoder = nn.Linear(cfg.input_dim, 2)
        init_weights(self)

    def _compute_loss(self, logits, labels):
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            loss = None
        return loss

    def forward(self, embed, labels=None):
        hidden_inputs = self.convbert_forward(embed)
        hidden_inputs = self.pooling(hidden_inputs)
        logits = self.decoder(hidden_inputs)
        loss = self._compute_loss(logits, labels)

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=None, attentions=None
        )


class ConvBertForMultiClassClassification(BaseModule):
    def __init__(self, cfg):
        super(ConvBertForMultiClassClassification, self).__init__(
            input_dim=cfg.input_dim,
            nhead=cfg.nhead,
            hidden_dim=cfg.hidden_dim // 2,
            num_hidden_layers=cfg.num_layers,
            num_layers=cfg.num_layers,
            kernel_size=cfg.kernel,
            dropout=cfg.dropout,
            pooling=cfg.pooling,
        )

        self.model_type = "Transformer"
        self.num_labels = cfg.num_labels
        self.decoder = nn.Linear(cfg.input_dim, cfg.num_labels)
        init_weights(self)

    def _compute_loss(self, logits, labels):
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        else:
            loss = None
        return loss

    def forward(self, embed, labels=None):
        hidden_inputs = self.convbert_forward(embed)
        hidden_inputs = self.pooling(hidden_inputs)
        logits = self.decoder(hidden_inputs)
        loss = self._compute_loss(logits, labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class ConvBertForRegression(BaseModule):
    def __init__(self, cfg):
        if cfg.pooling is None:
            raise ValueError(
                '`pooling` cannot be `None` in a regression task. Expected ["avg", "max"].'
            )

        super(ConvBertForRegression, self).__init__(
            input_dim=cfg.input_dim,
            nhead=cfg.nhead,
            hidden_dim=cfg.hidden_dim // 2,
            num_hidden_layers=cfg.num_layers,
            num_layers=cfg.num_layers,
            kernel_size=cfg.kernel,
            dropout=cfg.dropout,
            pooling=cfg.pooling,
        )

        self.training_labels_mean = cfg.training_labels_mean
        self.decoder = nn.Linear(cfg.input_dim, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if self.training_labels_mean is not None:
            self.decoder.bias.data.fill_(self.training_labels_mean)
        else:
            self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _compute_loss(self, logits, labels):
        if labels is not None:
            # Ensure that logits and labels have the same size before computing loss
            loss = F.mse_loss(logits.view(-1), labels.view(-1))
        else:
            loss = None
        return loss

    def forward(self, embed, labels=None):
        hidden_inputs = self.convbert_forward(embed)
        hidden_inputs = self.pooling(hidden_inputs)
        logits = self.decoder(hidden_inputs)
        loss = self._compute_loss(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class ConvBertForMultiLabelClassification(BaseModule):
    def __init__(self, cfg):
        super(ConvBertForMultiLabelClassification, self).__init__(
            input_dim=cfg.input_dim,
            nhead=cfg.nhead,
            hidden_dim=cfg.hidden_dim // 2,
            num_hidden_layers=cfg.num_layers,
            num_layers=cfg.num_layers,
            kernel_size=cfg.kernel,
            dropout=cfg.dropout,
            pooling=cfg.pooling,
        )

        self.model_type = "Transformer"
        self.num_labels = cfg.num_labels
        self.decoder = nn.Linear(cfg.input_dim, cfg.num_labels)
        init_weights(self)

    def _compute_loss(self, logits, labels):
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
        else:
            loss = None
        return loss

    def forward(self, embed, labels=None):
        hidden_inputs = self.convbert_forward(embed)
        hidden_inputs = self.pooling(hidden_inputs)
        logits = self.decoder(hidden_inputs)
        loss = self._compute_loss(logits, labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )