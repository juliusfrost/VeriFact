import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerDecoder, TransformerDecoderLayer
from transformers import *

MODELS = [(BertModel, BertTokenizer, 'bert-base-uncased'),
          (OpenAIGPTModel, OpenAIGPTTokenizer, 'openai-gpt'),
          (GPT2Model, GPT2Tokenizer, 'gpt2'),
          (CTRLModel, CTRLTokenizer, 'ctrl'),
          (TransfoXLModel, TransfoXLTokenizer, 'transfo-xl-wt103'),
          (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
          (XLMModel, XLMTokenizer, 'xlm-mlm-enfr-1024'),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
          (RobertaModel, RobertaTokenizer, 'roberta-base')]


def pretrained(model='bert', weights=None, freeze=True):
    output_size = 6
    if model == 'bert':
        model_class = BertModel
        tokenizer_class = BertTokenizer
        pretrained_weights = 'bert-base-uncased'
        hidden_size = 768
    elif model == 'gpt2':
        model_class = GPT2Model
        tokenizer_class = GPT2Tokenizer
        pretrained_weights = 'gpt2-large'
        hidden_size = 1280
    else:
        model_class = BertModel
        tokenizer_class = BertTokenizer
        pretrained_weights = 'bert-base-uncased'
        hidden_size = 768

    if weights is not None:
        pretrained_weights = weights

    return build_model_and_tokenizer(model_class, tokenizer_class, pretrained_weights, hidden_size, output_size, freeze)


def build_model_and_tokenizer(model_class, tokenizer_class, pretrained_weights,
                              hidden_size, output_size, freeze):
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    pretrained_model = model_class.from_pretrained(pretrained_weights)
    model = Model(pretrained_model, hidden_size, output_size, freeze=freeze)
    return tokenizer, model


class Model(torch.nn.Module):
    def __init__(self,
                 pretrained_model,
                 pretrained_hidden_size,
                 output_size,
                 hidden_size=512,
                 num_layers=2,
                 freeze=True):
        super().__init__()
        self.pretrained_model = pretrained_model
        if freeze:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(pretrained_hidden_size, hidden_size, num_layers, bidirectional=False)
        self.feed_forward = nn.Linear(hidden_size, output_size)

    def forward(self, input_ids):
        last_hidden_states = self.pretrained_model(input_ids)[0]
        rnn_output, h_n = self.rnn(torch.transpose(last_hidden_states, dim0=1, dim1=0))
        h_3 = h_n[0]
        output = self.feed_forward(h_3)
        return output


class VeriFact(torch.nn.Module):
    def __init__(self, pretrained_model, stacked_model, freeze_weights=True):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.stacked_model = stacked_model

    def forward(self, input_ids):
        return self.stacked_model(self.pretrained_model(input_ids))


class StackedModel(torch.nn.Module):
    def __init__(self,
                 pretrained_hidden_size,
                 output_size,
                 layer='rnn',
                 hidden_size=512,
                 num_layers=2, ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(pretrained_hidden_size, hidden_size, num_layers, bidirectional=False)
        self.feed_forward = nn.Linear(hidden_size, output_size)

    def forward(self, *input, **kwargs):
        return
