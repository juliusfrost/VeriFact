import os

import torch
from torch.utils.data.dataset import TensorDataset
from transformers import BertTokenizer

LABELS = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
label2index = dict([(v, k) for k, v in enumerate(LABELS)])


# def generate_tensors(tokenizer, data_dir, split='train', block_size=512):
#     torchtext_dataset = liar_dataset(data_dir, split)
#
#     num_examples = len(torchtext_dataset)
#
#     label_tensor = torch.zeros(num_examples)
#     token_tensor = torch.zeros(num_examples, block_size)
#     for i, batch in enumerate(data.Iterator(torchtext_dataset, 1)):
#         label_tensor[0] = label2index[batch.label[0]]
#
#         tokens = tokenizer.encode(batch.text[0])
#         num_tokens = len(tokens)
#         if num_tokens > block_size:
#             raise Exception('The sentence input is too long for block size:\n{}'.format(batch.text[0]))
#         token_tensor[i, :num_tokens] = torch.tensor(tokens)
#
#     return token_tensor, label_tensor
#
#
# def liar_dataset(root, split='train', extension='tsv'):
#     path = os.path.join(root, split + '.' + extension)
#     label = data.RawField()
#     text = data.RawField()
#     fields = [
#         ('id', None),
#         ('label', label),
#         ('text', text),
#         ('subject', None),
#         ('speaker', None),
#         ('job title', None),
#         ('state info', None),
#         ('party', None),
#         ('credit history barely true', None),
#         ('credit history false', None),
#         ('credit history half true', None),
#         ('credit history mostly true', None),
#         ('credit history pants on fire', None),
#         ('context', None),
#     ]
#     dataset = data.TabularDataset(path, format=extension, fields=fields)
#     # label.build_vocab(dataset)
#     # text.build_vocab(dataset)
#     return dataset
#
#
# def load_liar_dataset(root):
#     delimiter = '\t'
#     # encoding = 'utf-8'
#     train_file = os.path.join(root, 'train.tsv')
#     test_file = os.path.join(root, 'test.tsv')
#     valid_file = os.path.join(root, 'valid.tsv')
#
#     def read(file):
#         return pd.read_csv(file, sep=delimiter)
#
#     train = read(train_file)
#     test = read(test_file)
#     valid = read(valid_file)
#
#     return train, valid, test


def liar_dataset(tokenizer, data_dir, split='train', block_size=128):
    tokens, labels = load_liar_to_tensor(tokenizer, data_dir, split=split, block_size=block_size)
    dataset = TensorDataset(tokens, labels)
    return dataset


def load_liar_to_tensor(tokenizer, data_dir, split='train', block_size=512):
    path = os.path.join(data_dir, split + '.tsv')

    lines = open(path, encoding='utf-8').readlines()

    num_examples = len(lines)
    label_tensor = torch.zeros(num_examples)
    token_tensor = torch.zeros(num_examples, block_size)

    for i, line in enumerate(lines):
        columns = line.split('\t')
        label = columns[1]
        text = columns[2]

        label_tensor[i] = label2index[label]
        if '\t' in text:
            raise
        tokens = tokenizer.encode(text)
        num_tokens = len(tokens)
        if num_tokens > block_size:
            raise Exception('The sentence input is too long for block size:\n{}'.format(text[i]))
        token_tensor[i, :num_tokens] = torch.tensor(tokens)

    return token_tensor.long(), label_tensor.long()


if __name__ == '__main__':
    root = 'C:\\Users\\Julius\\Documents\\GitHub\\liar_dataset'
    tokenizer_class = BertTokenizer
    pretrained_weights = 'bert-base-uncased'
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    label_tensor, token_tensor = load_liar_to_tensor(tokenizer, root)
