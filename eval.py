import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import liar_dataset


def test(data_dir, model, tokenizer, batch_size=16, block_size=128, device='cpu'):
    logging.debug('load training data')
    test_dataset = liar_dataset(tokenizer, data_dir, split='test', block_size=block_size)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    lossFunc = nn.CrossEntropyLoss()

    logging.info('start testing')
    epoch_loss = 0
    y_true, y_pred = None, None

    i = 0
    steps = 0
    pbar = tqdm(test_dataloader)
    for batch in pbar:
        steps += 1
        tokens, labels = batch
        labels = labels.to(device)
        tokens = tokens.to(device)

        model.zero_grad()
        logits = model(tokens)

        loss = lossFunc(logits, labels)
        epoch_loss = steps / (steps + 1) * epoch_loss + loss.item() / (steps + 1)

        current_y_true = labels.cpu().numpy()
        current_y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_true = current_y_true if y_true is None else np.concatenate((y_true, current_y_true), axis=0)
        y_pred = current_y_pred if y_pred is None else np.concatenate((y_pred, current_y_pred), axis=0)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = 0 if accuracy == 0 else f1_score(y_true, y_pred, average='weighted')

        pbar.set_postfix({'loss': loss.item(), 'acc': accuracy, 'f1': f1})

        i += batch_size

    logging.info('Test split: [loss: ' + str(epoch_loss) + ', accuracy: ' +
                 str(accuracy_score(y_true, y_pred)) + ', f1: '
                 + str(f1_score(y_true, y_pred, average='weighted')) + ']')
