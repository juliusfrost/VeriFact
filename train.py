import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import liar_dataset


def train(data_dir, model, tokenizer,
          epochs=1000,
          batch_size=16,
          lr=0.001,
          block_size=128,
          save_file=None,
          save_interval=10,
          device='cpu',
          **kwargs):
    logging.debug('load training data')
    train_dataset = liar_dataset(tokenizer, data_dir, split='train', block_size=block_size)
    logging.debug('load validation data')
    valid_dataset = liar_dataset(tokenizer, data_dir, split='valid', block_size=block_size)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    lossFunc = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.01)
    num_updates = (len(train_dataset) // batch_size + 1) * epochs
    if kwargs.get('warm_up', False):
        scheduler = transformers.WarmupCosineSchedule(optimizer, 10, num_updates)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, 7, 0.1)
    epoch_loss_history = []
    pbar = tqdm(range(epochs), desc='Epochs')
    logging.info('start training')
    for epoch in pbar:
        for split in ['train', 'validation']:
            epoch_loss = 0
            y_true, y_pred = None, None

            i = 0
            steps = 0
            dataloader = train_dataloader if split == 'train' else valid_dataloader
            pbar2 = tqdm(dataloader)
            for batch in pbar2:
                steps += 1
                tokens, labels = batch
                labels = labels.to(device)
                tokens = tokens.to(device)

                model.zero_grad()
                logits = model(tokens)

                loss = lossFunc(logits, labels)
                epoch_loss = steps / (steps + 1) * epoch_loss + loss.item() / (steps + 1)
                if split == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step(epoch)

                current_y_true = labels.cpu().numpy()
                current_y_pred = torch.argmax(logits, dim=1).cpu().numpy()
                y_true = current_y_true if y_true is None else np.concatenate((y_true, current_y_true), axis=0)
                y_pred = current_y_pred if y_pred is None else np.concatenate((y_pred, current_y_pred), axis=0)
                accuracy = accuracy_score(y_true, y_pred)
                f1 = 0 if accuracy == 0 else f1_score(y_true, y_pred, average='weighted')
                # print(split + ':', loss.item(), accuracy, f1)
                pbar2.set_postfix({'loss': loss.item(), 'acc': accuracy, 'f1': f1})

                i += batch_size

            epoch_loss_history.append(epoch_loss)

            if split == 'train' and save_file is not None and epoch % save_interval == save_interval - 1:
                logging.debug('save model')
                torch.save(model.state_dict(), save_file)

            logging.info('Epoch ' + str(epoch) + ', ' + split + ' split: [loss: ' + str(epoch_loss) + ', accuracy: ' +
                         str(accuracy_score(y_true, y_pred)) + ', f1: '
                         + str(f1_score(y_true, y_pred, average='weighted')) + ']')
