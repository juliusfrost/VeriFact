import argparse
import datetime
import logging
import re
import os

import torch
import torch.nn as nn

from model import pretrained
from train import train
from eval import test


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--pretrained-model', type=str, default='bert')
    parser.add_argument('--pretrained-weights', type=str, default=None)
    parser.add_argument('--pretrained-hidden-size', type=int, default=768)
    parser.add_argument('--pretrained-heads', type=int, default=12)
    parser.add_argument('--dont-freeze', action='store_true')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--model-parallel', action='store_true')
    parser.add_argument('--liar-dataset-dir', type=str, default='')
    parser.add_argument('--save-file', type=str, default='model.pt')
    parser.add_argument('--dont-save', action='store_true')
    parser.add_argument('--logger-name', default=None)
    parser.add_argument('--dont-log', action='store_true')
    parser.add_argument('--dont-load-model-from-file', action='store_true')
    parser.add_argument('--test', action='store_true')

    return parser.parse_args()


def main():
    args = parse()

    if args.logger_name is not None:
        logger_name = args.logger_name
    else:
        logger_name = datetime.datetime.now().isoformat()

    if not args.dont_log:
        logging.getLogger(logger_name)
        logger_filename = re.subn('\\D+', '', logger_name)[0] + '.log'
        logging.basicConfig(filename=logger_filename, level=logging.DEBUG)

    save_file = None if args.dont_save else args.save_file

    root = args.liar_dataset_dir

    tokenizer, model = pretrained(model=args.pretrained_model, weights=args.pretrained_weights,
                                  freeze=not args.dont_freeze)
    if not args.dont_load_model_from_file and os.path.exists(args.save_file):
        model.load_state_dict(torch.load(args.save_file))

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    print('using', device)
    model = model.to(device)
    if torch.cuda.device_count() > 1 and args.model_parallel:
        model = nn.DataParallel(model)

    if args.test:
        test(root, model, tokenizer, batch_size=args.batch_size, device=device)
    else:
        train(root, model, tokenizer, epochs=args.epochs, batch_size=args.batch_size, save_file=save_file, device=device)


if __name__ == '__main__':
    main()
