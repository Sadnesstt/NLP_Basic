import sys
import argparse

import torch
import torch.nn as nn
version = list(map(int, torchtext.__version__.split('.')))
if version[0] <= 0 and version[1] < 9:
    from torchtext import data
else:
    from torchtext.legacy import data


from simple_ntc.models.rnn import RNNClassifier
from simple_ntc.models.cnn import CNNClassifier

def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type = int, default=-1)
    p.add_argument('--batch_size', type = int, default=256)
    p.add_argument('--top_k', type = int, default=1)
    p.add_argument('--max_length', type = int, default=256)

    p.add_argument('--drop_rnn', action='store_true')
    p.add_argument('--drop_cnn', action='store_true')

    config = p.parse_args()

    return config

def read_text(max_length=256):
    '''
    Read text from standard input for inference.
    '''
    lines = []

    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip().split(' ')[:max_length]]


    return lines

def define_field():
    '''
    To avoid use DataLoader class, just declare dummy fields.
    With those fields, we can restore mapping table between words and indice.
    '''

    return (
        data.Field(
            use_vocab = True,
            batch_first = True,
            include_lengths = False,
        ),
        data.Field(
            sequential=False,
            use_vocab = True,
            unk_token=None,
        )
    )

def main(config):
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )

    train_config = saved_data['config']
    rnn_best = saved_data['rnn']
    cnn_best = saved_data['cnn']
    vocab = saved_data['vocab']
    classes = saved_data['classes']

    