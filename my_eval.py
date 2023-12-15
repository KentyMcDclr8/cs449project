import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from agent import Seq2SeqAgent

import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince
from env_train import R2RBatch
from model import EncoderLSTM, AttnDecoderLSTM
from eval import Evaluation
import pprint
pp = pprint.PrettyPrinter(indent=4)



TRAIN_VOCAB = 'tasks/R2R/RxR_data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/RxR_data/trainval_vocab.txt'
RESULT_DIR = 'tasks/R2R/results/RxR_Model/'
SNAPSHOT_DIR = 'tasks/R2R/snapshots/RxR_Model/'
PLOT_DIR = 'tasks/R2R/plots/'

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
MAX_INPUT_LENGTH = 80
RESULT_DIR = 'tasks/R2R/results/RxR_Model/'

features = IMAGENET_FEATURES
batch_size = 3
max_episode_len = 20
word_embedding_size = 256
action_embedding_size = 32
hidden_size = 512
bidirectional = False
dropout_ratio = 0.5 #0.5
feedback_method = 'sample' # teacher or sample
learning_rate = 0.00001
weight_decay = 0.001 #0.0005
n_iters = 5000 if feedback_method == 'teacher' else 20000
model_prefix = 'seq2seq_%s_imagenet' % (feedback_method)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

def initialization():
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    test_env = R2RBatch(features, batch_size=batch_size, splits=['test'], tokenizer=tok)

    # Creat validation environments
    #val_envs = {split: (R2RBatch(features, batch_size=batch_size, splits=[split],
                #tokenizer=tok), Evaluation([split])) for split in ['val_seen', 'val_unseen']}

    # Build models and train
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx,
                  dropout_ratio, bidirectional=bidirectional).to('cpu')
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                  action_embedding_size, hidden_size, dropout_ratio).to('cpu')
    predicted_trajs_path= RESULT_DIR + 'seq2seq_sample_imagenet_test_iter_20000.json'
    agent = Seq2SeqAgent(test_env, predicted_trajs_path, encoder, decoder, max_episode_len)
    agent.load(encoder_path='tasks/R2R/snapshots/RxR_Model/weight_decay=0.001,batch=400/seq2seq_sample_imagenet_train_enc_iter_5000',
               decoder_path='tasks/R2R/snapshots/RxR_Model/weight_decay=0.001,batch=400/seq2seq_sample_imagenet_train_dec_iter_5000')
    agent.test()
    agent.write_results()

initialization()
outfiles = [
        #RESULT_DIR + 'seq2seq_teacher_imagenet_%s_iter_5000.json',
        RESULT_DIR + 'seq2seq_sample_imagenet_%s_iter_20000.json'
        ]

for outfile in outfiles:
        for split in ['test']:
            ev = Evaluation([split])
            score_summary, _ = ev.score(outfile % split)
            print('\n%s' % outfile)
            pp.pprint(score_summary)





