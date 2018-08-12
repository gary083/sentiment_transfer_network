import sys, os, argparse
import tensorflow as tf 
import numpy as np 

from lib.step import step1, step2, step3, chat
from lib.test import test

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step1', action='store_true', help='train seq2seq...')
    parser.add_argument('--step2', action='store_true', help='train classifier...')
    parser.add_argument('--step3', action='store_true', help='train transfer network...')
    parser.add_argument('--chat', action='store_true', help='chat...')
    parser.add_argument('--test', action='store_true', help='test...')
    parser.add_argument('--emb_size' , default=256, help='embedding_dim...')
    parser.add_argument('--max_length' , default=20, help='max_length...')
    parser.add_argument('--num_cell' , default=1024, help='num_cell...')
    parser.add_argument('--num_layers' , default=1, help='num_layers...')
    parser.add_argument('--batch_size' , default=32, help='batch_size...')
    parser.add_argument('--learning_rate' , default=0.1, help='learning_rate...')
    parser.add_argument('--vocab_size' , default=40000, help='vocab_size...')
    parser.add_argument('--checkpoint_step' , default=500, help='checkpoint_step...')
    parser.add_argument('--model' ,type=str , default='save', help='model_dir...')
    args = parser.parse_args()
    args.works_dir = 'works/' + args.model
    args.model_dir = args.works_dir + '/save'
    args.cyc_dir   = args.works_dir + '/trans_save'
    args.class_dir = args.works_dir + '/class_save'
    return args

def run(args):
    if args.step1: # training VAE
        step1(args)
    elif args.step2: #training sentiment classifier
        step2(args)
    elif args.step3: # training transfer network
        step3(args) 
    elif args.chat:
        chat(args)
    elif args.test: # test
        test(args)

if __name__ == '__main__':
    args = parse()
    run(args)

