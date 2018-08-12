import sys, os, argparse, math, time, re
import tensorflow as tf 
import numpy as np
from six.moves import xrange

from datetime import datetime
from lib import data_utils
from vae import v_autoencoder


def test(args):
    train_id_path, dev_id_path, vocab_path = data_utils.prepare_diologue(args.works_dir, args.vocab_size)
    pos_id_path, neg_id_path, vocab_path = data_utils.prepare_cyc_diologue(args.works_dir, args.vocab_size)

    checkpoint_step = 200
    args.batch_size = 1
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        print ()
        print ('-------loading model-------')
        model = v_autoencoder(sess, args, feed_previous=True)
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(args.cyc_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Reading model parameters from %s @ %s" % (ckpt.model_checkpoint_path, datetime.now()))
            model.trans_saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model reloaded @ %s" % (datetime.now()))
        ckpt = tf.train.get_checkpoint_state(args.class_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Reading model parameters from %s @ %s" % (ckpt.model_checkpoint_path, datetime.now()))
            model.class_saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model reloaded @ %s" % (datetime.now()))
        ckpt = tf.train.get_checkpoint_state(args.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Reading model parameters from %s @ %s" % (ckpt.model_checkpoint_path, datetime.now()))
            model.saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model reloaded @ %s" % (datetime.now()))

        vocab, rev_vocab = data_utils.initialize_vocab(vocab_path)

        with open('seq2seq', 'r') as fin:
            with open('neg2pos.txt', 'w') as fout:
                for l, seq in enumerate(fin):
                    print (('\r%d' %l) ,end='')
                    sentence = seq.strip()
                    out = get_sentence(sess, sentence, vocab, rev_vocab, model)
                    fout.write(out+'\n')

def get_sentence(sess, sentence, vocab, rev_vocab, model):
    pat = re.compile('(\W+)')
    input_sentence = re.split(pat, sentence.lower())
    sentence_id = data_utils.sentence_to_id(' '.join(input_sentence), vocab)

    feed_data = sentence_id
    encoder_inputs, decoder_inputs = model.single_get_batch([feed_data])
    selected, selected_pos = model.trans_chat(sess, encoder_inputs, decoder_inputs)

    selected_pos = [ i[0] for i in selected_pos]
    if data_utils.EOS_ID in selected_pos:
        eos = selected_pos.index(data_utils.EOS_ID)
        selected_pos = selected_pos[:eos]
    output_sentence = ' '.join([rev_vocab.get(t) for t in selected_pos])
    return output_sentence
