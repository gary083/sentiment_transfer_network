import sys, os, argparse, math, time, random, re
import tensorflow as tf 
import numpy as np
from six.moves import xrange

from datetime import datetime
from lib import data_utils
from vae import v_autoencoder

def step1(args): # training variational autoencoder
    train_id_path, dev_id_path, vocab_path = data_utils.prepare_diologue(args.works_dir, args.vocab_size)

    checkpoint_step = 200
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        print ()
        print ('-------loading model-------')
        model = v_autoencoder(sess, args, feed_previous=False)
        ckpt = tf.train.get_checkpoint_state(args.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Reading model parameters from %s @ %s" % (ckpt.model_checkpoint_path, datetime.now()))
            sess.run(tf.global_variables_initializer())
            model.saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model reloaded @ %s" % (datetime.now()))
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())

        print ('loading dictionary...')
        vocab, rev_vocab = data_utils.initialize_vocab(vocab_path)

        print ('loading dev set form %s...' %dev_id_path)
        dev_set = data_utils.read_data(dev_id_path)
        print ('loading train set form %s...' %train_id_path)
        train_set = data_utils.read_data(train_id_path)
        print ('-------start training-------')
        seq_loss, kl_loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        while True:
            encoder_inputs, decoder_inputs, weights = model.train_get_batch(train_set)
            step_seq_loss, step_kl_loss = model.ae_step(sess, encoder_inputs, decoder_inputs, weights, False)
            sess.run(model.kl_weight_op)
            sess.run(model.sample_rate_op)
            seq_loss += step_seq_loss / checkpoint_step
            kl_loss  += step_kl_loss / checkpoint_step
            current_step += 1
            if current_step % checkpoint_step == 0:
                print ("global step %d seq_loss %.4f kl_loss %.4f @ %s" %(model.global_step.eval(), math.exp(seq_loss), kl_loss, datetime.now()))

                checkpoint_path = os.path.join(args.model_dir, "model.ckpt")
                model.saver.save(sess, checkpoint_path )
                seq_loss, kl_loss = 0.0, 0.0
                encoder_inputs, decoder_inputs, weights = model.train_get_batch(dev_set)
                step_seq_loss, step_kl_loss = model.ae_step(sess, encoder_inputs, decoder_inputs, weights, True)
                print ("  eval: seq_loss %.2f" %(math.exp(step_seq_loss)))
                sys.stdout.flush()


def step2(args): # training sentiment classifier
    pos_id_path, neg_id_path, vocab_path = data_utils.prepare_cyc_diologue(args.works_dir, args.vocab_size)

    checkpoint_step = 200
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        print ()
        print ('-------loading model-------')
        model = v_autoencoder(sess, args, feed_previous=True)
        sess.run(tf.global_variables_initializer())
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

        print ('loading dictionary...')
        vocab, rev_vocab = data_utils.initialize_vocab(vocab_path)
        print ('loading dev set form %s...' %pos_id_path)
        pos_set = data_utils.read_cyc_data(pos_id_path)
        print ('loading train set from %s...' %neg_id_path)
        neg_set = data_utils.read_cyc_data(neg_id_path)
        test_pos = pos_set[-30000:]
        test_neg = neg_set[-30000:]
        train_pos = pos_set[:-30000]
        train_neg = neg_set[:-30000]
        print ('-------start training-------')
        
        acc = 0.0
        current_step = 0
        
        while True:
            encoder_inputs, decoder_inputs, masks, labels = model.class_get_batch(train_pos, train_neg)
            step_acc, _ = model.class_step(sess, encoder_inputs, decoder_inputs, masks, labels, False)
            # sess.run(model.class_rate_op)
            acc  += step_acc  / checkpoint_step
            current_step += 1

            if current_step % checkpoint_step == 0:
                print ("global step %d acc %.4f @ %s" %(model.class_global_step.eval(), acc, datetime.now()))
 
                checkpoint_path = os.path.join(args.class_dir, "model.ckpt")
                model.class_saver.save(sess, checkpoint_path )

                acc = 0.0
                encoder_inputs, decoder_inputs, masks, labels = model.class_get_batch(test_pos, test_neg)
                step_acc, _ = model.class_step(sess, encoder_inputs, decoder_inputs, masks,labels, True)
                # print (test)
                print ("  eval: acc %.4f" %(step_acc))
                sys.stdout.flush()

def print_sentence(result, rev_vocab, index):
    outputs = [ r[index] for r in result ]
    if data_utils.EOS_ID in outputs:
        eos = outputs.index(data_utils.EOS_ID)
        outputs = outputs[:eos]
    output_sentence = ' '.join([rev_vocab.get(t) for t in outputs])
    return (output_sentence)

def step3(args): # training transfer network
    pos_id_path, neg_id_path, vocab_path = data_utils.prepare_cyc_diologue(args.works_dir, args.vocab_size)
    checkpoint_step = 100
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


        print ('loading dictionary...')
        vocab, rev_vocab = data_utils.initialize_vocab(vocab_path)
        print ('loading dev set form %s...' %pos_id_path)
        pos_set = data_utils.read_cyc_data(pos_id_path)
        print ('loading train set from %s...' %neg_id_path)
        neg_set = data_utils.read_cyc_data(neg_id_path)
        print ('-------start training-------')

        style, content = 0.0, 0.0
        current_step = 0

        while True:
            encoder_inputs, decoder_inputs, _, _ = model.class_get_batch(pos_set, neg_set, 0.7)
            step_style, step_content = model.trans_step(sess, encoder_inputs, decoder_inputs, False)
            print ('\rstep: %d --style %.3f --content %.3f' %(current_step, step_style, step_content), end='')
            sess.run(model.trans_weight_op)
            style += step_style / checkpoint_step
            content += step_content / checkpoint_step
            current_step += 1

            if current_step % checkpoint_step == 0:
                print ()
                # print ("global step %d  @ %s" %(model.trans_global_step.eval(), datetime.now()))
                # print ("global step ", style,  " style ", content, )
                print ("global step %d style %.3f content %.3f @ %s" %(model.trans_global_step.eval(), style, content, datetime.now()))
                
                checkpoint_path = os.path.join(args.cyc_dir, "model.ckpt")
                model.trans_saver.save(sess, checkpoint_path )
                style, content = 0.0, 0.0

                encoder_inputs, decoder_inputs, _, _ = model.class_get_batch(pos_set, neg_set, 1.0)
                step_style, step_content = model.trans_step(sess, encoder_inputs, decoder_inputs, True)
                outputs, trans_outputs = model.trans_chat(sess, encoder_inputs, decoder_inputs)
                print ('score: ', step_style)
                print ('example --')
                print ('  outputs       : ', print_sentence(outputs, rev_vocab, 0))
                print ('  trans_outputs : ', print_sentence(trans_outputs, rev_vocab, 0))


def chat(args):
    with tf.Session() as sess:
        _, _, vocab_path = data_utils.prepare_diologue(args.works_dir, args.vocab_size)
        print ()
        print ('-------loading model-------')
        args.batch_size = 1
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

        sys.stdout.write("Input:     ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

        while sentence:
            get_sentence(sess, sentence, vocab, rev_vocab, model)
            sys.stdout.write("Input:     ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()


def get_sentence(sess, sentence, vocab, rev_vocab, model):
    pat = re.compile('(\W+)')
    input_sentence = re.split(pat, sentence.lower())

    sentence_id = data_utils.sentence_to_id(' '.join(input_sentence), vocab)
    print (sentence_id)
    # print (sentence_id)

    feed_data = sentence_id
    encoder_inputs, decoder_inputs = model.single_get_batch([feed_data])
    selected, selected_pos = model.trans_chat(sess, encoder_inputs, decoder_inputs)
    selected = [ i[0] for i in selected]

    if data_utils.EOS_ID in selected:
        eos = selected.index(data_utils.EOS_ID)
        selected = selected[:eos]
    output_sentence = ' '.join([rev_vocab.get(t) for t in selected])
    print (output_sentence)

    selected_pos_e = [ i[0] for i in selected_pos]
    print (selected_pos_e)
    if data_utils.EOS_ID in selected_pos_e:
        eos = selected_pos_e.index(data_utils.EOS_ID)
        selected_pos_e = selected_pos_e[:eos]
    output_sentence = ' '.join([rev_vocab.get(t) for t in selected_pos_e])
    print ('transform: ', output_sentence)
    print ()
