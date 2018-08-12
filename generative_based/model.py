import tensorflow as tf 
import numpy as np 
import _pickle as pk 
import random, sys, os, argparse, math, time, re

from datetime import datetime
from lib import data_utils
from lib.ops import *
from lib.encoder import encoder
from lib.generator import generator
from lib.discriminator import discriminator, discriminator_2

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"  

class sentiment_trans(object):
    def __init__(self, args, sess):
        _, _ = data_utils.set_file(args.works_dir, '')
        args = self.set_config(args)
        self.sess = sess
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.vocab_size = args.vocab_size
        self.latent_dim = args.latent_dim
        self.emb_size = args.emb_size
        self.checkpoint_step = args.checkpoint_step
        self.temperature = 0.001
        self.dim = args.dim
        self.works_dir = args.works_dir
        self.attention = args.attention
        self.pretrain = args.pretrain
        self.data_path = os.path.join('data', args.data_path)
        print (args)
        self.sample_rate = tf.Variable(float(1), trainable=False)
        self.global_step = tf.Variable(0, trainable=False)
        
        self.build_graph()
        self.sample_rate_op = self.sample_rate.assign( tf.maximum(0.5, 1-(tf.cast(self.global_step, tf.float32)/50000)) )
        self.saver = tf.train.Saver()

    def set_config(self, args):
        config_path = os.path.join(args.works_dir, 'config')
        if os.path.exists(config_path):
            return pk.load(open(config_path, 'rb'))
        else:
            pk.dump(args, open(config_path, 'wb'))
            return args

    def build_graph(self):
        with tf.variable_scope('data_process') as scope:
            self.pos_encoder_inputs   = []
            self.pos_decoder_targets  = []
            self.pos_target_weights   = []
            self.neg_encoder_inputs   = []
            self.neg_decoder_targets  = []
            self.neg_target_weights   = []
            for i in range(self.max_length):
                self.pos_encoder_inputs.append( tf.placeholder(tf.int32, shape=[None], name="pos_encoder{0}".format(i)))
                self.pos_decoder_targets.append(tf.placeholder(tf.int32, shape=[None], name="pos_decoder{0}".format(i)))
                self.pos_target_weights.append( tf.placeholder(tf.float32, shape=[None], name="pos_weight{0}".format(i)))
                self.neg_encoder_inputs.append( tf.placeholder(tf.int32, shape=[None], name="neg_encoder{0}".format(i)))
                self.neg_decoder_targets.append(tf.placeholder(tf.int32, shape=[None], name="neg_decoder{0}".format(i)))
                self.neg_target_weights.append( tf.placeholder(tf.float32, shape=[None], name="neg_weight{0}".format(i)))

            BOS_slice = tf.ones([self.batch_size], dtype=tf.int32) * data_utils.GO_ID
            POS_slice = tf.ones([self.batch_size], dtype=tf.int32) * data_utils.POS_ID
            NEG_slice = tf.ones([self.batch_size], dtype=tf.int32) * data_utils.NEG_ID
            self.pos_decoder_inputs = [BOS_slice] + self.pos_decoder_targets
            self.neg_decoder_inputs = [BOS_slice] + self.neg_decoder_targets
            self.pos_encoder_inputs = self.pos_encoder_inputs + [POS_slice]
            self.neg_encoder_inputs = self.neg_encoder_inputs + [NEG_slice] 
            self.encoder_length = [self.max_length+1 for _ in range(self.batch_size)]

        with tf.variable_scope('word_embedding') as scope:
            init = tf.contrib.layers.xavier_initializer()
            self.word_embedding = tf.get_variable(name='embedding', 
                                                  shape=[self.vocab_size, self.emb_size],
                                                  initializer=init)

        with tf.variable_scope('encoder') as scope:
            Encoder = encoder(self.latent_dim, self.word_embedding)
            pos_encoder_outputs, pos_encoder_state = Encoder(self.pos_encoder_inputs, self.encoder_length)
            scope.reuse_variables()
            neg_encoder_outputs, neg_encoder_state = Encoder(self.neg_encoder_inputs, self.encoder_length)

        with tf.variable_scope('generator') as scope:
            Generator = generator(self.latent_dim, self.word_embedding, self.vocab_size, self.temperature, self.attention)
            true_pos_dist, self.true_pos_id, true_pos_out = Generator(encoder_state=pos_encoder_state, 
                                                            encoder_outputs=pos_encoder_outputs, 
                                                            sentiment_code=tf.ones([self.batch_size, 1], dtype=tf.float32),
                                                            decoder_inputs=self.pos_decoder_inputs,
                                                            sample_rate=self.sample_rate,
                                                            feed_previous=False,
                                                            scope=scope)

            scope.reuse_variables()
            fake_pos_dist, self.fake_pos_id, fake_pos_out = Generator(encoder_state=neg_encoder_state, 
                                                            encoder_outputs=neg_encoder_outputs, 
                                                            sentiment_code=tf.ones([self.batch_size, 1], dtype=tf.float32),
                                                            decoder_inputs=self.pos_decoder_inputs,
                                                            sample_rate=self.sample_rate,
                                                            feed_previous=True,
                                                            scope=scope)

            true_neg_dist, self.true_neg_id, true_neg_out = Generator(encoder_state=neg_encoder_state, 
                                                            encoder_outputs=neg_encoder_outputs, 
                                                            sentiment_code=tf.zeros([self.batch_size, 1], dtype=tf.float32),
                                                            decoder_inputs=self.neg_decoder_inputs,
                                                            sample_rate=self.sample_rate,
                                                            feed_previous=False,
                                                            scope=scope)

            fake_neg_dist, self.fake_neg_id, fake_neg_out = Generator(encoder_state=pos_encoder_state, 
                                                            encoder_outputs=pos_encoder_outputs, 
                                                            sentiment_code=tf.zeros([self.batch_size, 1], dtype=tf.float32),
                                                            decoder_inputs=self.neg_decoder_inputs,
                                                            sample_rate=self.sample_rate,
                                                            feed_previous=True,
                                                            scope=scope)

            self.pos_sequence_loss = Generator.get_sequence_loss(true_pos_dist, self.pos_decoder_targets, self.pos_target_weights)
            self.neg_sequence_loss = Generator.get_sequence_loss(true_neg_dist, self.neg_decoder_targets, self.neg_target_weights)

        with tf.variable_scope('pos_discriminator') as scope:
            POS_Discriminator = discriminator(self.vocab_size, self.max_length, self.dim)

            pos_alpha = tf.random_uniform(shape=[self.batch_size,1,1])
            inter_pos_d_input = true_pos_dist + pos_alpha * (fake_pos_dist - true_pos_dist)

            true_pos_pred = POS_Discriminator(true_pos_dist)
            scope.reuse_variables()
            fake_pos_pred = POS_Discriminator(fake_pos_dist)

            pos_gradient = tf.gradients(POS_Discriminator(inter_pos_d_input), [inter_pos_d_input])[0] 
            pos_slope = tf.sqrt(tf.reduce_sum(tf.square(pos_gradient), reduction_indices=[1,2]))
            self.pos_penalty = tf.reduce_mean((pos_slope - 1.0)**2)

        # with tf.variable_scope('pos_discriminator_2') as scope:
        #     POS_Discriminator = discriminator_2(self.latent_dim, self.max_length, self.dim)

        #     pos_alpha = tf.random_uniform(shape=[self.batch_size,1,1])
        #     inter_pos_d_input = true_pos_out + pos_alpha * (fake_pos_out - true_pos_out)

        #     true_pos_pred = POS_Discriminator(true_pos_out)
        #     scope.reuse_variables()
        #     fake_pos_pred = POS_Discriminator(fake_pos_out)

        #     pos_gradient = tf.gradients(POS_Discriminator(inter_pos_d_input), [inter_pos_d_input])[0] 
        #     pos_slope = tf.sqrt(tf.reduce_sum(tf.square(pos_gradient), reduction_indices=[1,2]))
        #     self.pos_penalty = tf.reduce_mean((pos_slope - 1.0)**2)

        with tf.variable_scope('neg_discriminator') as scope:
            NEG_Discriminator = discriminator(self.vocab_size, self.max_length, self.dim)

            neg_alpha = tf.random_uniform(shape=[self.batch_size,1,1])
            inter_neg_d_input = true_neg_dist + neg_alpha * (fake_neg_dist - true_neg_dist) 

            true_neg_pred = NEG_Discriminator(true_neg_dist)
            scope.reuse_variables()
            fake_neg_pred = NEG_Discriminator(fake_neg_dist)

            neg_gradient = tf.gradients(NEG_Discriminator(inter_neg_d_input), [inter_neg_d_input])[0]
            neg_slope = tf.sqrt(tf.reduce_sum(tf.square(neg_gradient), reduction_indices=[1,2]))
            self.neg_penalty = tf.reduce_mean((neg_slope - 1.0)**2)
            
        # with tf.variable_scope('neg_discriminator') as scope:
        #     NEG_Discriminator = discriminator_2(self.latent_dim, self.max_length, self.dim)

        #     neg_alpha = tf.random_uniform(shape=[self.batch_size,1,1])
        #     inter_neg_d_input = true_neg_out + neg_alpha * (fake_neg_out - true_neg_out) 

        #     true_neg_pred = NEG_Discriminator(true_neg_out)
        #     scope.reuse_variables()
        #     fake_neg_pred = NEG_Discriminator(fake_neg_out)

        #     neg_gradient = tf.gradients(NEG_Discriminator(inter_neg_d_input), [inter_neg_d_input])[0]
        #     neg_slope = tf.sqrt(tf.reduce_sum(tf.square(neg_gradient), reduction_indices=[1,2]))
        #     self.neg_penalty = tf.reduce_mean((neg_slope - 1.0)**2)

        self.gen_vars = [var for var in tf.trainable_variables() if var.name.startswith('generator') or var.name.startswith('encoder') or var.name.startswith('word_embedding')]
        self.pos_dis_vars = [var for var in tf.trainable_variables() if var.name.startswith('pos_discriminator')]
        self.neg_dis_vars = [var for var in tf.trainable_variables() if var.name.startswith('neg_discriminator')]

        with tf.variable_scope('loss') as scope:
            self.pretrain_loss = self.pos_sequence_loss + self.neg_sequence_loss

            self.pos_adv_loss = - tf.reduce_mean(true_pos_pred - fake_pos_pred) 
            self.neg_adv_loss = - tf.reduce_mean(true_neg_pred - fake_neg_pred) 

            self.gen_loss = (self.pos_sequence_loss + self.neg_sequence_loss) - (self.pos_adv_loss + self.neg_adv_loss ) * 0.5
            self.pos_dis_loss = self.pos_adv_loss + 10.0 * self.pos_penalty
            self.neg_dis_loss = self.neg_adv_loss + 10.0 * self.neg_penalty 

        with tf.variable_scope('optimizer') as scope:
            pre_gradients = tf.gradients(self.pretrain_loss, self.gen_vars)
            pre_clipped_gradients, _ = tf.clip_by_global_norm(pre_gradients, 1.0)
            self.pretrain_gen_op = tf.train.AdamOptimizer(0.0001).apply_gradients(zip(pre_clipped_gradients, self.gen_vars))

            gradients = tf.gradients(self.gen_loss, self.gen_vars)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            self.train_gen_op = tf.train.AdamOptimizer(0.0001).apply_gradients(zip(clipped_gradients, self.gen_vars), global_step=self.global_step)
            self.train_pos_dis_op = tf.train.AdamOptimizer(0.0001).minimize(self.pos_dis_loss, var_list=self.pos_dis_vars)
            self.train_neg_dis_op = tf.train.AdamOptimizer(0.0001).minimize(self.neg_dis_loss, var_list=self.neg_dis_vars)

    def pretrain_step(self, sess, encoder_inputs, target_weights):
        input_feed = {}
        for l in range(self.max_length):
            input_feed[self.pos_encoder_inputs[l].name]  = encoder_inputs[0][self.max_length-1-l]
            input_feed[self.pos_decoder_targets[l].name] = encoder_inputs[0][l]
            input_feed[self.pos_target_weights[l].name]  = target_weights[0][l]
            input_feed[self.neg_encoder_inputs[l].name]  = encoder_inputs[1][self.max_length-1-l]
            input_feed[self.neg_decoder_targets[l].name] = encoder_inputs[1][l]
            input_feed[self.neg_target_weights[l].name]  = target_weights[1][l]

        output_feed = [self.pos_sequence_loss, self.neg_sequence_loss, self.pretrain_gen_op] 

        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]

    def g_step(self, sess, encoder_inputs, target_weights, train):
        input_feed = {}
        for l in range(self.max_length):
            input_feed[self.pos_encoder_inputs[l].name]  = encoder_inputs[0][self.max_length-1-l]
            input_feed[self.pos_decoder_targets[l].name] = encoder_inputs[0][l]
            input_feed[self.pos_target_weights[l].name]  = target_weights[0][l]
            input_feed[self.neg_encoder_inputs[l].name]  = encoder_inputs[1][self.max_length-1-l]
            input_feed[self.neg_decoder_targets[l].name] = encoder_inputs[1][l]
            input_feed[self.neg_target_weights[l].name]  = target_weights[1][l]

        output_feed = [self.gen_loss, self.pos_adv_loss, self.neg_adv_loss, self.pos_sequence_loss, self.neg_sequence_loss]
        if train:
            output_feed.append(self.train_gen_op)

        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]

    def d_step(self, sess, encoder_inputs, train):
        input_feed = {}
        for l in range(self.max_length):
            input_feed[self.pos_encoder_inputs[l].name]  = encoder_inputs[0][self.max_length-1-l]
            input_feed[self.pos_decoder_targets[l].name] = encoder_inputs[0][l]
            input_feed[self.neg_encoder_inputs[l].name]  = encoder_inputs[1][self.max_length-1-l]
            input_feed[self.neg_decoder_targets[l].name] = encoder_inputs[1][l]

        output_feed = [self.pos_dis_loss, self.neg_dis_loss, self.pos_adv_loss, self.neg_adv_loss]
        if train:
            output_feed.append(self.train_pos_dis_op)
            output_feed.append(self.train_neg_dis_op)
        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[2], outputs[3]

    def get_sentence(self, sess, encoder_inputs):
        input_feed = {}
        for l in range(self.max_length):
            input_feed[self.pos_encoder_inputs[l].name]  = encoder_inputs[0][self.max_length-1-l]
            input_feed[self.pos_decoder_targets[l].name] = encoder_inputs[0][l]
            input_feed[self.neg_encoder_inputs[l].name]  = encoder_inputs[1][self.max_length-1-l]
            input_feed[self.neg_decoder_targets[l].name] = encoder_inputs[1][l]

        output_feed = [self.true_pos_id, self.true_neg_id, self.fake_pos_id, self.fake_neg_id]
        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[2], outputs[3]

    def get_batch(self, pos_data, neg_data):
        pos_encoder_inputs, neg_encoder_inputs  = [], []

        for _ in range(self.batch_size):
            pos_encoder_input, _ = random.choice(pos_data)
            pos_encoder_pad = [data_utils.EOS_ID] * (self.max_length - len(pos_encoder_input))
            pos_encoder_inputs.append(pos_encoder_input + pos_encoder_pad)

            neg_encoder_input, _ = random.choice(neg_data)
            neg_encoder_pad = [data_utils.EOS_ID] * (self.max_length - len(neg_encoder_input))
            neg_encoder_inputs.append(neg_encoder_input + neg_encoder_pad)

        batch_pos_encoder_inputs, batch_neg_encoder_inputs, batch_pos_weights, batch_neg_weights = [], [], [], []

        for l in range(self.max_length):
            batch_pos_encoder_inputs.append(np.array([pos_encoder_inputs[b][l] for b in range(self.batch_size)], dtype=np.int32))
            batch_neg_encoder_inputs.append(np.array([neg_encoder_inputs[b][l] for b in range(self.batch_size)], dtype=np.int32))

            batch_pos_weight = np.ones(self.batch_size, dtype=np.float32)
            batch_neg_weight = np.ones(self.batch_size, dtype=np.float32) 
            pos_flag, neg_flag = False, False
            for b in range(self.batch_size):
                if l < self.max_length - 1:
                    pos_target = pos_encoder_inputs[b][l+1]
                    neg_target = neg_encoder_inputs[b][l+1]
                if pos_flag:
                    batch_pos_weight[b] = 0.0
                if neg_flag:
                    batch_neg_weight[b] = 0.0
                if pos_target == data_utils.EOS_ID:
                    pos_flag = True
                if neg_target == data_utils.EOS_ID:
                    neg_flag = True
            batch_pos_weights.append(batch_pos_weight)
            batch_neg_weights.append(batch_neg_weight)

        batch_encoder_inputs = (batch_pos_encoder_inputs, batch_neg_encoder_inputs)
        batch_target_weights = (batch_pos_weights, batch_neg_weights)

        return batch_encoder_inputs, batch_target_weights

    def get_batch_epoch(self, pos_data, neg_data):
        len_pos = len(pos_data)
        len_neg = len(neg_data)
        if len_pos%self.batch_size != 0:
            pad_size = (self.batch_size - (len(pos_data)%self.batch_size))
            len_pos = len(pos_data) + pad_size
            pos = pos_data + pos_data[:pad_size]
        if len_neg%self.batch_size != 0:
            pad_size = (self.batch_size - (len(neg_data)%self.batch_size))
            len_neg = len(neg_data) + pad_size
            neg = neg_data + neg_data[:pad_size]
        if len_pos > len_neg:
            data_len = len_pos
            neg.extend(neg[:len_pos-len_neg])
        if len_neg >= len_pos:
            data_len = len_neg
            pos.extend(pos[:len_neg-len_pos])

        i = 0
        while i < data_len:
            start = i
            end = start + self.batch_size
            pos_encoder_inputs, neg_encoder_inputs  = [], []

            for t in range(self.batch_size):
                pos_encoder_input, _ = pos[start+t]
                pos_encoder_pad = [data_utils.EOS_ID] * (self.max_length - len(pos_encoder_input))
                pos_encoder_inputs.append(pos_encoder_input + pos_encoder_pad)

                neg_encoder_input, _ = neg[start+t]
                neg_encoder_pad = [data_utils.EOS_ID] * (self.max_length - len(neg_encoder_input))
                neg_encoder_inputs.append(neg_encoder_input + neg_encoder_pad)

            batch_pos_encoder_inputs, batch_neg_encoder_inputs, batch_pos_weights, batch_neg_weights = [], [], [], []

            for l in range(self.max_length):
                batch_pos_encoder_inputs.append(np.array([pos_encoder_inputs[b][l] for b in range(self.batch_size)], dtype=np.int32))
                batch_neg_encoder_inputs.append(np.array([neg_encoder_inputs[b][l] for b in range(self.batch_size)], dtype=np.int32))

                batch_pos_weight = np.ones(self.batch_size, dtype=np.float32)
                batch_neg_weight = np.ones(self.batch_size, dtype=np.float32) 
                pos_flag, neg_flag = False, False
                for b in range(self.batch_size):
                    if l < self.max_length - 1:
                        pos_target = pos_encoder_inputs[b][l+1]
                        neg_target = neg_encoder_inputs[b][l+1]
                    if pos_flag:
                        batch_pos_weight[b] = 0.0
                    if neg_flag:
                        batch_neg_weight[b] = 0.0
                    if pos_target == data_utils.EOS_ID:
                        pos_flag = True
                    if neg_target == data_utils.EOS_ID:
                        neg_flag = True
                batch_pos_weights.append(batch_pos_weight)
                batch_neg_weights.append(batch_neg_weight)

            batch_encoder_inputs = (batch_pos_encoder_inputs, batch_neg_encoder_inputs)
            batch_target_weights = (batch_pos_weights, batch_neg_weights)

            yield batch_encoder_inputs, batch_target_weights, i
            i = end

    def train(self):
        print ('---------prepare data---------')
        pos_id_path, neg_id_path, vocab_path = data_utils.prepare_diologue(self.works_dir, self.data_path, self.vocab_size)
        print ('loading dictionary...')
        vocab, rev_vocab = data_utils.initialize_vocab(vocab_path)
        print ('loading pos set form %s...' %pos_id_path)
        pos_set = data_utils.read_data(pos_id_path, 1.0)
        print ('loading neg set from %s...' %neg_id_path)
        neg_set = data_utils.read_data(neg_id_path, 0.0)

        print ('---------building model---------')
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(self.works_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Reading model parameters from %s @ %s" % (ckpt.model_checkpoint_path, datetime.now()))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model reloaded @ %s" % (datetime.now()))
        else:
            print ('Creating new parameters @ %s'  % (datetime.now()))

        pos_seq_loss, neg_seq_loss = 0.0, 0.0

        if self.pretrain:
            print ('--------start pretraining-------')
            for i in range(2000):
                batch_encoder_inputs, batch_target_weights = self.get_batch(pos_set, neg_set)
                pos_sequence_loss, neg_sequence_loss = self.pretrain_step(self.sess, batch_encoder_inputs, batch_target_weights)

                pos_seq_loss += pos_sequence_loss / self.checkpoint_step
                neg_seq_loss += neg_sequence_loss / self.checkpoint_step

                if (i+1) % self.checkpoint_step == 0:
                    print ("iter: ", i, "pos_perplexity %.4f neg_d_loss %.4f @ %s" %(math.exp(pos_seq_loss), math.exp(neg_seq_loss), datetime.now()))
                    pos_seq_loss, neg_seq_loss = 0.0, 0.0

        g_loss, pos_d_loss, neg_d_loss, pos_seq_loss, neg_seq_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        current_step = 0

        print ('---------start training---------')
        while True:
            if current_step == 200000:
                break
            batch_encoder_inputs, batch_target_weights = self.get_batch(pos_set, neg_set)
            gen_loss, pos_adv_loss, neg_adv_loss, pos_sequence_loss, neg_sequence_loss = self.g_step(self.sess, batch_encoder_inputs, batch_target_weights, train=True)
            pos_dis_loss, neg_dis_loss, pos_adv_loss, neg_adv_loss = self.d_step(self.sess, batch_encoder_inputs, train=True)

            g_loss += gen_loss / self.checkpoint_step
            pos_d_loss += pos_dis_loss / self.checkpoint_step
            neg_d_loss += neg_dis_loss / self.checkpoint_step
            pos_seq_loss += pos_sequence_loss / self.checkpoint_step
            neg_seq_loss += neg_sequence_loss / self.checkpoint_step

            # print (current_step)
            current_step += 1
            if current_step % self.checkpoint_step == 0:
                print ('global step', self.sess.run(self.global_step), end='')
                print (" g_loss %.4f pos_d_loss %.4f neg_d_loss %.4f " %(g_loss, pos_d_loss, neg_d_loss), end='')
                print ("pos_perplexity %.4f neg_d_loss %.4f @ %s" %(math.exp(pos_seq_loss), math.exp(neg_seq_loss), datetime.now()))
                g_loss, pos_d_loss, neg_d_loss, pos_seq_loss, neg_seq_loss = 0.0, 0.0, 0.0, 0.0, 0.0

            if current_step % (5*self.checkpoint_step) == 0:
                checkpoint_path = os.path.join(self.works_dir, "model.ckpt")
                self.saver.save(self.sess, checkpoint_path)
                true_pos_id, true_neg_id, fake_pos_id, fake_neg_id = self.get_sentence(self.sess, batch_encoder_inputs)
                print ('fake_pos:', self.print_sentence(fake_pos_id, rev_vocab, 0))
                print ('fake_neg:', self.print_sentence(fake_neg_id, rev_vocab, 0))
                print ('true_pos:', self.print_sentence(true_pos_id, rev_vocab, 0))
                print ('true_neg:', self.print_sentence(true_neg_id, rev_vocab, 0))

    def chat(self):
        print ('---------prepare data---------')
        _, _, vocab_path = data_utils.prepare_diologue(self.works_dir, self.data_path, self.vocab_size)
        print ('loading dictionary...')
        vocab, rev_vocab = data_utils.initialize_vocab(vocab_path)
        print ('---------building model---------')
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(self.works_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Reading model parameters from %s @ %s" % (ckpt.model_checkpoint_path, datetime.now()))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model reloaded @ %s" % (datetime.now()))
        else:
            raise ValueError("can't find the path" )

        sentence = input('Input: ')
        pat = re.compile('(\W+)')

        while sentence:
            true_neg_id, fake_pos_id = self.test_sentence(self.sess, sentence, vocab, rev_vocab, pat)
            print ('fake_pos:', self.print_sentence(fake_pos_id, rev_vocab, 0))
            print ('true_neg:', self.print_sentence(true_neg_id, rev_vocab, 0))
            sentence = input('Input: ')

    def test(self):
        print ('---------prepare data---------')
        data_id_path, vocab_path = data_utils.prepare_test_data(self.works_dir, self.data_path, self.vocab_size)
        print ('loading dictionary...')
        vocab, rev_vocab = data_utils.initialize_vocab(vocab_path)
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(self.works_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Reading model parameters from %s @ %s" % (ckpt.model_checkpoint_path, datetime.now()))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model reloaded @ %s" % (datetime.now()))
        else:
            raise ValueError("can't find the path" )

        print ('loading data set form %s...' %data_id_path)
        data_set = data_utils.read_data(data_id_path, 1.0)
        fout = open(os.path.join(self.works_dir, 'seq2seq_pos.txt'), 'w')
        for batch_encoder_inputs, _, i in self.get_batch_epoch(data_set, data_set):
            _, _, fake_pos_id, _ = self.get_sentence(self.sess, batch_encoder_inputs)
            data_size = self.batch_size
            if i + self.batch_size > len(data_set):
                data_size = len(data_set) - i
            print ('\r', data_size, i, end='')

            for t in range(data_size):
                fout.write(self.print_sentence(fake_pos_id, rev_vocab, t)+'\n')
        
    def print_sentence(self, result, rev_vocab, index):
        outputs = [ r[index] for r in result ]
        if data_utils.EOS_ID in outputs:
            eos = outputs.index(data_utils.EOS_ID)
            outputs = outputs[:eos]
        output_sentence = ' '.join([rev_vocab.get(t) for t in outputs])
        return output_sentence

    def test_sentence(self, sess, sentence, vocab, rev_vocab, pat):
        feed_data = data_utils.sentence_to_id(' '.join(re.split(pat, sentence.lower())), vocab)
        pos_dummy = [([0], 1)]
        batch_encoder_inputs, _ = self.get_batch(pos_dummy, [(feed_data, 0)])
        _, true_neg_id, fake_pos_id, _ = self.get_sentence(self.sess, batch_encoder_inputs)
        return true_neg_id, fake_pos_id
        





