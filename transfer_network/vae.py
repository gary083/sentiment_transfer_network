import tensorflow as tf 
import tensorflow.contrib.layers as tcl
import numpy as np 
import random, math
from six.moves import xrange

from lib.seq2seq import encoder, decoder, rnn_decoder, bi_encoder
from lib import data_utils
from lib.ops import *

class v_autoencoder(object):
    def __init__(self, sess, args, feed_previous=True):
        self.sess = sess
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.feed_previous = feed_previous
        self.vocab_size = args.vocab_size
        self.emb_size = args.emb_size
        self.num_cell = 128
        self.num_layers = 1
        self.region = 10000
        self.phase = not feed_previous
        self.sample_rate = tf.Variable(float(0), trainable=False)
        self.kl_weight = tf.Variable(float(0), trainable=False)
        self.trans_weight = tf.Variable(float(0), trainable=False)
        # 
        self.global_step = tf.Variable(0, trainable=False)
        self.class_global_step = tf.Variable(0, trainable=False)
        self.trans_global_step = tf.Variable(0, trainable=False)
        self.build_graph()  
        self.sample_rate_op = self.sample_rate.assign( tf.maximum(0.5, 0.9-(tf.cast(self.global_step, tf.float32)/100)*0.001) )
        self.kl_weight_op = self.kl_weight.assign(-1/(0.5*(tf.exp(10*tf.cast(self.global_step, tf.float32)/10000) + tf.exp(-10*tf.cast(self.global_step, tf.float32)/10000))) + 1)
        self.trans_weight_op = self.trans_weight.assign(tf.minimum(0.4, self.trans_weight/10000))
        self.class_rate_op = self.class_rate.assign( tf.maximum(0.5, 0.9-tf.cast(self.class_global_step, tf.float32)/10000) ) 
        self.saver = tf.train.Saver(self.ae_save_vars)
        self.class_saver = tf.train.Saver(self.class_save_vars)
        self.trans_saver = tf.train.Saver(self.trans_save_vars)


        # 0.5*(tf.exp(10*self.global_step/10000) + tf.exp(-10*self.global_step/10000))

    def encoder(self, encoder_inputs, embedding):
        # encoder
        cell_fw = tf.contrib.rnn.LSTMCell(num_units=self.num_cell/2, state_is_tuple=True)
        cell_bw = tf.contrib.rnn.LSTMCell(num_units=self.num_cell/2, state_is_tuple=True)
        encoder_outputs, encoder_state_c, encoder_state_h = bi_encoder( encoder_inputs, embedding, cell_fw, cell_bw )
        return encoder_outputs, encoder_state_c, encoder_state_h

    def decoder(self, decoder_inputs, encoder_state, embedding, output_projection ):
        cell = tf.contrib.rnn.LSTMCell(num_units=self.num_cell, state_is_tuple=True)
        sample_prob = tf.reduce_sum(tf.random_uniform([], seed=1))

        return rnn_decoder( decoder_inputs, encoder_state, embedding, cell, output_projection, self.feed_previous, sample_prob, self.sample_rate )

    def sample_gumbel(self, shape, eps=1e-20):
        U = tf.random_uniform(shape, minval=0, maxval=1)
        return -tf.log(-tf.log(U + eps) + eps)

    def gumbel_softmax(self, logits, temperature):
        y = tf.nn.softmax(logits / temperature)
        return y

    def classifier(self, encoder_inputs, class_embedding, train=True):
        cell_fw = tf.contrib.rnn.LSTMCell(num_units=self.num_cell/2, state_is_tuple=True)
        cell_bw = tf.contrib.rnn.LSTMCell(num_units=self.num_cell/2, state_is_tuple=True)
        if train:
            emb_encoder_inputs = [tf.nn.embedding_lookup(class_embedding, e) for e in encoder_inputs]
        else:
            emb_encoder_inputs = [tf.matmul(e, class_embedding) for e in encoder_inputs]
        encoder_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, emb_encoder_inputs, dtype=tf.float32 )
        h1 = tf.nn.relu(linear(encoder_outputs[-1], self.num_cell, name='class_linear1'))
        h2 = tf.nn.relu(linear(h1, self.num_cell/2, name='class_linear2'))
        pred = linear(h2, 1, name='class_linear3')
        return pred

    def transer_network(self, inputs, inputs_c, phase):
        h1 = tf.nn.relu(linear(inputs, self.num_cell, name='linear1'))
        h2 = tf.nn.relu(linear(h1, self.num_cell, name='linear2'))
        h3 = tf.nn.relu(linear(h2, self.num_cell, name='linear3'))
        h4 = tf.nn.relu(linear(h3, self.num_cell, name='linear4'))
        h5 = linear(h4, self.num_cell, name='linear5')
        h1_c = tf.nn.relu(linear(inputs_c, self.num_cell, name='linear1_c'))
        h2_c = tf.nn.relu(linear(h1_c, self.num_cell, name='linear2_c'))
        h3_c = tf.nn.relu(linear(h2_c, self.num_cell, name='linear3_c'))
        h4_c = tf.nn.relu(linear(h3_c, self.num_cell, name='linear4_c'))
        h5_c = linear(h4_c, self.num_cell, name='linear5_c')
        return h5, h5_c

    def eval_accuracy(self, pred, labels):
        pred_sig = tf.cast(tf.greater(tf.sigmoid(pred), 0.5), tf.float32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_sig, labels), tf.float32))
        return accuracy

    def sample(self, outputs):
        mean = tf.slice(outputs, [0, 0], [-1, self.num_cell])
        logvar = tf.slice(outputs, [0, self.num_cell], [-1, self.num_cell])
        var = tf.exp( 0.5 * logvar )
        noise = tf.random_normal(tf.shape(var))
        encoder_state_h = mean + tf.multiply(var,noise)
        return encoder_state_h

    def build_graph(self):
        with tf.variable_scope('data_process') as scope:
            # placholder
            self.encoder_inputs = []
            self.decoder_inputs = []
            self.target_weights = []
            self.target_masks   = []
            for i in xrange(self.max_length):
                self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
                self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
                self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))
                self.target_masks.append(tf.placeholder(tf.float32, shape=[None, self.vocab_size], name="mask{0}".format(i)))
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(self.max_length)))
            self.targets = self.decoder_inputs[1:]
            self.inter_sample_h = tf.placeholder(tf.float32, shape=[None, self.num_cell], name='h_inter')
            self.inter_encoder_c = tf.placeholder(tf.float32, shape=[None, self.num_cell], name='c_inter')
            self.labels = tf.placeholder(tf.float32, shape=[None, 1], name='labels')
    
        with tf.variable_scope('ae_encoder') as scope:
            embedding = tf.get_variable('embedding', shape=[self.vocab_size, self.emb_size])
            self.encoder_outputs, self.encoder_state_c, self.encoder_state_h = self.encoder( self.encoder_inputs, embedding)

            self.mean   = linear(self.encoder_state_h, self.num_cell, name='vae_mean', stddev=0.001)
            self.logvar = linear(self.encoder_state_h, self.num_cell, name='vae_var', stddev=0.001)
            self.var = tf.exp( 0.5 * self.logvar )
            noise = tf.random_normal(tf.shape(self.var))
            self.encoder_sample_h = self.mean + tf.multiply(self.var,noise)
            # self.encoder_sample_c = linear(self.encoder_sample_h, self.num_cell, name='encoder_sample_c', stddev=0.001)
            self.encoder_state = tf.contrib.rnn.LSTMStateTuple(c=self.encoder_state_c, h=self.encoder_sample_h) 
            
        with tf.variable_scope('ae_decoder') as scope:
            proj_w_t = tf.get_variable("proj_w_t", [self.vocab_size, self.num_cell], dtype=tf.float32)
            proj_w = tf.transpose(proj_w_t)
            proj_b = tf.get_variable("proj_b", [self.vocab_size], dtype=tf.float32)
            output_projection = (proj_w, proj_b)
            def sampled_loss(labels, inputs):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(proj_w_t, proj_b, labels, inputs, self.num_cell, self.vocab_size)
            softmax_loss_function = sampled_loss
            self.decoder_outputs = self.decoder(self.decoder_inputs, self.encoder_state, embedding, output_projection )
        
        with tf.variable_scope('trans_') as scope:
            self.trans_encoder_h, self.trans_encoder_c = self.transer_network(self.mean, self.encoder_state_c, self.phase)

        trans_encoder_state = tf.contrib.rnn.LSTMStateTuple(c=self.encoder_state_c, h=self.trans_encoder_h) 
        inter_encoder_state = tf.contrib.rnn.LSTMStateTuple(c=self.inter_encoder_c, h=self.inter_sample_h) 

        with tf.variable_scope('ae_decoder', reuse=True):
            trans_decoder_outputs = self.decoder(self.decoder_inputs, trans_encoder_state, embedding, output_projection )
            inter_decoder_outputs = self.decoder(self.decoder_inputs, inter_encoder_state, embedding, output_projection)
        
        self.predict_prob = [self.gumbel_softmax(tf.nn.xw_plus_b(d, proj_w, proj_b), temperature=0.0001) for d in trans_decoder_outputs[:self.max_length]]
        self.outputs_prob = [self.gumbel_softmax(tf.nn.xw_plus_b(d, proj_w, proj_b), temperature=0.0001) for d in self.decoder_outputs[:self.max_length]] 
        # self.outputs_prob = [self.gumbel_softmax(tf.nn.xw_plus_b(self.decoder_outputs[i], proj_w, proj_b), temperature=0.5)*self.target_masks[i] for i in xrange(self.max_length)]
        # self.outputs_prob = [tf.nn.softmax(tf.nn.xw_plus_b(d, proj_w, proj_b)) for d in self.decoder_outputs[:self.max_length]]
        # self.predict_prob = [tf.nn.softmax(tf.nn.xw_plus_b(d, proj_w, proj_b)) for d in trans_decoder_outputs[:self.max_length]]
        self.predict_outputs = [tf.argmax(tf.nn.xw_plus_b(d, proj_w, proj_b), axis=1) for d in self.decoder_outputs[:self.max_length]]
        self.trans_predict_outputs = [tf.argmax(tf.nn.xw_plus_b(d, proj_w, proj_b), axis=1) for d in trans_decoder_outputs[:self.max_length]]
        self.inter_predict_outputs = [tf.argmax(tf.nn.xw_plus_b(d, proj_w, proj_b), axis=1) for d in inter_decoder_outputs[:self.max_length]]

        with tf.variable_scope('sentiment_') as scope:
            self.class_rate = tf.Variable(float(0), trainable=False)
            class_embedding = tf.get_variable('class_embedding', shape=[self.vocab_size, self.emb_size])
            pred = self.classifier(self.encoder_inputs, class_embedding, True)

        # neg -> pos
        with tf.variable_scope('sentiment_', reuse=True):
            self.test_score = tf.sigmoid(pred)
            post_pred = self.classifier(self.predict_prob, class_embedding, False)
            # self.test_score = tf.sigmoid(self.classifier(self.outputs_prob, class_embedding, False))
            self.score = tf.sigmoid(post_pred)

        self.en_vars          = [var for var in tf.trainable_variables() if 'ae_encoder' in var.name]
        self.de_vars          = [var for var in tf.trainable_variables() if 'ae_decoder' in var.name]
        self.train_ae_vars    = self.en_vars + self.de_vars
        self.train_class_vars = [var for var in tf.trainable_variables() if 'sentiment_' in var.name]
        self.train_trans_vars = [var for var in tf.trainable_variables() if 'trans_' in var.name]
        
        with tf.variable_scope('ae_updates_'):
            self.seq_loss   = tf.contrib.legacy_seq2seq.sequence_loss(self.decoder_outputs[:self.max_length], self.targets, self.target_weights, softmax_loss_function=softmax_loss_function) 
            self.kl_loss    = tf.reduce_mean(tf.reduce_sum( -0.5 * (self.logvar - tf.square(self.mean) - tf.exp(self.logvar) + 1.0) , 1), 0) 
            self.ae_loss    = self.seq_loss + self.kl_weight * self.kl_loss
            gradients       = tf.gradients(ys=self.ae_loss, xs=self.train_ae_vars)
            clipped_gradient, _ = tf.clip_by_global_norm(gradients, 1)
            self.ae_updates = tf.train.GradientDescentOptimizer(1).apply_gradients(zip(clipped_gradient, self.train_ae_vars), global_step=self.global_step)

        with tf.variable_scope('class_updates_'):
            self.accuracy      = self.eval_accuracy(pred, self.labels)
            pre_loss           = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=self.labels)) 
            # self.class_loss    = 0.1 * post_loss + 0.9 * pre_loss 
            self.class_loss    = pre_loss
            self.class_updates = tf.train.AdamOptimizer(0.001).minimize(loss=self.class_loss, var_list=self.train_class_vars, global_step=self.class_global_step)

        with tf.variable_scope('trans_updates_'):
            self.style_loss = 1 - tf.reduce_mean(self.score)
            # self.content_loss = tf.reduce_mean(tf.square(self.encoder_state_h - self.trans_encoder_h))
            # self.K = -self.num_cell * tf.log((2*math.pi)) - tf.log(self.var) + 2 * self.region
            # self.trans_distance = tf.divide(tf.square(self.trans_encoder_h - self.mean), self.var)
            # self.content_loss = tf.reduce_mean(tf.log(self.K - self.trans_distance))

            self.content_loss_h = tf.sqrt(tf.reduce_mean(tf.square(self.trans_encoder_h - self.mean)))
            self.content_loss_c = tf.sqrt(tf.reduce_mean(tf.square(self.trans_encoder_c - self.encoder_state_c)))
            self.content_loss   = self.content_loss_h #+ self.content_loss_c
            self.trans_loss     = self.style_loss #+ 0.01 * self.content_loss
            self.trans_updates     = tf.train.AdamOptimizer(0.0005).minimize(loss=self.trans_loss, var_list=self.train_trans_vars, global_step=self.trans_global_step)
            self.pre_trans_updates = tf.train.AdamOptimizer(0.0005 ).minimize(loss=self.content_loss, var_list=self.train_trans_vars)


        self.ae_up_vars    = [var for var in tf.trainable_variables() if 'ae_updates_' in var.name]
        self.ae_save_vars = self.train_ae_vars + self.ae_up_vars + [self.global_step]
        self.class_up_vars = [var for var in tf.trainable_variables() if 'class_updates_' in var.name]
        self.class_save_vars = self.train_class_vars + self.class_up_vars + [self.class_global_step]
        self.trans_up_vars = [var for var in tf.trainable_variables() if 'trans_updates_' in var.name]
        self.trans_save_vars = self.train_trans_vars + self.trans_up_vars + [self.trans_global_step]


    def ae_step(self, sess, encoder_inputs, decoder_inputs, target_weights, forward):
        input_feed = {}
        for l in xrange(self.max_length):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        input_feed[self.decoder_inputs[self.max_length].name] = np.zeros([self.batch_size], dtype=np.int32)

        output_feed = [self.seq_loss, self.kl_loss]
        if not forward:
            output_feed.append(self.ae_updates)
        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]

    def class_step(self, sess, encoder_inputs, decoder_inputs, masks, labels, forward):
        input_feed = {}
        for l in xrange(self.max_length):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_masks[l].name] = masks[l]
        input_feed[self.decoder_inputs[self.max_length].name] = np.zeros([self.batch_size], dtype=np.int32)
        input_feed[self.labels.name] = labels

        output_feed = [self.accuracy,  self.test_score]
        if not forward:
            output_feed.append(self.class_updates)
        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]


    def class_test(self, sess, encoder_inputs, decoder_inputs):
        input_feed = {}
        for l in xrange(self.max_length):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        input_feed[self.decoder_inputs[self.max_length].name] = np.zeros([self.batch_size], dtype=np.int32)

        output_feed = [self.test_score]
        outputs = sess.run(output_feed, input_feed)
        return outputs[0]

    def trans_step(self, sess, encoder_inputs, decoder_inputs, forward):    
        input_feed = {}
        for l in xrange(self.max_length):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        input_feed[self.decoder_inputs[self.max_length].name] = np.zeros([self.batch_size], dtype=np.int32)

        output_feed = [self.style_loss, self.content_loss]
        if not forward:
            output_feed.append(self.trans_updates)
        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]

    def pre_trans_step(self, sess, encoder_inputs, decoder_inputs, forward):    
        input_feed = {}
        for l in xrange(self.max_length):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        input_feed[self.decoder_inputs[self.max_length].name] = np.zeros([self.batch_size], dtype=np.int32)

        output_feed = [self.content_loss]
        if not forward:
            output_feed.append(self.pre_trans_updates)
        outputs = sess.run(output_feed, input_feed)
        return outputs[0]

    def ae_test(self, sess, encoder_inputs, decoder_inputs, masks):
        input_feed = {}
        for l in xrange(self.max_length):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_masks[l].name] = masks[l]
        input_feed[self.decoder_inputs[self.max_length].name] = np.zeros([self.batch_size], dtype=np.int32)

        output_feed = [self.encoder_sample_h, self.encoder_state_c]

        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]

    def inter_test(self, sess, h_inputs, c_inputs, decoder_inputs):
        input_feed = {}
        input_feed[self.inter_sample_h.name] = h_inputs
        input_feed[self.inter_encoder_c.name] = c_inputs
        for l in xrange(self.max_length):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        input_feed[self.decoder_inputs[self.max_length].name] = np.zeros([self.batch_size], dtype=np.int32)

        output_feed = [self.inter_predict_outputs]
        outputs = sess.run(output_feed, input_feed)
        return outputs[0]

    def chat_step(self, sess, encoder_inputs, decoder_inputs):
        input_feed = {}
        for l in xrange(self.max_length):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        input_feed[self.decoder_inputs[self.max_length].name] = np.zeros([self.batch_size], dtype=np.int32)

        output_feed = [self.predict_outputs]
        outputs = sess.run(output_feed, input_feed)
        return outputs[0]

    def trans_chat(self, sess, encoder_inputs, decoder_inputs):
        input_feed = {}
        for l in xrange(self.max_length):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        input_feed[self.decoder_inputs[self.max_length].name] = np.zeros([self.batch_size], dtype=np.int32)

        output_feed = [self.predict_outputs, self.trans_predict_outputs]
        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]

    def chat_get_batch(self, data):
        encoder_inputs, decoder_inputs = [], []
        for _ in xrange(self.batch_size):
            encoder_input = random.choice(data)
            decoder_input = encoder_input + [data_utils.EOS_ID]
            encoder_pad = [data_utils.PAD_ID] * (self.max_length - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            decoder_pad = [data_utils.PAD_ID] * (self.max_length - len(decoder_input) - 1)
            decoder_inputs.append([data_utils.GO_ID] + decoder_input + decoder_pad)   

        batch_encoder_inputs, batch_decoder_inputs, batch_masks = [], [], []

        for l in xrange(self.max_length):
            batch_encoder_inputs.append(np.array([encoder_inputs[b][l] for b in xrange(self.batch_size)], dtype=np.int32))
            batch_decoder_inputs.append(np.array([decoder_inputs[b][l] for b in xrange(self.batch_size)], dtype=np.int32))
            
            batch_mask = np.ones(shape=[self.batch_size, self.vocab_size], dtype=np.float32)
            for b in xrange(self.batch_size):
                if l < self.max_length - 1:
                    target = decoder_inputs[b][l+1]
                if l == self.max_length - 1 or target == data_utils.PAD_ID:
                    batch_mask[b] = np.zeros(shape=[self.vocab_size], dtype=np.float32)
            batch_masks.append(batch_mask)

        return batch_encoder_inputs, batch_decoder_inputs, batch_masks

    def get_batch(self, pos_data, neg_data):
        encoder_inputs, decoder_inputs = [], []
        for _ in xrange(int(self.batch_size/2)):
            encoder_input = random.choice(pos_data)
            decoder_input = encoder_input + [data_utils.EOS_ID]
            encoder_pad = [data_utils.PAD_ID] * (self.max_length - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            decoder_pad = [data_utils.PAD_ID] * (self.max_length - len(decoder_input) - 1)
            decoder_inputs.append([data_utils.GO_ID] + decoder_input + decoder_pad)

            encoder_input = random.choice(pos_data)
            decoder_input = encoder_input + [data_utils.EOS_ID]
            encoder_pad = [data_utils.PAD_ID] * (self.max_length - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            decoder_pad = [data_utils.PAD_ID] * (self.max_length - len(decoder_input) - 1)
            decoder_inputs.append([data_utils.GO_ID] + decoder_input + decoder_pad)            

        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        for l in xrange(self.max_length):
            batch_encoder_inputs.append(np.array([encoder_inputs[b][l] for b in xrange(self.batch_size)], dtype=np.int32))
            batch_decoder_inputs.append(np.array([decoder_inputs[b][l] for b in xrange(self.batch_size)], dtype=np.int32))

            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for b in xrange(self.batch_size):
                if l < self.max_length - 1:
                    target = decoder_inputs[b][l+1]
                if l == self.max_length - 1 or target == data_utils.PAD_ID:
                    batch_weight[b] = 0.0
            batch_weights.append(batch_weight)

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

    def class_get_batch(self, pos_data, neg_data, rate):
        encoder_inputs, decoder_inputs, labels = [], [], []
        decoder_pad = [data_utils.PAD_ID] * (self.max_length - 1)
        for _ in xrange(self.batch_size):
            prob = random.uniform(0, 1)
            if prob > rate:
                encoder_input = random.choice(pos_data)
                labels.append([1.0])
            elif prob <= rate:
                encoder_input = random.choice(neg_data)
                labels.append([0.0])
            encoder_pad = [data_utils.PAD_ID] * (self.max_length - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            decoder_inputs.append([data_utils.GO_ID] + decoder_pad)

        batch_encoder_inputs, batch_decoder_inputs, batch_masks = [], [], []

        for l in xrange(self.max_length):
            batch_encoder_inputs.append(np.array([encoder_inputs[b][l] for b in xrange(self.batch_size)], dtype=np.int32))
            batch_decoder_inputs.append(np.array([decoder_inputs[b][l] for b in xrange(self.batch_size)], dtype=np.int32))

            batch_mask = np.ones(shape=[self.batch_size, self.vocab_size], dtype=np.float32)
            for b in xrange(self.batch_size):
                if l < self.max_length - 1:
                    target = decoder_inputs[b][l+1]
                if l == self.max_length - 1 or target == data_utils.PAD_ID:
                    batch_mask[b] = np.zeros(shape=[self.vocab_size], dtype=np.float32)
            batch_masks.append(batch_mask)
        batch_labels = np.array(labels)

        return batch_encoder_inputs, batch_decoder_inputs, batch_masks, batch_labels

    def single_get_batch(self, data):
        encoder_inputs, decoder_inputs = [], []
        decoder_pad = [data_utils.PAD_ID] * (self.max_length - 1)
        for _ in xrange(self.batch_size):
            encoder_input = random.choice(data)
            encoder_pad = [data_utils.PAD_ID] * (self.max_length - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            decoder_inputs.append([data_utils.GO_ID] + decoder_pad)

        batch_encoder_inputs, batch_decoder_inputs = [], []
        for l in xrange(self.max_length):
            batch_encoder_inputs.append(np.array([encoder_inputs[b][l] for b in xrange(self.batch_size)], dtype=np.int32))
            batch_decoder_inputs.append(np.array([decoder_inputs[b][l] for b in xrange(self.batch_size)], dtype=np.int32))

        return batch_encoder_inputs, batch_decoder_inputs

    def train_get_batch(self, data):
        encoder_inputs, decoder_inputs = [], []
        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data)
            encoder_pad = [data_utils.PAD_ID] * (self.max_length - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            decoder_pad = [data_utils.PAD_ID] * (self.max_length - len(decoder_input) - 1)
            decoder_inputs.append([data_utils.GO_ID] + decoder_input + decoder_pad)

        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        for l in xrange(self.max_length):
            batch_encoder_inputs.append(np.array([encoder_inputs[b][l] for b in xrange(self.batch_size)], dtype=np.int32))
            batch_decoder_inputs.append(np.array([decoder_inputs[b][l] for b in xrange(self.batch_size)], dtype=np.int32))

            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for b in xrange(self.batch_size):
                if l < self.max_length - 1:
                    target = decoder_inputs[b][l+1]
                if l == self.max_length - 1 or target == data_utils.PAD_ID:
                    batch_weight[b] = 0.0
            batch_weights.append(batch_weight)

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights