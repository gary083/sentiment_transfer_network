import tensorflow as tf 
from lib.ops import *

class generator(object):
    def __init__(self, latent_dim, word_embedding, vocab_size, temperature, attention):
        self.latent_dim = latent_dim
        self.word_embedding = word_embedding
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.attention = attention
        # cell
        self.cell = tf.contrib.rnn.LSTMCell(num_units=self.latent_dim, state_is_tuple=True)
        # variables
        if self.attention:
            self.v = tf.get_variable(name="v", shape=[self.latent_dim, 1])
            self.w_h = tf.get_variable(name="w_h", shape=[self.latent_dim, self.latent_dim])
            self.w_s = tf.get_variable(name="w_s", shape=[self.latent_dim, self.latent_dim])
            self.w_c = tf.get_variable(name="w_c", shape=[self.latent_dim])
            self.b_attn = tf.get_variable(name="b_attn", shape=[self.latent_dim])

    def __call__(self, 
                encoder_state, 
                encoder_outputs, 
                sentiment_code, 
                decoder_inputs, 
                sample_rate, 
                feed_previous, 
                scope):

        emb_decoder_inputs = [tf.nn.embedding_lookup(self.word_embedding, d) for d in decoder_inputs] 
        encoder_outputs = tf.unstack(encoder_outputs, axis=1)
        self.encoder_shape = [tf.shape(decoder_inputs[0])[0], len(encoder_outputs)]

        state = encoder_state
        prev_output = tf.concat([emb_decoder_inputs[0], sentiment_code], axis=1)

        if self.attention:
            atten_coverage = tf.zeros([self.encoder_shape[0], self.encoder_shape[1]])
            atten_weight, atten_output = self._attnetion(state.h, atten_coverage, encoder_outputs)

        decoder_outputs_distribution = []
        decoder_outputs_id = []
        decoder_outputs = []

        for i in range(len(emb_decoder_inputs)-1):
            if i > 0:
                scope.reuse_variables()
                prev_output = tf.concat([tf.matmul(decoder_dist, self.word_embedding), sentiment_code], axis=1)
                if not feed_previous:
                    hard_inputs = tf.concat([emb_decoder_inputs[i], sentiment_code], axis=1)
                    sample_prob = tf.reduce_sum(tf.random_uniform([], seed=1))
                    # big sample_rate tends to choice hard_inputs.
                    prev_output = tf.cond(sample_prob > sample_rate, lambda: prev_output, lambda: hard_inputs) 


            if self.attention:
                cell_input = linear(tf.concat([prev_output, atten_output], axis=1), self.latent_dim, name='input_projection')
            else:
                cell_input = linear(prev_output, self.latent_dim, name='input_projection')

            cell_output, state = self.cell(cell_input, state)

            if self.attention:
                atten_coverage += atten_weight
                atten_weight, atten_output = self._attnetion(state.h, atten_coverage, encoder_outputs)
                decoder_output = linear(tf.concat([atten_output, state.h], axis=1), self.latent_dim, name='attention_projection')
            else:
                decoder_output = state.h

            decoder_dist = self._get_distribution(decoder_output)
            decoder_outputs.append(decoder_output)
            decoder_outputs_distribution.append(decoder_dist)
            decoder_outputs_id.append(tf.argmax(decoder_dist, axis=1))

        print ('distribution size:', decoder_outputs_distribution[0].get_shape().as_list())
        print ('id size:', decoder_outputs_id[0].get_shape().as_list())
        print ('output size:', decoder_outputs[0].get_shape().as_list())

        decoder_outputs = tf.stack(decoder_outputs, axis=1)
        decoder_outputs_distribution = tf.stack(decoder_outputs_distribution, axis=1)
        return decoder_outputs_distribution, decoder_outputs_id, decoder_outputs

    def _attnetion(self, state, atten_c, encoder_outputs):
        e = []
        attention_state = tf.stack(encoder_outputs, axis=1)
        atten_c = tf.split(atten_c, num_or_size_splits=self.encoder_shape[1], axis=1)

        for h, c in zip(encoder_outputs, atten_c):
            hidden = tf.tanh(tf.matmul(h, self.w_h) + tf.matmul(state, self.w_s) + self.w_c*c + self.b_attn)
            e_t = tf.squeeze(tf.matmul(hidden, self.v), axis=1)
            e.append(e_t)
        atten_weight = tf.nn.softmax(tf.stack(e, axis=1))
        atten_output = tf.squeeze(tf.matmul(tf.expand_dims(atten_weight, axis=1), attention_state), axis=1)
        return atten_weight, atten_output

    def _get_distribution(self, decoder_output):
        word_distribution = linear(decoder_output, self.vocab_size, name='output_projection_1')
        # word_distribution = linear(hidden, self.vocab_size, name='output_projection_2')
        return tf.nn.softmax(word_distribution / self.temperature)

    def get_sequence_loss(self, decoder_outputs_distribution, decoder_targets, decoder_weights):
        decoder_outputs_distribution = tf.unstack(decoder_outputs_distribution, axis=1)
        log_prep = []
        for prob, target, weight in zip(decoder_outputs_distribution, decoder_targets, decoder_weights):
            target_prob = tf.reduce_max(prob * tf.one_hot(target, self.vocab_size), axis=1)
            cross_entropy = -tf.log(tf.clip_by_value(target_prob, 1e-10, 1.0))
            log_prep.append(cross_entropy * weight)
        total_log_prep = tf.add_n(log_prep)
        total_size = 1e-12 + tf.add_n(decoder_weights)
        sequence_loss = tf.reduce_mean(total_log_prep/total_size)

        return sequence_loss
        







