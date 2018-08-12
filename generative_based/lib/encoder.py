import tensorflow as tf 
from lib.ops import *

class encoder(object):
    def __init__(self, latent_dim, word_embedding):
        self.latent_dim = latent_dim
        self.word_embedding = word_embedding
        # cell
        self.cell = tf.contrib.rnn.LSTMCell(num_units=self.latent_dim, state_is_tuple=True)

    def __call__(self, encoder_inputs, encoder_length):
        emb_encoder_inputs = [tf.nn.embedding_lookup(self.word_embedding, e) for e in encoder_inputs]
        emb_encoder_inputs = tf.stack(emb_encoder_inputs, axis=1)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=self.cell, 
                                                           dtype=tf.float32,
                                                           sequence_length=encoder_length,
                                                           inputs=emb_encoder_inputs)
        return encoder_outputs, encoder_state