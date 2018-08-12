import tensorflow as tf 
import numpy as np
import copy


def encoder( encoder_inputs, embedding, cell ):
  emb_encoder_inputs = [tf.nn.embedding_lookup(embedding, e) for e in encoder_inputs]
  with tf.variable_scope('encoder') as scope:
    encoder_outputs, encoder_state = tf.contrib.rnn.static_rnn(cell, emb_encoder_inputs, dtype=tf.float32)
  return encoder_outputs, encoder_state

def bi_encoder( encoder_inputs, embedding, cell_fw, cell_bw ):
  emb_encoder_inputs = [tf.nn.embedding_lookup(embedding, e) for e in encoder_inputs]
  with tf.variable_scope('bi_encoder') as scope:
    encoder_outputs, encoder_state_fw, encoder_state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, emb_encoder_inputs, dtype = tf.float32 )
    encoder_state_c = tf.concat([encoder_state_fw.c, encoder_state_bw.c], axis=1)
    encoder_state_h = tf.concat([encoder_state_fw.h, encoder_state_bw.h], axis=1)
  return encoder_outputs, encoder_state_c, encoder_state_h

def gru_encoder( encoder_inputs, embedding, cell_fw, cell_bw ):
  emb_encoder_inputs = [tf.nn.embedding_lookup(embedding, e) for e in encoder_inputs]
  with tf.variable_scope('bi_encoder') as scope:
    encoder_outputs, encoder_state_fw, encoder_state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, emb_encoder_inputs, dtype = tf.float32 )
    encoder_state = tf.concat([encoder_state_fw, encoder_state_bw], axis=1)
  return encoder_outputs, encoder_state

def decoder( decoder_inputs, encoder_outputs, encoder_state, embedding, cell, output_projection, feed_previous ):
  num_cell = cell.output_size
  top_state = [tf.reshape(e, [-1, 1, num_cell]) for e in encoder_outputs]
  attention_states = tf.concat(top_state, 1)
  emb_decoder_inputs = [tf.nn.embedding_lookup(embedding, d) for d in decoder_inputs]

  with tf.variable_scope('decoder') as scope:
    def train_decoder_loop(prev, i):
      prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
      prev_index = tf.argmax(prev, axis=1)
      pred_prev = tf.nn.embedding_lookup(embedding, prev_index)
      return pred_prev

    loop_function = train_decoder_loop if feed_previous else None
    decoder_outputs, _ = tf.contrib.legacy_seq2seq.attention_decoder( 
      decoder_inputs = emb_decoder_inputs,
      initial_state = encoder_state,
      attention_states = attention_states,
      cell = cell,
      output_size = num_cell,
      loop_function = loop_function,
      scope = scope )  

    return decoder_outputs

def rnn_decoder( decoder_inputs, encoder_state, embedding, cell, output_projection, feed_previous, sample_prob, sample_rate ):
  emb_decoder_inputs = [tf.nn.embedding_lookup(embedding, d) for d in decoder_inputs]

  with tf.variable_scope('decoder') as scope:
    def train_decoder_loop(prev, i):
      prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
      prev_index = tf.argmax(prev, axis=1)
      pred_prev = tf.nn.embedding_lookup(embedding, prev_index)
      pred_prev_emb = tf.cond(sample_prob > sample_rate, lambda: pred_prev, lambda: emb_decoder_inputs[i] )
      return pred_prev_emb  

    def test_decoder_loop(prev, i):
      prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
      prev_index = tf.argmax(prev, axis=1)
      pred_prev = tf.nn.embedding_lookup(embedding, prev_index)
      return pred_prev  

    loop_function = train_decoder_loop if not feed_previous else test_decoder_loop
    decoder_outputs, _ = tf.contrib.legacy_seq2seq.rnn_decoder( 
      decoder_inputs = emb_decoder_inputs,
      initial_state = encoder_state,
      cell = cell,
      loop_function = loop_function,
      scope = scope )  

    return decoder_outputs