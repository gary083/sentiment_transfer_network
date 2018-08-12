# -*- coding: utf-8 -*-
# File: scope_utils.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import six
if six.PY2:
    import functools32 as functools
else:
    import functools

import math
import numpy as np 
import tensorflow as tf


class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def conv2d(input_, output_dim, 
       k_h=8, k_w=8, d_h=1, d_w=1, stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def max_pooling(input_, output_dim):
  pooled = tf.reduce_max(input_, reduction_indices=1)
  pooled = tf.reshape(pooled, [-1, output_dim])
  return pooled

def conv1d(input_, input_dim, output_dim, width, name="conv1d"):
  with tf.variable_scope(name):
    input_ = tf.expand_dims(input_, axis=1)
    filter_ = tf.get_variable('filter', shape=[1, width, int(input_dim), int(output_dim)])
    conv = tf.nn.conv2d(input_, filter=filter_, strides=[1,1,1,1], padding='SAME')
    output = tf.squeeze(conv, axis=1)
    return output

def ResBlock2(inputs, num_layer, name):
  output = lrelu(inputs)
  output = conv1d(output, num_layer, num_layer, 1, name=name+'.1')
  output = lrelu(output)
  output = conv1d(output, num_layer, num_layer, 1, name=name+'.2')
  return output

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, name=None, stddev=0.02, bias_start=0.0, with_b=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(name or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], int(output_size)], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    if with_b:
      bias = tf.get_variable("bias", [int(output_size)], initializer=tf.constant_initializer(bias_start))
      return tf.matmul(input_, matrix) + bias
    else:
      return tf.matmul(input_, matrix) 
