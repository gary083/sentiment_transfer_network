import tensorflow as tf 
from lib.ops import *

class discriminator(object):
    def __init__(self, latent_dim, max_length, dim):
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.dim = dim

    def __call__(self, decoder_outputs):
        outputs = conv1d(decoder_outputs, self.latent_dim, self.dim, 1, name='discriminator.Input')
        outputs = self.ResBlock(name='discriminator.1', inputs=outputs)
        outputs = self.ResBlock(name='discriminator.2', inputs=outputs)
        outputs = self.ResBlock(name='discriminator.3', inputs=outputs)
        outputs = self.ResBlock(name='discriminator.4', inputs=outputs)
        outputs = tf.reshape(outputs, [-1, self.max_length*self.dim])
        outputs = linear(outputs, 1, name='discriminator.Output')
        return outputs

    def ResBlock(self, name, inputs):
        outputs = inputs
        outputs = tf.nn.relu(outputs)
        outputs = conv1d(outputs, self.dim, self.dim, 3, name=name+'.1')
        outputs = tf.nn.relu(outputs)
        outputs = conv1d(outputs, self.dim, self.dim, 3, name=name+'.2')
        return inputs + (outputs * 0.3)

class discriminator_2(object):
    def __init__(self, latent_dim, max_length, dim):
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.dim = dim
        self.filter_size = [3, 4, 5]

    def __call__(self, decoder_outputs):
        input_ = tf.expand_dims(decoder_outputs, -1)
        outputs = []
        for size in self.filter_size:
            filter_name = 'fileter_'+str(size)
            outputs.append(self.CNNBlock(name=filter_name, inputs=input_, filter_size=size))
        outputs = tf.concat(outputs, 1)
        outputs = tf.nn.dropout(outputs, 0.5)
        outputs = linear(outputs, 1, name='CNNC_output')
        return outputs

    def CNNBlock(self, name, inputs, filter_size):
        outputs = conv2d(inputs, output_dim=self.dim, k_h=filter_size, k_w=self.latent_dim, name=name)
        outputs = lrelu(outputs)
        outputs = max_pooling(outputs, self.dim)
        return outputs
