from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Flatten
from keras.initializers import constant
import tensorflow as tf

from spectral_norm import ConvSN2D as Conv2D


def hw_flatten(x):
    # Input shape x: [BATCH, HEIGHT, WIDTH, CHANNELS]
    # flat the feature volume across the width and height dimensions
    x_shape = K.shape(x)
    return K.reshape(x, [x_shape[0], -1, x_shape[-1]])  # return [BATCH, W*H, CHANNELS]


class SelfAttention(Layer):

    def __init__(self, data_format=None, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.data_format = data_format

    def build(self, input_shape):
        self.final_shape = input_shape

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        num_of_filters = input_shape[channel_axis]
        self.f = Conv2D(num_of_filters // 8, 1, spectral_normalization=True,
                        strides=1, padding='SAME',
                        name='f_x', activation=None)
        self.g = Conv2D(num_of_filters // 8, 1, spectral_normalization=True,
                        strides=1, padding='SAME',
                        name='g_x', activation=None)
        self.h = Conv2D(num_of_filters, 1, spectral_normalization=True,
                        strides=1, padding='SAME',
                        name='h_x', activation=None)

        self.gamma = self.add_weight(name="gamma", initializer=constant(0.0), shape=[1])

        super(SelfAttention, self).build(input_shape)

    def call(self, x, **kwargs):

        f = self.f(x)
        g = self.g(x)
        h = self.h(x)

        f_flatten = hw_flatten(f)
        g_flatten = hw_flatten(g)
        h_flatten = hw_flatten(h)

        s = tf.matmul(g_flatten, f_flatten, transpose_b=True)  # [B,N,C] * [B, N, C] = [B, N, N]

        b = tf.nn.softmax(s, axis=-1)
        o = tf.matmul(b, h_flatten)
        y = self.gamma * tf.reshape(o, tf.shape(x)) + x

        return y

    def compute_output_shape(self, input_shape):
        return input_shape
