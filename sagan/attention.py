from keras import backend as K
from keras.engine.topology import Layer

import math


class Attention2D(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Attention2D, self).__init__(**kwargs)

    def build(self, input_shape):
        s_h, s_w, n_c = input_shape
        self.w_g = self.add_weight(name="Wg",
                                   shape=(n_c, math.ceil(n_c / 8)),
                                   initializer='xavier',
                                   trainable=True)
        self.w_f = self.add_weight(name="Wf",
                                   shape=(n_c, math.ceil(n_c / 8)),
                                   initializer='uniform',
                                   trainable=True)
        self.w_h = self.add_weight(name="Wh",
                                   shape=(n_c, n_c),
                                   initializer='uniform',
                                   trainable=True)
        super(Attention2D, self).build(input_shape)

    def call(self, x, **kwargs):

        # Reshape x to (C, N), where N=s_h * s_w
        s_h, s_w, n_c = self.input_shape
        x = K.reshape(x, (s_h * s_w, n_c))
        x = K.permute_dimensions(x, (1, 0))

        g = K.dot(self.w_g, x)
        f = K.dot(self.w_f, x)
        s = K.dot(f, g)
