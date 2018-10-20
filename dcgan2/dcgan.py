"""
该DCGAN结构更加符合论文
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from data_loader import DataLoader
from keras.layers import Dense, Reshape, Conv2D, UpSampling2D, BatchNormalization, ReLU, Activation, Input, LeakyReLU, \
    Flatten, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


def discriminator_loss(y_true, y_pred):
    return y_true * K.mean(K.maximum(1. - y_pred, 0.), axis=-1) \
           + (1. - y_true) * K.mean(K.maximum(1. + y_pred, 0.), axis=-1)


def generator_loss(_, y_pred):
    return - K.mean(y_pred)


class DCGAN:

    def __init__(self, image_shape=(64, 64, 1), dataset_name='mnist', latent_dim=100, gf_dim=64, df_dim=64):
        self.img_rows, self.img_cols, self.channels = self.img_shape = image_shape
        self.latent_dim = latent_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.dataset_name = dataset_name
        self.data_loader = DataLoader(dataset_name)

        optimizer = Adam(0.0002, 0.5)

        # Build discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer=optimizer, loss=discriminator_loss, metrics=["accuracy"])

        # Build generator
        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dim,))
        gen_img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(gen_img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(optimizer=optimizer,
                              loss=generator_loss)

    def build_generator(self):
        s_h, s_w = self.img_rows, self.img_cols
        s_h16, s_w16 = math.ceil(s_h / 16), math.ceil(s_w / 16)

        model = Sequential()
        model.add(Dense(self.gf_dim * 8 * s_h16 * s_h16, input_dim=self.latent_dim))
        model.add(Reshape((s_h16, s_w16, self.gf_dim * 8)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(ReLU())
        model.add(UpSampling2D())
        model.add(Conv2D(self.gf_dim * 4, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(ReLU())
        model.add(UpSampling2D())
        model.add(Conv2D(self.gf_dim * 2, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(ReLU())
        model.add(UpSampling2D())
        model.add(Conv2D(self.gf_dim * 1, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(ReLU())
        model.add(UpSampling2D())
        model.add(Conv2D(self.channels, kernel_size=5, padding="same"))

        model.add(Activation('tanh'))

        print("Structure: generator")
        model.summary()

        noise = Input(shape=(self.latent_dim,))

        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(self.df_dim * 1, kernel_size=5, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(self.df_dim * 2, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(self.df_dim * 4, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(self.df_dim * 8, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU())
        model.add(Flatten())
        model.add(Dense(1, activation=None))

        print("Structure: discriminator")
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            imgs = self.data_loader.load_data(batch_size)

            # Train discriminator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(f"images/{self.dataset_name}/{epoch}.png")
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)
