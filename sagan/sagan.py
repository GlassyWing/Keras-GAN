from __future__ import print_function, division

import datetime

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard
from keras.layers import BatchNormalization, Activation, ReLU
from keras.layers import Input, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam

from data_loader import DataLoader
from spectral_norm import ConvSN2D as Conv2D
from spectral_norm import ConvSN2DTranspose as Conv2DTranspose
from spectral_norm import DenseSN as Dense
from attention import SelfAttention

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.666
set_session(tf.Session(config=config))


def discriminator_loss(y_true, y_pred):
    return y_true * K.mean(K.maximum(1. - y_pred, 0.), axis=-1) \
           + (1. - y_true) * K.mean(K.maximum(1. + y_pred, 0.), axis=-1)


def generator_loss(_, y_pred):
    return - K.mean(y_pred)


class SAGAN:

    def __init__(self,
                 model_id=None,
                 image_shape=(64, 64, 1),
                 gf_dim=4,
                 gfc_dim=512,
                 df_dim=4,
                 dfc_dim=32,
                 latent_dim=100,
                 learning_rate_g=0.0001,
                 learning_rate_d=0.0004,
                 alpha=0.2,
                 beta1=0.0,
                 beta2=0.5,
                 dtype='float64',
                 dataset_name='mnist',
                 dataset_path="."):

        # Input shape
        self.img_rows, self.img_cols, self.channels = self.img_shape = image_shape
        self.latent_dim = latent_dim

        self.alpha = alpha
        self.dtype = dtype
        self.gf_dim = gf_dim
        self.gfc_dim = gfc_dim
        self.df_dim = df_dim
        self.dfc_dim = dfc_dim
        self.dataset_name = dataset_name
        self.data_loader = DataLoader(dataset_path=dataset_path, dataset_name=dataset_name)

        generator_optimizer = Adam(learning_rate_g, beta1)
        discriminator_optimizer = Adam(learning_rate_d, beta2)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=discriminator_loss,
                                   optimizer=discriminator_optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss=generator_loss, optimizer=generator_optimizer)

        if model_id is not None:
            self.combined.load_weights(f"saved_model/{dataset_name}/{model_id}.h5")

    def build_generator(self):

        def create_conv_transp(filters):
            return Conv2DTranspose(filters, self.gf_dim,
                                   padding="SAME", activation=None, dtype=self.dtype,
                                   use_bias=False, strides=2)

        model = Sequential()
        model.add(Dense(self.gf_dim * self.gf_dim * self.gfc_dim, input_dim=self.latent_dim))
        model.add(Reshape((self.gf_dim, self.gf_dim, self.gfc_dim)))
        model.add(create_conv_transp(self.gfc_dim // 2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(ReLU())

        model.add(create_conv_transp(self.gfc_dim // 4))
        model.add(BatchNormalization(momentum=0.8))
        model.add(ReLU())

        model.add(SelfAttention())

        model.add(create_conv_transp(self.gfc_dim // 8))
        model.add(BatchNormalization(momentum=0.8))
        model.add(ReLU())

        model.add(create_conv_transp(self.gfc_dim // 16))
        model.add(BatchNormalization(momentum=0.8))
        model.add(ReLU())

        model.add(Conv2D(self.channels, 3, strides=1,
                         dtype=self.dtype, padding='SAME', activation=None))
        model.add(Activation('tanh'))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        def create_conv(filters):
            return Conv2D(filters, self.gf_dim,
                          padding="SAME", activation=None, dtype=self.dtype,
                          use_bias=False, strides=2)

        model = Sequential()
        model.add(Conv2D(self.dfc_dim, self.df_dim,
                         padding="SAME", activation=None, dtype=self.dtype,
                         use_bias=False, strides=2, input_shape=self.img_shape))
        model.add(create_conv(self.dfc_dim))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=self.alpha))

        model.add(create_conv(self.dfc_dim * 2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=self.alpha))

        model.add(SelfAttention())

        model.add(create_conv(self.dfc_dim * 4))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=self.alpha))

        model.add(create_conv(self.dfc_dim * 8))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=self.alpha))

        model.add(Flatten())
        model.add(Dense(units=1, dtype=self.dtype, activation=None))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        start_time = datetime.datetime.now()

        tensorboard = TensorBoard(batch_size=batch_size, write_grads=True)
        tensorboard.set_model(self.combined)

        def named_logs(model, logs):
            result = {}
            for l in zip(model.metrics_names, logs):
                result[l[0]] = l[1]
            return result

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            imgs = self.data_loader.load_data(batch_size)

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            elapsed_time = datetime.datetime.now() - start_time

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] time: %s" % (
                epoch, d_loss[0], 100 * d_loss[1], g_loss, elapsed_time))

            tensorboard.on_epoch_end(epoch, named_logs(self.combined, [g_loss]))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
                self.combined.save_weights(f"saved_model/{self.dataset_name}/{epoch}.h5")

        self.save_imgs(epochs - 1)
        self.combined.save_weights(f"saved_model/{self.dataset_name}/{epoch}.h5")

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
                if self.dataset_name == 'mnist':
                    axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                else:
                    axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(f"images/{self.dataset_name}/{epoch}.png", dpi=200)
        plt.close()


if __name__ == '__main__':
    sagan = SAGAN(image_shape=(64, 64, 3),
                  dataset_name="img_align_celeba", dataset_path="G:\data\GAN")
    sagan.train(epochs=4000, batch_size=64, save_interval=100)
