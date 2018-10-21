from concurrent.futures import ThreadPoolExecutor
from glob import glob

import numpy as np
import scipy.misc
from keras.datasets import mnist


class DataLoader:

    def __init__(self, dataset_path=".", dataset_name='mnist', img_res=(64, 64)):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.img_res = img_res
        if self.dataset_name == 'mnist':
            self.dataset = mnist.load_data()
        else:
            self.dataset = None

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)

    def make_image_thumbnail(self, img_path, is_testing=False):
        img = self.imread(img_path)
        img = scipy.misc.imresize(img, self.img_res)

        # If training => do random flip
        if not is_testing and np.random.random() < 0.5:
            img = np.fliplr(img)

        return img

    def load_data(self, batch_size=1, is_testing=False):
        if self.dataset_name == 'mnist':
            (X_train, X_test), (_, _) = self.dataset
            X = X_test if is_testing else X_train

            idxs = np.random.randint(0, X.shape[0], size=batch_size)
            imgs = X[idxs]

            return_imgs = []

            with ThreadPoolExecutor() as executor:
                for img in executor.map(lambda img: scipy.misc.imresize(img, self.img_res), imgs):
                    return_imgs.append(img)
            return np.expand_dims(np.array(return_imgs), axis=3) / 127.5 - 1
        elif self.dataset_name == 'img_align_celeba':
            path = glob(self.dataset_path + '/%s/*' % (self.dataset_name))

            batch_images = np.random.choice(path, size=batch_size)

            return_imgs = []
            with ThreadPoolExecutor() as executor:
                for thumbnail in executor.map(self.make_image_thumbnail, batch_images):
                    return_imgs.append(thumbnail)

            return np.array(return_imgs) / 127.5 - 1.

        return None
