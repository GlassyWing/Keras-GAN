from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.misc
from keras.datasets import mnist


class DataLoader:

    def __init__(self, dataset_name='mnist', img_res=(64, 64)):
        self.dataset_name = dataset_name
        self.img_res = img_res
        if self.dataset_name == 'mnist':
            self.dataset = mnist.load_data()
        else:
            self.dataset = None

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
        return None
