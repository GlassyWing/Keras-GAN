import numpy as np


class ImagePool:

    def __init__(self, pool_size=50):
        self.images = []
        self.pool_size = pool_size
        self.num_imgs = 0
        if pool_size < 0:
            raise ValueError(f"Illegal pool size ${pool_size} which should be >= 0")

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for img in images:
            img = np.expand_dims(img, axis=0)
            if self.num_imgs < self.pool_size:
                self.images.append(img)
                self.num_imgs += 1
                return_images.append(img)
            else:
                idx = np.random.randint(0, self.pool_size)
                return_images.append(self.images[idx])
                self.images[idx] = img
        return np.concatenate(return_images, axis=0)
