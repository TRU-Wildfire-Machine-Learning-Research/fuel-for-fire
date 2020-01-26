import os
import struct
import numpy as np

class Data(object):

    def load_mnist(path, kind='train'):
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

        print(images_path)
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)

        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
            images = ((images / 255) - .5) * 2

        return images, labels

    def mean_center_normalize(data):
        mean_vals = np.mean(data, axis=0)
        std_val = np.std(data)
        return mean_vals, std_val

    def create_batch_generator(X, y, batch_size=128, shuffle=False):
        X_copy = np.array(X)
        y_copy = np.array(y)

        if shuffle:
            data = np.column_stack((X_copy, y_copy))
            np.random.shuffle(data)
            X_copy = data[:,:-1]
            y_copy = data[:,:-1].astype(int)

        for i in range(0, X.shape[0],batch_size):
            yield (X_copy[i : i+batch_size, :], y_copy[i : i+batch_size])


    # def load_bcgw(path, kind='train'):
