import os
import struct
import numpy as np

class DataTest(object):

    @staticmethod
    def load_mnist(path, kind='train'):
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)


        with open(labels_path, 'rb') as lbpath:
            print("Opened labels path")
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)

        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
            images = np.fromfile(
                imgpath, dtype=np.uint8).reshape(len(labels), 784)
            images = ((images / 255) - .5) * 2

        return images, labels

    @staticmethod
    def mean_center_normalize(data):
        mean_vals = np.mean(data, axis=0)
        std_val = np.std(data)
        return mean_vals, std_val

