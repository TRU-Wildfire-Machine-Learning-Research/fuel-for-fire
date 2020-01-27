import os
import struct
import numpy as np
from Utils.Misc import *

class DataTest(object):

    @staticmethod
    def load_mnist(path, kind='train'):
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)

        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
            images = ((images / 255) - .5) * 2

        return images, labels

    @staticmethod
    def mean_center_normalize(data):
        mean_vals = np.mean(data, axis=0)
        std_val = np.std(data)
        return mean_vals, std_val

class Data(object):
    def __init__(self, src, images_path, labels_path):
        self.src = src
        self.build(images_path, labels_path)

    def build(self, images_path, labels_path):
        images_path = os.path.join(self.src, '%s' % images_path)
        labels_path = os.path.join(self.src, '%s' % labels_path)
        img_bins, img_hdr = self.build_headers_binaries(images_path)
        lbl_bins, lbl_hdr = self.build_headers_binaries(labels_path)


    @staticmethod
    def build_headers_binaries(path):
        print(path)
        try:
            for root, dirs, files in os.walk(path, topdown=False):
                hdr_files = [
                    os.path.join(path, '%s' % file) \
                        for file in files if '.hdr' in file
                ]
                print("ok")
                bin_files = [
                    os.path.join(path, '%s' % file) \
                        for file in files if '.hdr' not in file
                        ]
                return bin_files, hdr_files
        except:
            err("Error building headers and binaries for %s" % path)

"""
    General use functions
"""
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

