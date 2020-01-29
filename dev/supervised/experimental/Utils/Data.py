import os
import math
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
            images = np.fromfile(
                imgpath, dtype=np.uint8).reshape(len(labels), 784)
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
        self.__build(images_path, labels_path)

    def __build(self, images_path, labels_path):
        images_path = os.path.join(self.src, '%s' % images_path)
        labels_path = os.path.join(self.src, '%s' % labels_path)
        img_bins = self.__build_binaries(images_path)
        lbl_bins = self.__build_binaries(labels_path)
        self.__build_images(img_bins)

    def __build_images(self, bins):
        for idx, bin in enumerate(bins):
            if 'S2' in bin:
                self.S2 = Image(bins[idx])
            elif 'L8' in bin:
                self.L8 = Image(bins[idx])
            else:
                err("Do not recognize file", bin)

    @staticmethod
    def __build_binaries(path):
        try:
            for root, dirs, files in os.walk(path, topdown=False):
                bin_files = [
                    os.path.join(path, '%s' % file)
                    for file in files if '.hdr' not in file
                ]
                return bin_files
        except:
            err("Error building headers and binaries for %s" % path)


class Image(object):
    def __init__(self, bin):
        samples, lines, bands, data = read_binary(bin)
        self.samples, self.lines, self.bands = \
            int(samples), int(lines), int(bands)

        # Unsure of proper shape here --
        self.Data = data
    def ravel(self):
        return self.Data.reshape(self.lines * self.samples, self.bands)
    def spatial(self):
        return self.Data.reshape(self.lines, self.samples, self.bands)
    def rgb(self):
        res = np.zeros((self.lines, self.samples, 3))
        red = rescale(self.spatial()[:,:,3])
        blue = rescale(self.spatial()[:,:,2])
        green = rescale(self.spatial()[:,:,1])
        res[:,:,0] = red
        res[:,:,1] = green
        res[:,:,2] = blue
        return res

"""
    General use functions
"""


def create_batch_generator(X, y, batch_size=128, shuffle=False):
    X_copy = np.array(X)
    y_copy = np.array(y)

    if shuffle:
        data = np.column_stack((X_copy, y_copy))
        np.random.shuffle(data)
        X_copy = data[:, :-1]
        y_copy = data[:, :-1].astype(int)

    for i in range(0, X.shape[0], batch_size):
        yield (X_copy[i: i+batch_size, :], y_copy[i: i+batch_size])
