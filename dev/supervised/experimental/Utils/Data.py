import os
import math
import yaml
import struct
import numpy as np
from Utils.Misc import *
from Utils.Label import *
from Utils.Image import *


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
        self.__build_labels(lbl_bins)

    def __build_images(self, bins):
        for idx, bin in enumerate(bins):
            if 'S2' in bin:
                self.S2 = Image(bins[idx])
            elif 'L8' in bin:
                self.L8 = Image(bins[idx])
            else:
                err("Do not recognize file ", bin)

    def __build_labels(self, bins):
        self.Label = dict()
        cfg = load_config()
        classes = sorted(cfg['bcgw_labels'])

        for _, bin in enumerate(bins):
            # add a dict item to the Label dict --
            # labelname : Label
            self.Label.update({
                c:Label(c, bin, cfg)
                for c in classes if c in bin.lower()
            })

    def __build_binaries(self,path):
        try:
            for root, dirs, files in os.walk(path, topdown=False):
                bin_files = [
                    os.path.join(path, '%s' % file)
                    for file in files if '.hdr' not in file
                ]
            return bin_files
        except:
            err("Error building headers and binaries for %s" % path)

    def labels_onehot(self):
    """
        We are going to loop through each class label here
        = an array of shape, 164410, with each index equal
        to either 0 (False) or 1 (True).

        Loop through all the labels, and the pixels in that
        label. If the pixel is one, check if the pixel in
        labels has already been set, if it has, set it to
        10(pixel belonging to more than one class).
        If it hasn't, set it to the class_idx.

    """

        print(len(self.Label.keys()))

        labels = np.zeros((self.S2.samples * self.S2.lines))

        for class_idx, label in enumerate(self.Label.keys()):
            for pixel_idx, pixel in enumerate(self.Label[label].Binary):
                # if this pixel belongs to the class label
                if pixel == 1:
                    # if the pixel in labels has already been set,
                    # we have a conflict pixel, set it to 10
                    if labels[pixel_idx] == 0:
                        labels[pixel_idx] = class_idx
                    else:
                        # equal to some other class, set to 10
                        labels[pixel_idx] = 10
                elif pixel == 0:
                    continue
        return labels