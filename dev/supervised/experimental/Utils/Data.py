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
        print(lbl_bins)
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
        classes = cfg['bcgw_labels']

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
