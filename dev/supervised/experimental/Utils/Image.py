import numpy as np
from Utils.Misc import *
from Utils.DataManip import *

class Image(object):
    def __init__(self, bin):
        samples, lines, bands, data = read_binary(bin)
        self.samples, self.lines, self.bands = \
            int(samples), int(lines), int(bands)
        self.Data = data

    def ravel(self):
        return ravel(self.lines, self.samples, self.bands, self.Data)
    def spatial(self):
        return spatial(self.lines, self.samples, self.bands, self.Data)

    def rgb(self):
        res = np.zeros((self.lines, self.samples, 3))
        red = rescale(self.spatial()[:,:,3])
        blue = rescale(self.spatial()[:,:,2])
        green = rescale(self.spatial()[:,:,1])
        res[:,:,0] = red
        res[:,:,1] = green
        res[:,:,2] = blue
        return res