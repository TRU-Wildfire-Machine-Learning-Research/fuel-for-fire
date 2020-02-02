import numpy as np
from Utils.Misc import *
from Utils.DataManip import *

class Image(object):
    def __init__(self, bin):
        ## Read in the samples, lines and bands, pop data
        samples, lines, bands, data = read_binary(bin)

        ## cast to int
        self.samples, self.lines, self.bands = \
            int(samples), int(lines), int(bands)
        print("Samples",self.samples)
        print("Lines",self.lines)
        print("Bands",self.bands)

        # each pixel, organized by band - 164410 by 12 for s2, by 11 for l8
        self.Data = data.reshape((self.samples * self.lines,self.bands))

        print("Data Shape", self.Data.shape)
        self.__build_rgb()

    def __build_rgb(self):
        arr = np.zeros((self.samples, self.lines, 3))
        print("rgb shape:", arr.shape)
        tmp = self.Data.reshape(self.samples, self.lines, self.bands)
        print("temp shape:", tmp.shape)
        arr[:,:,0] = rescale(tmp[:,:,3]) # red
        arr[:,:,1] = rescale(tmp[:,:,2]) # blue
        arr[:,:,2] = rescale(tmp[:,:,1]) # green
        self.rgb = arr

    def ravel(self):
        return ravel(self.lines, self.samples, self.bands, self.Data)
    def spatial(self):
        return spatial(self.lines, self.samples, self.bands, self.Data)

