import numpy as np
from Utils.Misc import *
from Utils.DataManip import *

class Image(object):
    def __init__(self, bin):
<<<<<<< HEAD
        samples, lines, bands, data = read_binary(bin)
        self.samples, self.lines, self.bands = \
            int(samples), int(lines), int(bands)
        self.Data = data
=======
        ## Read in the samples, lines and bands, pop data
        samples, lines, bands, data = read_binary(bin)

        ## cast to int
        self.samples, self.lines, self.bands = \
            int(samples), int(lines), int(bands)
        print("Samples",self.samples)
        print("Lines",self.lines)
        print("Bands",self.bands)

        # each pixel, organized by band - 164410 by 12 for s2, by 11 for l8
        self.Data = data.reshape((self.bands, self.lines * self.samples))

        print("Data Shape", self.Data.shape)
        self.__build_rgb()

    def __build_rgb(self):
        arr = np.zeros((self.lines, self.samples, 3))
        print("rgb shape:", arr.shape)
        #print("temp shape:", tmp.shape)
        
        for i in range(0, 3):
            arr[:, :, i] = self.Data[3 - i, :].reshape((self.lines, self.samples))
    
        for i in range(0, 3):
            arr[:, :, i] = rescale(arr[:, :, i])
        
        self.rgb = arr
>>>>>>> 376351d262e8c6339e557797f18e673f8584f5e2

    def ravel(self):
        return ravel(self.lines, self.samples, self.bands, self.Data)
    def spatial(self):
        return spatial(self.lines, self.samples, self.bands, self.Data)

<<<<<<< HEAD
    def rgb(self):
        res = np.zeros((self.lines, self.samples, 3))
        red = rescale(self.spatial()[:,:,3])
        blue = rescale(self.spatial()[:,:,2])
        green = rescale(self.spatial()[:,:,1])
        res[:,:,0] = red
        res[:,:,1] = green
        res[:,:,2] = blue
        return res
=======
>>>>>>> 376351d262e8c6339e557797f18e673f8584f5e2
