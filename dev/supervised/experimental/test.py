import os
import numpy as np
import matplotlib.pyplot as plt
from Utils.Misc import *
from Utils.Data import Data

if __name__ == "__main__":

    data = Data("data", "data_img", "data_bcgw")
<<<<<<< HEAD

    data.Label['conifer'].showplot()
=======
    plt.imshow(data.S2.rgb)
    plt.show()



    """
    working, keep for now
    """
    #data.Label['conifer'].showplot()
>>>>>>> 376351d262e8c6339e557797f18e673f8584f5e2
    # for label in data.Label.keys():
    #     yb = data.Label[label].spatial()
    #     yr = data.Label[label].spatial(binary=False)

    #     plt.imshow(yb, cmap='gray')
    #     plt.show()

    #     plt.imshow(yr)
    #     plt.show()