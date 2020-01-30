import os
import numpy as np
import matplotlib.pyplot as plt
from Utils.Misc import *
from Utils.Data import Data

if __name__ == "__main__":

    data = Data("data", "data_img", "data_bcgw")

    data.Label['conifer'].showplot()
    # for label in data.Label.keys():
    #     yb = data.Label[label].spatial()
    #     yr = data.Label[label].spatial(binary=False)

    #     plt.imshow(yb, cmap='gray')
    #     plt.show()

    #     plt.imshow(yr)
    #     plt.show()