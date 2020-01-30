import os
import numpy as np
import matplotlib.pyplot as plt
from Utils.Misc import *
from Utils.Data import Data

if __name__ == "__main__":

    data = Data("data", "data_img", "data_bcgw")


    print(data.Label['water'].Binary.shape)
    # for label in data.Label.keys():
    #     print(label)
    #     #u_elements, count_elements = np.unique(data.Label[label].Binary, return_counts=True)

    #     print("Frequency of unique values of %s array:" % label)
    #     print(np.asarray((u_elements, count_elements)))
