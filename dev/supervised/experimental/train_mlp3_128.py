import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from Utils.Misc import *
from Utils.Data import Data
from Utils.DataTest import *
from Utils.Model import LayersMultiLayerPerceptron3_128
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    data = Data("data", "data_img", "data_bcgw")
    X = data.S2.ravel()
    y = data.labels_onehot()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    mean_vals, std_val = DataTest.mean_center_normalize(X_train)

    X_train_centered = (X_train - mean_vals) / std_val
    X_test_centered = (X_test - mean_vals) / std_val

    del X_train, X_test

    print(X_train_centered.shape, y_train.shape)
    print(X_test_centered.shape, y_test.shape)

    mlpmodel = LayersMultiLayerPerceptron3_128(X_test_centered.shape[1], 9, learning_rate=0.003333)

    sess = tf.Session(graph=mlpmodel.g)
    training_costs = LayersMultiLayerPerceptron3_128.train_mlp(sess,
                                                    mlpmodel,
                                                    X_train_centered,
                                                    y_train,
                                                    num_epochs=100,
                                                    batch_size=128)

    y_pred = LayersMultiLayerPerceptron3_128.predict_mlp(sess, mlpmodel, X_test_centered)

    print('Test Accuracy: %.2f%%' % (
        100*np.sum(y_pred == y_test) / y_test.shape[0]
    ))
    # plt.imshow(data.S2.rgb)
    # plt.show()



    """
    working, keep for now
    """
    #data.Label['conifer'].showplot()
    # for label in data.Label.keys():
    #     yb = data.Label[label].spatial()
    #     yr = data.Label[label].spatial(binary=False)

    #     plt.imshow(yb, cmap='gray')
    #     plt.show()

    #     plt.imshow(yr)
    #     plt.show()