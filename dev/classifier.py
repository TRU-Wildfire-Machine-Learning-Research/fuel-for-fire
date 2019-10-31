import numpy as np
import matplotlib.pyplot as plt
import os
import rasterio
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

"""
This function taken from
https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
"""

"""
getData returns a list of the file names in the data folder.
"""


def getData(fp):
    # Specify the path to our data

    rasterBin = []
    for root, dirs, files in os.walk(fp, topdown=False):
        for file in files:
            if ".hdr" not in file:
                rasterBin.append(os.path.join(fp, file))

    print("files retrieved")
    return rasterBin


"""
Fill the data frame with data
"""


def populateDataFrame(rasterBin):
    data_frame = pd.DataFrame(columns=(
        'coastal_aerosol',
        'blue',
        'green',
        'red',
        'vre1',
        'vre2',
        'vre3',
        'nir',
        'narrow_nir',
        'water_vapour',
        'swir_cirrus',
        'swir2',
        'label_water_val',
        'label_water_bool',
        'label_river_val',
        'label_river_bool',
        'label_broadleaf_val',
        'label_broadleaf_bool',
        'label_shrub_val',
        'label_shrub_bool'))

    for raster in rasterBin:

        if "S2A.bin_4x.bin_sub.bin" in raster:

            dataset = rasterio.open(raster)
            for idx in dataset.indexes:
                """
                reads in the current band which is a mat of 401, 410 and ravels it
                storing the result in the current column
                """
                data_frame.iloc[:, idx-1] = dataset.read(idx).ravel()

        elif "WATERSP.tif_project_4x.bin_sub.bin" in raster:
            water = rasterio.open(raster).read(1)
            data_frame['label_water_val'] = water.ravel()
            data_frame['label_water_bool'] = data_frame['label_water_val'] != 128

        elif "RiversSP.tif_project_4x.bin_sub.bin" in raster:
            river = rasterio.open(raster).read(1)
            data_frame['label_river_val'] = river.ravel()
            data_frame['label_river_bool'] = data_frame['label_river_val'] == 1.0

        elif "BROADLEAF_SP.tif_project_4x.bin_sub.bin" in raster:
            broadleaf = rasterio.open(raster).read(1)
            data_frame['label_broadleaf_val'] = broadleaf.ravel()
            data_frame['label_broadleaf_bool'] = data_frame['label_broadleaf_val'] == 1.0

        elif "SHRUB_SP.tif_project_4x.bin_sub.bin" in raster:
            shrub = rasterio.open(raster).read(1)
            data_frame['label_shrub_val'] = shrub.ravel()
            # yet to decode the shrub values boolean correspondance

    return data_frame


def buildTrainingSet(X_true, X_false, class_):
    os_list = [X_true, X_false]

    X_full = pd.concat(os_list)

    X = X_full.loc[:, : 'swir2']  # only considers the columns up to swir2
    y = X_full[class_]

    return X, y


def normalizeData(X):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    X_norm = scaler.fit_transform(X)

    return X_norm


def oversample(X):
    X_true_oversample = X.copy()

    while len(X_true_oversample) < len(X_false):
        os_l = [X_true_oversample, X]
        X_true_oversample = pd.concat(os_l)
    return X_true_oversample


"""
Undersample the data (take a subset of false pixels
the same size as the true pixels)
"""


def getSample(data_frame, classes, undersample, normalize):

    # If we are combining classes
    class_dict = {"water": "label_water_bool",
                  "river": "label_river_bool",
                  "shrub": "label_shrub_bool",
                  "broadleaf": "label_broadleaf_bool"}
    if len(classes) > 1:
        print("OK")
        # TO DO
    else:
        class_ = class_dict[classes[0].lower()]

    print(class_dict)

    X_true = data_frame[data_frame[class_] == True]

    if undersample:

        # Get the same number of false samples as true samples
        # ie (balance the classes)
        X_false = data_frame[data_frame[class_]
                             == False].sample(len(X_true))

        X, y = buildTrainingSet(X_true, X_false, class_)

        # Normalize the data
        if normalize:
            X_norm = normalizeData(X)

            return X_norm, y
        else:
            return X, y
    else:
        X_false = data_frame[data_frame[class_]
                             == False]

        X_oversample = oversample(X_true)

        X, y = buildTrainingSet(X_oversample, X_false, class_)

        if normalize:
            X_norm = normalizeData(X)

            return X_norm, y
        else:
            return X, y


def outputClassifierMetrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    truenegative, falsepositive, falsenegative, truepositive = confusion_matrix(
        y_test, y_pred).ravel()

    for arr in cm:
        print(arr)
    print("Confusion Matrix\n", cm)
    print("\nTrue Negative", truenegative)  # land guessed correctly
    print("True Positive", truepositive)  # water guessed correctly
    print("False Negative", falsenegative)  # Land guessed as water
    print("False Positive", falsepositive)  # Water guessed as land
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print("\nConfusion Matrix Percentages\n", cm)


def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=0.2)

    sgd_classifier = SGDClassifier(random_state=42, verbose=1, warm_start=True)

    print("\n\nBegin Fitting\n")

    y_pred = sgd_classifier.fit(X_train, y_train).predict(X_test)

    print("\n{:*^30}\n".format("Training complete"))
    print("Test score: {:.3f}".format(sgd_classifier.score(X_test, y_test)))

    np.set_printoptions(precision=3)

    outputClassifierMetrics(y_test, y_pred)


"""
        LIST OF THE AVAILABLE DATA

    # vri_s2_objid2.tif_project_4x.bin_sub.bin
    # S2A.bin_4x.bin_sub.bin
    # BROADLEAF_SP.tif_project_4x.bin_sub.bin
    # WATERSP.tif_project_4x.bin_sub.bin
    # MIXED_SP.tif_project_4x.bin_sub.bin
    # SHRUB_SP.tif_project_4x.bin_sub.bin
    # vri_s3_objid2.tif_project_4x.bin_sub.bin
    # RiversSP.tif_project_4x.bin_sub.bin
    # L8.bin_4x.bin_sub.bin
    # CONIFER_SP.tif_project_4x.bin_sub.bin
    # HERB_GRAS_SP.tif_project_4x.bin_sub.bin
    # CCUTBL_SP.tif_project_4x.bin_sub.bin
    # EXPOSED_SP.tif_project_4x.bin_sub.bin
"""

if __name__ == "__main__":

    data_frame = populateDataFrame(getData("../data/"))
    # print(data_frame.label_shrub_val)

    X_us, y_us = getSample(
        data_frame, ['water'], undersample=True, normalize=True)
    # X_os, y_os = getSample(data_frame, undersample=False, normalize=True)

    train(X_us, y_us)
    # train(X_os, y_os)
