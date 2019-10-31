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
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.pipeline import Pipeline


def getData(fp):
    # Specify the path to our data

    rasterBin = []
    for root, dirs, files in os.walk(fp, topdown=False):
        for file in files:
            if ".hdr" not in file:
                rasterBin.append(os.path.join(fp, file))

    print("files retrieved")
    return rasterBin


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
    X_norm = StandardScaler(copy=True, with_mean=True,
                            with_std=True).fit_transform(X)

    return X_norm


def oversample(X, X_false):
    X_true_oversample = X.copy()

    while len(X_true_oversample) < len(X_false):
        os_l = [X_true_oversample, X]
        X_true_oversample = pd.concat(os_l)
    return X_true_oversample


def getSample(data_frame, classes, undersample=True, normalize=True):

    # If we are combining classes
    class_dict = {"water": "label_water_bool",
                  "river": "label_river_bool",
                  "shrub": "label_shrub_bool",
                  "broadleaf": "label_broadleaf_bool"}
    if len(classes) > 1:
        print("Program does not support multiple class unions")
        # TO DO
    else:
        class_ = class_dict[classes[0].lower()]

    X_true = data_frame[data_frame[class_] == True]

    if undersample:
        X_false = data_frame[data_frame[class_]
                             == False].sample(len(X_true))

        X, y = buildTrainingSet(X_true, X_false, class_)

        if normalize:
            return normalizeData(X), y
        else:
            return X, y
    else:  # oversampling
        X_false = data_frame[data_frame[class_]
                             == False]

        X_oversample = oversample(X_true, X_false)

        X, y = buildTrainingSet(X_oversample, X_false, class_)

        if normalize:
            return normalizeData(X), y
        else:
            return X, y


def outputClassifierMetrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    truenegative, falsepositive, falsenegative, truepositive = confusion_matrix(
        y_test, y_pred).ravel()
    print("Confusion Matrix\n")
    for arr in cm:
        print(arr)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("")
    for arr in cm:
        print(arr)
    print("\nTrue Negative", truenegative)  # land guessed correctly
    print("True Positive", truepositive)  # water guessed correctly
    print("False Negative", falsenegative)  # Land guessed as water
    print("False Positive", falsepositive)  # Water guessed as land


def train(X, y):

    # skfolds = StratifiedKFold(n_splits=3, random_state=42)
    sgd_classifier = SGDClassifier(
        random_state=42, verbose=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=0.2)

    np.set_printoptions(precision=3)

    print("\n\nBegin Fitting SGD\n")

    y_pred_sgd = sgd_classifier.fit(X_train, y_train).predict(X_test)

    print("\n{:*^30}\n".format("Training complete"))
    print("Test score: {:.3f}".format(sgd_classifier.score(X_test, y_test)))
    outputClassifierMetrics(y_test, y_pred_sgd)


"""
    for train_index, test_index in skfolds.split(X_train, y_train):

        clone_clf = clone(sgd_classifier)

        X_train_folds = X_train[train_index]
        y_train_folds = (y_train[train_index])
        X_test_fold = X_train[test_index]
        y_test_fold = (y_train[test_index])
        print(X_test_fold)
        # clone_clf.fit(X_train_folds, y_train_folds)
        # y_pred = clone_clf.predict(X_test_fold)

        # n_correct = sum(y_pred == y_test_fold)
        # print(n_correct / len(y_pred))  # prints 0.9502, 0.96565 and 0.96495
"""


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

    print("\n\n{:-^50}".format("WATER"))
    X_us, y_us = getSample(
        data_frame, ['water'])
    X_os, y_os = getSample(
        data_frame, ['water'], undersample=False)
    train(X_us, y_us)
    train(X_os, y_os)
"""
    print("\n\n{:-^50}".format("RIVER"))
    X_us, y_us = getSample(
        data_frame, ['river'])
    X_os, y_os = getSample(
        data_frame, ['river'], undersample=False)
    train(X_us, y_us)
    train(X_os, y_os)
"""
print("\n\n{:-^50}".format("BROADLEAF"))
X_us, y_us = getSample(
    data_frame, ['broadleaf'])
X_os, y_os = getSample(
    data_frame, ['broadleaf'], undersample=False)
train(X_us, y_us)
train(X_os, y_os)
