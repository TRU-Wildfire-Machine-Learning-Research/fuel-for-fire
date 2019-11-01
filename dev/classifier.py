import numpy as np
import matplotlib.pyplot as plt
import os
import rasterio
from rasterio import plot
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
from rasterio.plot import show


def getData(fp):
    # Specify the path to our data

    rasterBin = []
    for root, dirs, files in os.walk(fp, topdown=False):
        for file in files:
            if ".hdr" not in file:
                rasterBin.append(os.path.join(fp, file))

    print("files retrieved")
    return rasterBin


def populateDataFrame(rasterBin, showplots=False):
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
        'water_val',
        'water_bool',
        'river_val',
        'river_bool',
        'broadleaf_val',
        'broadleaf_bool',
        'shrub_val',
        'shrub_bool',
        'mixed_val',
        'mixed_bool',
        'conifer_val',
        'conifer_bool',
        'herb_val',
        'herb_bool',
        'clearcut_val',
        'clearcut_bool',
        'exposed_val',
        'exposed_bool'))

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
            data_frame['water_val'] = water.ravel()
            data_frame['water_bool'] = data_frame['water_val'] != 128

        elif "RiversSP.tif_project_4x.bin_sub.bin" in raster:
            river = rasterio.open(raster).read(1)
            data_frame['river_val'] = river.ravel()
            data_frame['river_bool'] = data_frame['river_val'] == 1.0

        elif "BROADLEAF_SP.tif_project_4x.bin_sub.bin" in raster:
            broadleaf = rasterio.open(raster).read(1)
            data_frame['broadleaf_val'] = broadleaf.ravel()
            data_frame['broadleaf_bool'] = data_frame['broadleaf_val'] == 1.0

        elif "SHRUB_SP.tif_project_4x.bin_sub.bin" in raster:
            shrub = rasterio.open(raster).read(1)
            data_frame['shrub_val'] = shrub.ravel()
            data_frame['shrub_bool'] = data_frame['shrub_val'] != 0.0

        elif "MIXED_SP.tif_project_4x.bin_sub.bin" in raster:
            mixed = rasterio.open(raster).read(1)
            data_frame['mixed_val'] = mixed.ravel()
            data_frame['mixed_bool'] = data_frame['mixed_val'] != 0.0

        elif "CONIFER_SP.tif_project_4x.bin_sub.bin" in raster:
            conifer = rasterio.open(raster).read(1)
            data_frame['conifer_val'] = conifer.ravel()
            # Bit funky here on the vis side of things

        elif "HERB_GRAS_SP.tif_project_4x.bin_sub.bin" in raster:
            herb = rasterio.open(raster).read(1)
            data_frame['herb_val'] = herb.ravel()
            data_frame['herb_bool'] = data_frame['herb_val'] != 0.0

        elif "CCUTBL_SP.tif_project_4x.bin_sub.bin" in raster:
            clearcut = rasterio.open(raster).read(1)
            data_frame['clearcut_val'] = clearcut.ravel()
            data_frame['clearcut_bool'] = data_frame['clearcut_val'] != 0.0

        elif "EXPOSED_SP.tif_project_4x.bin_sub.bin" in raster:
            exposed = rasterio.open(raster).read(1)
            data_frame['exposed_val'] = exposed.ravel()
            data_frame['exposed_bool'] = data_frame['exposed_val'] != 0.0

    if showplots:
        fig, axes = plt.subplots(3, 3, figsize=(9, 7))
        show(broadleaf, ax=axes[0, 0], title="broadleaf")
        show(water, ax=axes[0, 1], title="water")
        show(mixed, ax=axes[0, 2], title="mixed")
        show(shrub, ax=axes[1, 0], title="shrub")
        show(river, ax=axes[1, 1], title="river")
        show(conifer, ax=axes[1, 2], title="conifer")
        show(herb, ax=axes[2, 0], title="herb")
        show(clearcut, ax=axes[2, 1], title="clearcut")
        show(exposed, ax=axes[2, 2], title="exposed")

        plt.tight_layout()
        plt.show()
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


def makeClassDictionary(data_frame):
    column_values = [
        col_name for col_name in data_frame.columns if "bool" in col_name]
    column_keys = []
    for key in column_values:
        column_keys.append(key.replace("_bool", ""))
    dictionary = dict(zip(column_keys, column_values))

    return dictionary


def generateClassString(classes):
    class_str = ""
    for class_ in classes:
        class_str = class_str + "_u_" + class_

    return class_str[3:] + "_bool"


def generateUnionColumn(data_frame, classes, dictionary):

    class_str = generateClassString(classes)

    data_frame[class_str] = data_frame[dictionary[classes[0]]]
    for class_ in classes:
        data_frame[class_str] = data_frame[dictionary[class_]
                                           ] | data_frame[class_str]

    dictionary = makeClassDictionary(data_frame)

    return data_frame, class_str, dictionary


def getSample(data_frame, classes, undersample=True, normalize=True):

    # If we are combining classes
    class_dict = makeClassDictionary(data_frame)
    if len(classes) > 1:
        data_frame, class_, class_dict = generateUnionColumn(
            data_frame, classes, class_dict)
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
        # print(n_correct / len(y_pred))
"""


"""
        LIST OF THE AVAILABLE DATA NOT INVESTIGATED

    # vri_s2_objid2.tif_project_4x.bin_sub.bin
    # S2A.bin_4x.bin_sub.bin
    # vri_s3_objid2.tif_project_4x.bin_sub.bin
    # L8.bin_4x.bin_sub.bin
"""

if __name__ == "__main__":

    data_frame = populateDataFrame(getData("../data/"), showplots=False)
    X_us, y_us = getSample(
        data_frame, ['water', 'river'])
    print(data_frame[data_frame['water_u_river_bool'] == True])

"""
    print("\n\n{:-^50}".format("WATER"))
    X_us, y_us = getSample(
        data_frame, ['water'])
    X_os, y_os = getSample(
        data_frame, ['water'], undersample=False)
    train(X_us, y_us)
    train(X_os, y_os)

    print("\n\n{:-^50}".format("RIVER"))
    X_us, y_us = getSample(
        data_frame, ['river'])
    X_os, y_os = getSample(
        data_frame, ['river'], undersample=False)
    train(X_us, y_us)
    train(X_os, y_os)

    print("\n\n{:-^50}".format("BROADLEAF"))
    X_us, y_us = getSample(
        data_frame, ['broadleaf'])
    X_os, y_os = getSample(
        data_frame, ['broadleaf'], undersample=False)
    train(X_us, y_us)
    train(X_os, y_os)
"""
