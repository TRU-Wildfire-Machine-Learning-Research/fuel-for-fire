import os
import sys
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import rasterio
from rasterio import plot
from rasterio.plot import show
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
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

# these guys can be read off the input data variable:
lines = 401  # y dimension
samples = 410  # x dimension


def get_data(fp):
    """Assumes the filepath provided contains both
        .hdr and .bin files within the fp provided in the function
        call.

        returns a list of file paths to .bin files
    """

    if not os.path.exists(fp):
        print("Error: couldn't find path:\n\t" + fp)
        sys.exit(1)

    rasterBin = []
    for root, dirs, files in os.walk(fp, topdown=False):
        for file in files:
            if ".hdr" not in file:
                rasterBin.append(os.path.join(fp, file))

    print("files retrieved")
    return rasterBin


def populate_data_frame(rasterBin, showplots=False):
    """Receives a list of file paths to .bin files.
        A ground truth raster (predefined) is read,
        band by band, and stored in the data_frame.
        Additional 'labeled' truth data images are
        read and saved to the dataframe. These values
        are also decoded to binary interpretations, and
        the binary interpretation is stored in its own
        column.

        Showplots argument added to visualize the raw data
        and the truth data.


        Returns a populated pandas dataframe.
    """
    data_frame = pd.DataFrame(columns=(
        'S2_coastal_aerosol',
        'S2_blue',
        'S2_green',
        'S2_red',
        'S2_vre1',
        'S2_vre2',
        'S2_vre3',
        'S2_nir',
        'S2_narrow_nir',
        'S2_water_vapour',
        'S2_swir_cirrus',
        'S2_swir2',
        'L8_coastal_aerosol',
        'L8_blue',
        'L8_green',
        'L8_red',
        'L8_near_infrared',
        'L8_shortwave_infrared1',
        'L8_shortwave_infrared2',
        'L8_panchromatic',
        'L8_cirrus',
        'L8_longwave_infrared1',
        'L8_longwave_infrared2',
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
                data_frame.iloc[:, idx-1] = dataset.read(idx).ravel()

        elif "L8.bin_4x.bin_sub.bin" in raster:
            dataset = rasterio.open(raster)
            for idx in dataset.indexes:
                data_frame.iloc[:, idx+11] = dataset.read(idx).ravel()

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
            data_frame['conifer_bool'] = data_frame['conifer_val'] != 0.0

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
        show_original_image(data_frame, 'l')
        show_original_image(data_frame, 's')
        show_truth_data_subplot(data_frame)
    return data_frame


def create_image_array(df, class_):
    """Generates a numpy.ndarray the same
        size as the raw data (statically defined)
        that is used to visualize a binary representation
        of truth data.

        returns a ndarray.
    """
    arr = np.ones([len(df)], dtype='int')
    class_bool = class_ + "_bool"
    true_df = df[class_bool].loc[df[class_bool] == True]
    for idx in true_df.index:
        arr[idx] = 0
        rs_arr = arr.reshape(lines, samples)  # 401, 410)
    return rs_arr


def create_class_dictionary(data_frame):
    """dictionary for making simple requests for data from
        the dataframe. Key corresponds to a class name
        ie, water. The value associated with any key is
        simply the key with '_bool' appended to the end
        of the string.

        returns a dictionary

    """
    column_values = [
        col_name for col_name in data_frame.columns if "bool" in col_name]
    column_keys = []
    for key in column_values:
        column_keys.append(key.replace("_bool", ""))
    dictionary = dict(zip(column_keys, column_values))

    return dictionary


def show_truth_data_subplot(df, window_title="Truth Data"):
    """Displays a subplot of all the truth data in a binary
        fashion.

    """
    dictionary = create_class_dictionary(df)
    fig, axes = plt.subplots(3, 3, figsize=(9, 7))
    fig.canvas.set_window_title(window_title)
    col = 0
    row = 0
    for key in dictionary.keys():
        arr = create_image_array(df, key)
        show(arr, cmap='Greys', ax=axes[row, col], title=key)
        axes[row, col].set(xlabel='width (px)', ylabel='height (px)')

        # indexing the subplot
        col = col + 1
        if col > 2:
            col = 0
            row = row + 1

    # make this plot a bit bigger
    plt.gcf().set_size_inches(14, 14. * float(lines) / float(samples))
    plt.tight_layout()
    print("+w truth_data.png")
    plt.savefig("truth_data.png")  # show()


def show_original_image(df, data):
    """builds an array of the RGB values of the
        truth image, scaling the original values
        between 0 - 1 (matplotlib req.) and
        displays that image in a plot.

    """

    arr = np.zeros((lines, samples, 3))

    if data == 's':

        blue = rescale(df.S2_blue.values.reshape(lines, samples))
        red = rescale(df.S2_red.values.reshape(lines, samples))
        green = rescale(df.S2_green.values.reshape(lines, samples))

    elif data == 'l':

        blue = rescale(df.L8_blue.values.reshape(lines, samples))
        red = rescale(df.L8_red.values.reshape(lines, samples))
        green = rescale(df.L8_green.values.reshape(lines, samples))

    else:
        print("Specify an image to show in show_original_image(df, __) ('s' for sentinel2 or 'l' for landsat8)")

    arr[:, :, 0] = red
    arr[:, :, 1] = green
    arr[:, :, 2] = blue

    plt.gcf().set_size_inches(7, 7. * float(lines) / float(samples))
    plt.imshow(arr)
    plt.tight_layout()
    print("+w original_image.png")
    plt.savefig("original_image" + "_" + data + ".png")


def concatenate_dataframes(X_true, X_false):

    os_list = [X_true, X_false]

    X_full = pd.concat(os_list)
    return X_full


def get_training_set(data, X_true, X_false, class_, image_type='all'):
    """Concatenates true pixels and false pixels into a single dataframe to
        create a dataset.

        Returns X, a pandas dataframe, and y, a pandas series
    """
    X_full = concatenate_dataframes(X_true, X_false)

    # grab the raw data that we want to use (sentinel2, landsat8, or both)
    if image_type == 'all':
        X = X_full.loc[:, : 'L8_longwave_infrared2']

    elif image_type == 'l':
        X = X_full.loc[:, 'L8_coastal_aerosol': 'L8_longwave_infrared2']

    elif image_type == 's':
        X = X_full.loc[:, : 'S2_swir2']

    y = X_full[class_]
    return X, y


def normalizeData(X):
    X_norm = StandardScaler(copy=True, with_mean=True,
                            with_std=True).fit_transform(X)

    return X_norm


def oversample(smallerClass, largerClass):
    """A naive approach to oversampling data.

        Returns pandas dataframe.

    """
    oversample = smallerClass.copy()

    while len(oversample) < len(largerClass):
        os_l = [oversample, smallerClass]
        oversample = pd.concat(os_l)

    return oversample


def create_union_class_string(classes):
    """creates a string used for naming
        a new column (union of classes)in a dataframe.

        returns a string
    """
    class_str = ""
    for class_ in classes:
        class_str = class_str + "_u_" + class_

    return class_str[3:] + "_bool"


def create_union_column(data_frame, classes, dictionary):
    """generates a new column of data based on two or more
        columns of data. This augments the passed dataframe.

        returns a pandas dataframe with a new column, the string
        that corresponds to that columns name in the dataframe,
        and a class dictionary with the new class added to the
        dictionary.

    """

    class_str = create_union_class_string(classes)

    data_frame[class_str] = data_frame[dictionary[classes[0]]]
    for class_ in classes:
        data_frame[class_str] = data_frame[dictionary[class_]
                                           ] | data_frame[class_str]

    dictionary = create_class_dictionary(data_frame)

    return data_frame, class_str, dictionary


def check_intersection_column(df, col1, col2):
    """Does an AND operation on col1 and col2,
        printing the result. This represents
        the number of pixels that exist in class1
        and class2
    """
    col1b = col1 + "_bool"
    col2b = col2 + "_bool"
    res = df[col1b] & df[col2b]

    x = pd.value_counts(res)
    print(col1, "and", col2 + ": ", x[1])


def true_sample_is_smaller(t, f):
    return len(t) < len(f)


def get_sample(data_frame, classes, image_type='all', undersample=True, normalize=True):
    """retrieves a class balanced sample of data from the dataframe.
        That is, false class, balanced with true class, with options to
        normalize the feature data, or oversample.

        returns features, X and targets, y

    """

    class_dict = create_class_dictionary(data_frame)

    if isinstance(classes, (list)):
        data_frame, class_, class_dict = create_union_column(
            data_frame, classes, class_dict)
    else:
        class_ = class_dict[classes.lower()]

    X_true = data_frame[data_frame[class_] == True]
    X_false = data_frame[data_frame[class_]
                         == False]

    if undersample:

        if true_sample_is_smaller(X_true, X_false):
            # true is smaller, so take a subset of the false data
            X_false = data_frame[data_frame[class_]
                                 == False].sample(len(X_true))

        else:
            # false is smaller, so take a subset of the true data
            X_true = data_frame[data_frame[class_]
                                == True].sample(len(X_false))

        X, y = get_training_set(
            data_frame, X_true, X_false, class_, image_type)

        if normalize:
            return normalizeData(X), y
        else:
            return X, y

    else:  # oversampling

        if true_sample_is_smaller(X_true, X_false):
            X_true = oversample(X_true, X_false)

        else:
            X_false = oversample(X_false, X_true)

        X, y = get_training_set(
            data_frame, X_true, X_false, class_, image_type)

        if normalize:
            return normalizeData(X), y
        else:
            return X, y


def print_classifier_metrics(y_test, y_pred):
    """

    """
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix")
    print("[tn fp]")
    print("[fn tp]\n")
    for arr in cm:
        print(arr)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("")
    for arr in cm:
        print(arr)
    return cm


def plot_confusion_matrix_image(df, clf, true_val, image_type='all'):
    """UNDER DEVELOPMENT

    """

    # grab the test data (includes data the system was trained on)
    raw_data = get_x_data(df, image_type)

    try:
        y_pred = clf.predict(raw_data)  # predict on all the data
    except:
        print("Error: It's likely that you are trying to plot using an image type that isn't the same as the image type in the classifier")
        print("Ensure that your image_type argument in get_sample() is the same as your image_type argument in plot_confusion_matrix_image()")
        sys.exit()

    y_true = df[true_val + "_bool"]  # store the true values

    arr = np.zeros([lines * samples], dtype='int')

    for x in range(len(y_pred)):  # iterate the length of the arrays
        if y_true[x]:
            if y_pred[x]:
                arr[x] = 0
                # this is true positive
            else:
                arr[x] = 5
                # This is false positive
        else:
            if y_pred[x]:
                arr[x] = 10
                # this is false negative
            else:
                arr[x] = 15
                # this is true negative
    arr = arr.reshape(lines, samples)  # 401, 410
    plt.xlabel('width (px)')
    plt.ylabel('height (px)')

    plt.clf()
    plt.gcf().set_size_inches(7, 7. * float(lines) / float(samples))
    plt.imshow(arr)
    plt.tight_layout()

    print('+w ' + true_val + '.png')
    plt.savefig(true_val + '.png')


def rescale(arr, two_percent=True):
    """
    
    """
    arr_min = arr.min()
    arr_max = arr.max()
    scaled = (arr - arr_min) / (arr_max - arr_min)

    if two_percent:
        # 2%-linear stretch transformation for hi-contrast vis
        values = copy.deepcopy(scaled)
        values = values.reshape(np.prod(values.shape))
        values = values.tolist()
        values.sort()
        npx = len(values)  # number of pixels
        if values[-1] < values[0]:
            print('error: failed to sort')
            sys.exit(1)
        v_min = values[int(math.floor(float(npx)*0.02))]
        v_max = values[int(math.floor(float(npx)*0.98))]
        scaled -= v_min
        rng = v_max - v_min
        if rng > 0.:
            scaled /= (v_max - v_min)

    return scaled


def train(X_train, X_test, y_train, y_test):
    """sklearn SGDClassifier

    """
    sgd_clf = SGDClassifier(
        random_state=42, verbose=False, max_iter=1000, tol=1.e-3)

    np.set_printoptions(precision=3)

    print("\n\nBegin Fitting SGD\n")

    sgd_clf = sgd_clf.fit(X_train, y_train)
    y_pred = sgd_clf.predict(X_test)

    print("\n{:*^30}\n".format("Training complete"))
    score = "{:.3f}".format(sgd_clf.score(X_test, y_test))
    print("Test score:", score)

    cm = print_classifier_metrics(y_test, y_pred)
    return sgd_clf, score, cm


def trainGB(X_train, X_test, y_train, y_test):
    gbrt = GradientBoostingClassifier(random_state=0)

    np.set_printoptions(precision=3)

    print("\n\nBegin Fitting GBC\n")

    gbrt = gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_test)

    score = "{:.3f}".format(gbrt.score(X_test, y_test))
    print("Test score:", score)
    print_classifier_metrics(y_test, y_pred)

    return gbrt, score


def fold(df, class_, n_folds=5, normalize=True):
    """
    """
    cd = create_class_dictionary(data_frame)

    # Retrieve the original data
    class_ = cd[class_]
    X_true = data_frame[data_frame[class_] == True].copy()
    X_false = data_frame[data_frame[class_]
                         == False].copy()

    # Decide which class needs to take a subset
    if len(X_true) < len(X_false):
        X_false = X_false.sample(len(X_true))
    else:
        X_true = X_true.sample(len(X_false))

    # size
    size = len(X_true) * 2
    fold_size = int(math.floor(size/n_folds))
    half_fold_size = int(math.floor(fold_size/2))

    # Shuffle the data
    X_true.sample(frac=1)
    X_false.sample(frac=1)

    folds = list()
    for f in range(n_folds):
        X_tsample = X_true.sample(half_fold_size)
        X_fsample = X_false.sample(half_fold_size)
        X_true.drop(index=X_tsample.index.values.tolist(), inplace=True)
        X_false.drop(index=X_fsample.index.values.tolist(), inplace=True)
        print("f sample:", len(X_fsample))
        print("t sample:", len(X_tsample))
        fold = concatenate_dataframes(X_tsample, X_fsample)
        fold.sample(frac=1) # shuffle fold
        folds.append(fold)

    # Append the left over data from using the floor method
    fold = concatenate_dataframes(X_true, X_false)
    folds[0] = folds[0].append(fold)

    return folds

def get_x_data(df, image_type):
    if image_type == "all":
        X = df.loc[: ,  : "L8_longwave_infrared2" ]
    elif image_type == "s":
        X = df.loc[: ,  : "S2_swir2" ]
    elif image_type == "l":
        X = df.loc[: ,  "L8_coastal_aerosol" : "L8_longwave_infrared2" ]

    return X

def train_test_split_folded_data(fd, test_data_idx, class_, image_type="all"):
    cd = create_class_dictionary(fd[0])

    train_is_empty = True

    for f in range(len(fd)):
        if f != test_data_idx:
            if train_is_empty:
                X_train = get_x_data(fd[test_data_idx], image_type)
                y_train = fd[test_data_idx][cd[class_]]
                train_is_empty = False
            else:
                X_train = concatenate_dataframes(get_x_data(fd[test_data_idx],image_type), X_train)
                y_train = concatenate_dataframes(fd[test_data_idx][cd[class_]], y_train)

        else:
            X_test = get_x_data(fd[test_data_idx], image_type)
            y_test = fd[test_data_idx][cd[class_]]

    return X_train, X_test, y_train, y_test

def train_all_variations(df):
    """Rudimentary test run of a SGD classifier with all of the possible
        iterations with the given functionality. Pass a data_frame, the
        script will save a png representation of each class with the highest
        performace on the raw data. Variations include: Training on Sentinel2
        data, Landsat8 data, or both at the same time; Normalizing the data or
        not normalizing; Undersampling or oversampling; and each of the 9 classes.

    """
    it = ['all', 'l', 's']
    us = [True, False]
    nm = [True, False]
    class_dictionary = create_class_dictionary(df)

    for class_ in class_dictionary.keys():  # for each class
        max_score = 0.0
        for s in us:  # for each type of sampling
            for n in nm:  # for normalizing or not
                for i in it:  # for each image type
                    X, y = get_sample(df, class_, image_type=i,
                                      undersample=s, normalize=n)
                    print("")
                    print("{:*^50}".format(""))
                    print("{:<40}".format("Class: " + class_))
                    print("{:<40}".format("Undersampling:" + str(s)))
                    print("{:<40}".format("Normalize: " + str(n)))
                    print("{:<40}".format("Image_Type: " + i))
                    clf, score = train(X, y)

                    # Save the top scored plot
                    if max_score < float(score):
                        max_score = float(score)
                        plot_confusion_matrix_image(
                            df, clf, class_, image_type=i)


"""
        LIST OF THE AVAILABLE DATA NOT INVESTIGATED

    # vri_s2_objid2.tif_project_4x.bin_sub.bin
    # S2A.bin_4x.bin_sub.bin
    # vri_s3_objid2.tif_project_4x.bin_sub.bin
"""

if __name__ == "__main__":

    data_frame = populate_data_frame(get_data("../data/"), showplots=False)
    folded_data = fold(data_frame, "water")

    total_score = 0.0
    for idx in range(len(folded_data)):
        print(folded_data[idx].value_counts())
        X_train, X_test, y_train, y_test = train_test_split_folded_data(folded_data, idx, "clearcut")
        print(y_test.value_counts())
        #clf, score = train(X_train, X_test, y_train, y_test)
        #total_score = total_score + float(score)

    #mean_score = "{:.3f}".format(total_score/len(folded_data))
    #print("Mean Average:", mean_score)

    # create_class_dictionary(data_frame)
    # X, y = get_sample(data_frame, "water", image_type='l' undersample=False, normalize=True)
    # X, y = get_sample(data_frame, ["water", "river"], image_type='all' undersample=False, normalize=True)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
    # check_intersection_column(X_full, "water", "river")
    # train_all_variations(data_frame)
