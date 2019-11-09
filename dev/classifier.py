import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import rasterio
from rasterio import plot
from rasterio.plot import show
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


def get_data(fp):
    """Assumes the filepath provided contains both
        .hdr and .bin files within the fp provided in the function
        call. 

        returns a list of file paths to .bin files
    """

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
                storing the result in the current column. ie(X values)
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
        show_original_image(data_frame)
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
        rs_arr = arr.reshape(401, 410)
    return rs_arr


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

    plt.tight_layout()
    plt.show()


def get_training_set(X_true, X_false, class_):
    """Concatenates true pixels and false pixels into a fingle data to
        create a dataset. 

        Returns X, a pandas dataframe, and y, a pandas series
    """
    os_list = [X_true, X_false]

    X_full = pd.concat(os_list)

    # only considers the columns up to swir2 (truth data's end col)
    X = X_full.loc[:, : 'swir2']
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


def true_sample_is_smaller(t, f):
    return len(t) < len(f)


def get_sample(data_frame, classes, undersample=True, normalize=True):
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

        X, y = get_training_set(X_true, X_false, class_)

        if normalize:
            return normalizeData(X), y
        else:
            return X, y

    else:  # oversampling

        if true_sample_is_smaller(X_true, X_false):
            X_true = oversample(X_true, X_false)

        else:
            X_false = oversample(X_false, X_true)

        X, y = get_training_set(X_true, X_false, class_)

        if normalize:
            return normalizeData(X), y
        else:
            return X, y


def print_classifier_metrics(y_test, y_pred):
    """

    """
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
    print("\nTrue Negative", truenegative)  # False class guessed correctly
    print("True Positive", truepositive)  # True class guessed correctly
    print("False Negative", falsenegative)  # False class guessed as true class
    print("False Positive", falsepositive)  # True class guessed as false class


def plot_confusion_matrix_image(df, clf, true_val):
    """UNDER DEVELOPMENT

    """

    # grab the test data (includes data the system was trained on)
    all_data = df.loc[:, : 'swir2']

    y_pred = clf.predict(all_data)  # predict on all the data
    y_true = df[true_val + "_bool"]  # store the true values

    arr = np.zeros([164410], dtype='int')

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
    arr = arr.reshape(401, 410)
    plt.xlabel(xlabel='width (px)')
    plt.ylabel(ylabel='height (px)')

    # legend_elements = [Patch(
    #     label='True Negative'), Patch(
    #     label='False Positive')]

    # Create the figure
    # fig, ax = plt.subplots()
    # ax.legend(handles=legend_elements)
    plt.imshow(arr)
    # colors = [im.cmap(im.value) for value in arr]

    # patches = [Patch(color=colors[i], label="Level {l}".format(l = arr[i])) for i in range(len(arr))]
    # put those patched as legend-handles into the legend
    # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()


def show_original_image(df):
    """builds an array of the RGB values of the
        truth image, scaling the original values
        between 0 - 1 (matplotlib req.) and
        displays that image in a plot.

    """
    blue = rescale(df.blue.values.reshape(401, 410))
    red = rescale(df.red.values.reshape(401, 410))
    green = rescale(df.green.values.reshape(401, 410))
    arr = np.zeros((401, 410, 3))
    arr[:, :, 0] = red
    arr[:, :, 1] = green
    arr[:, :, 2] = blue
    plt.imshow(arr)
    plt.show()


def rescale(arr):
    arr_min = arr.min()
    arr_max = arr.max()

    return (arr - arr_min) / (arr_max - arr_min)


def train(X, y):
    """sklearn SGDClassifier

    """
    sgd_clf = SGDClassifier(
        random_state=42, verbose=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=0.2)

    np.set_printoptions(precision=3)

    print("\n\nBegin Fitting SGD\n")

    sgd_clf = sgd_clf.fit(X_train, y_train)
    y_pred = sgd_clf.predict(X_test)

    print("\n{:*^30}\n".format("Training complete"))
    print("Test score: {:.3f}".format(sgd_clf.score(X_test, y_test)))
    print_classifier_metrics(y_test, y_pred)
    return sgd_clf


"""
        LIST OF THE AVAILABLE DATA NOT INVESTIGATED

    # vri_s2_objid2.tif_project_4x.bin_sub.bin
    # S2A.bin_4x.bin_sub.bin
    # vri_s3_objid2.tif_project_4x.bin_sub.bin
    # L8.bin_4x.bin_sub.bin
"""

if __name__ == "__main__":

    data_frame = populate_data_frame(get_data("../data/"), showplots=True)

    class_dictionary = create_class_dictionary(data_frame)

    X, y = get_sample(data_frame, ["shrub", "herb"],
                      undersample=False, normalize=True)

    clf = train(X, y)  # generate a classifier

    plot_confusion_matrix_image(data_frame, clf, "shrub_u_herb")
