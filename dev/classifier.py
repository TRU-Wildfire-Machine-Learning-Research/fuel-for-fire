import os
import sys
import math
import copy
import datetime
import numpy as np
from misc import *
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import rasterio
from rasterio import plot
from rasterio.plot import show
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.base import clone
from sklearn.pipeline import Pipeline
'''
partial setup instructions for ubuntu 18:
  python3 -m pip install pandas
  python3 -m pip install rasterio
  python3 -m pip install --upgrade pytest
'''

global data_frame
data_frame = None  # global variable for data frame (need for parallelism)
# assume data dimensions are the same for every input file
lines, samples = None, None # read these from header file

def get_data(fp):
    """Assumes the filepath provided contains both
        .hdr and .bin files within the fp provided in the function
        call.

        returns a list of file paths to .bin files
    """
    rasterBin = []
    for f in fp:
        if not exist(f):
            err("couldn't find path:\n\t" + f)

        for root, dirs, files in os.walk(f, topdown=False):
            for file in files:
                if ".hdr" not in file:
                    rasterBin.append(os.path.join(f, file))
    return rasterBin


def get_date_string():
    day = datetime.datetime.now().day
    hour = datetime.datetime.now().hour
    min = datetime.datetime.now().minute

    return str(day) + "_" + str(hour) + "_" + str(min) + "/"


def populate_data_frame(rasterBin, showplots=False):
    global samples, lines
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
    data_frame = pd.DataFrame()
    '''columns=(
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
    '''
    print("putting stuff in data_Frame:")
    for raster in rasterBin:
        print("raster", raster)
        name = raster.split(os.path.sep)[-1].split(".")[0]

        if 'bool' in name:
            err('unexpected: bool substr of filename')

        hdr = '.'.join(raster.split(".")[:-1]) + '.hdr'
        if not exist(hdr):
            err("couldn't find hdr file")

        # pragmatic programming: make sure all the dimensions match
        samples2, lines2, bands2 = read_hdr(hdr)
        if not samples or not lines:
            samples, lines = samples2, lines2
        else:
            if samples2 != samples or lines2 != lines:
                err("unexpected dimensions: " +
                    str(samples2) + "x" + str(lines2) + ', expected: ' +
                    str(samples) + 'x' + str(lines))


        if "S2A" in name:
            dataset = rasterio.open(raster)
            for idx in dataset.indexes:
                print(name + "_" + str(idx))
                data_frame[name + "_" + str(idx)] = dataset.read(idx).ravel()

        elif "L8" in name:
            dataset = rasterio.open(raster)
            for idx in dataset.indexes:
                print(name + "_" + str(idx))
                data_frame[name + "_" + str(idx)] = dataset.read(idx).ravel()

        elif "WATERSP" in name:
            water = rasterio.open(raster).read(1)
            print('water_val')
            data_frame['water_val'] = water.ravel()
            print('water_bool')
            data_frame['water_bool'] = data_frame['water_val'] != 128
            
        elif "RiversSP" in name:
            river = rasterio.open(raster).read(1)
            print('river_val')
            data_frame['river_val'] = river.ravel()
            print('river_bool')
            data_frame['river_bool'] = data_frame['river_val'] == 1.0

        elif "BROADLEAF_SP" in name:
            broadleaf = rasterio.open(raster).read(1)
            print('broadleaf_val')
            data_frame['broadleaf_val'] = broadleaf.ravel()
            print('broadleaf_bool')
            data_frame['broadleaf_bool'] = data_frame['broadleaf_val'] == 1.0

        elif "SHRUB_SP" in name:
            shrub = rasterio.open(raster).read(1)
            print('shrub_val')
            data_frame['shrub_val'] = shrub.ravel()
            print('shrub_bool')
            data_frame['shrub_bool'] = data_frame['shrub_val'] != 0.0

        elif "MIXED_SP" in name:
            mixed = rasterio.open(raster).read(1)
            print('mixed_val')
            data_frame['mixed_val'] = mixed.ravel()
            print('mixed_bool')
            data_frame['mixed_bool'] = data_frame['mixed_val'] != 0.0

        elif "CONIFER_SP" in name:
            conifer = rasterio.open(raster).read(1)
            print('conifer_val')
            data_frame['conifer_val'] = conifer.ravel()
            print('conifer_bool')
            data_frame['conifer_bool'] = data_frame['conifer_val'] != 0.0

        elif "HERB_GRAS_SP" in name:
            herb = rasterio.open(raster).read(1)
            print('herb_val')
            data_frame['herb_val'] = herb.ravel()
            print('herb_bool')
            data_frame['herb_bool'] = data_frame['herb_val'] != 0.0

        elif "CCUTBL_SP" in name:
            clearcut = rasterio.open(raster).read(1)
            print('clearcut_val')
            data_frame['clearcut_val'] = clearcut.ravel()
            print('clearcut_bool')
            data_frame['clearcut_bool'] = data_frame['clearcut_val'] != 0.0

        elif "EXPOSED_SP" in name:
            exposed = rasterio.open(raster).read(1)
            print('exposed_val')
            data_frame['exposed_val'] = exposed.ravel()
            print('exposed_bool')
            data_frame['exposed_bool'] = data_frame['exposed_val'] != 0.0

        else:
            layer = rasterio.open(raster)
            if len(layer.indexes) > 1:
                err("expected one band only")
            layer = rasterio.open(raster).read(1)

            # pragmatic programming: make sure "bool" isn't in filename already
            if 'bool' in name:
                err('unexpected "bool" in filename')

            # pragmatic programming: make sure no name collision
            if name in data_frame:
                err(name + " already in data_frame")

            d = layer.ravel()
            counts = hist(d)

            enough_points = min(counts.values()) > 1000

            # only add the layer to the stack, if it has at least two values
            if len(counts) > 1 and enough_points:

                print(len(counts), name, counts)
                data_frame[name] = d
            
                print(name + '_bool')
                data_frame[name + '_bool'] = data_frame[name] != 0.
    
    if showplots:
        show_original_image(data_frame, 'l')
        show_original_image(data_frame, 's')
        show_truth_data_subplot(data_frame)

    print("data_frame.columns", data_frame.columns)
    return data_frame


def create_image_array(df, class_):
    """Generates a numpy.ndarray the same
        size as the raw data (statically defined)
        that is used to visualize a binary representation
        of truth data.

        returns a ndarray.

        Comment:
          in future, might find it simpler to use numpy array 
        for everything (i.e., init one numpy array for each 
        input file). That way, don't need methods to convert
        between data types!
    """
    arr = np.ones([len(df)], dtype='int')
    class_bool = class_ + "_bool"
    true_df = df[class_bool].loc[df[class_bool] is True]
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
        print("Specify an image to show in show_original_image(df, __) " +
              "('s' for sentinel2 or 'l' for landsat8)")

    arr[:, :, 0] = red
    arr[:, :, 1] = green
    arr[:, :, 2] = blue

    plt.gcf().set_size_inches(7, 7. * float(lines) / float(samples))
    plt.imshow(arr)
    plt.tight_layout()
    impath = "original_image" + "_" + data + ".png"
    print("+w " + impath)
    plt.savefig(impath)


def concatenate_dataframes(df1, df2):

    os_list = [df1, df2]

    X_full = pd.concat(os_list)
    return X_full


def get_x_data(df, image_type):
    """Returns a slice of the original dataframe corresponding
        to the training data
    """
    if image_type == "all":
        X = df.loc[:, :'L8_11']  # :"L8_longwave_infrared2"]
    elif image_type == "s":
        X = df.loc[:, :'S2A_12']  #:"S2_swir2"]
    elif image_type == "l":
        X = df.loc[:, 'L8_1': 'L8_11']  # "L8_coastal_aerosol": "L8_longwave_infrared2"]
    return X


def get_training_set(data, X_true, X_false, class_, image_type='all'):
    """Concatenates true pixels and false pixels into a single dataframe to
        create a dataset.

        Returns X, a pandas dataframe, and y, a pandas series
    """
    X_full = concatenate_dataframes(X_true, X_false)

    X = get_x_data(X_full, image_type)

    y = X_full[class_]
    return X, y


def normalize_data(X):
    X_norm = StandardScaler(copy=True, with_mean=True,
                            with_std=True).fit_transform(X)

    return X_norm


def oversample(smallerClass, largerClass):
    """A naive approach to oversampling data.

        Returns pandas dataframe.

    """
    oversample = smallerClass.copy()

    while len(oversample) < len(largerClass):
        oversample = concatenate_dataframes(oversample, smallerClass)

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
        data_frame[class_str] = \
            data_frame[dictionary[class_]] | data_frame[class_str]

    dictionary = create_class_dictionary(data_frame)

    return data_frame, class_str, dictionary


def get_intersection_indices(df, col1, col2):
    """Does an AND operation on col1 and col2,
        printing the result. This represents
        the number of pixels that exist in class1
        and class2
    """
    col1b = col1 + "_bool"
    col2b = col2 + "_bool"
    indices = df.loc[df[col1b] & df[col2b]].index.values
    print("Intersection of", col1, "and", col2, len(indices))
    return indices


def remove_intersections(df, class_):
    cd = create_class_dictionary(df)
    print(len(df))
    for c in cd.keys():
        if c == class_:
            continue
        else:
            list_of_indices = get_intersection_indices(df, class_, c)
            df = df.drop(list_of_indices, axis=0)
    print("Length after removing intersections: ", len(df))
    return df


def true_sample_is_smaller(t, f):
    return len(t) < len(f)


def get_sample(data_frame, classes, image_type='all', undersample=True,
               normalize=True):
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

    X_true = data_frame[data_frame[class_] is True]
    X_false = data_frame[data_frame[class_] is False]

    if undersample:

        if true_sample_is_smaller(X_true, X_false):
            # true is smaller, so take a subset of the false data
            X_false = data_frame[data_frame[class_]
                                 is False].sample(len(X_true))

        else:
            # false is smaller, so take a subset of the true data
            X_true = data_frame[data_frame[class_]
                                is True].sample(len(X_false))

        X, y = get_training_set(
            data_frame, X_true, X_false, class_, image_type)

        if normalize:
            return normalize_data(X), y
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
            return normalize_data(X), y
        else:
            return X, y


def print_classifier_metrics(y_test, y_pred):
    """

    """
    debug = False

    cm = confusion_matrix(y_test, y_pred)

    if debug:
        print("Confusion Matrix")
        print("[tn fp]")
        print("[fn tp]\n")
        for arr in cm:
            print(arr)
    cm_p = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    if debug:
        print("")
        for arr in cm:
            print(arr)
    
    return cm, cm_p


def plot_confusion_matrix_image(df, clf, true_val, image_type='all'):
    """UNDER DEVELOPMENT

    """

    # grab the test data (includes data the system was trained on)
    raw_data = get_x_data(df, image_type)

    try:
        y_pred = clf.predict(raw_data)  # predict on all the data
    except Exception:
        print("Error: It's likely that you are trying to plot using an image" +
              "type that isn't the same as the image type in the classifier")
        print("Ensure that your image_type argument in get_sample() equals " +
              "the image_type argument in plot_confusion_matrix_image()")
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
    t = datetime.datetime.now().time()
    impath = true_val + str(t) + '.png'
    print('+w ' + impath)
    plt.savefig(impath)


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
    debug = False
    """sklearn SGDClassifier

    """
    sgd_clf = SGDClassifier(random_state=42, verbose=False,
                            max_iter=1000, tol=1.e-3)

    np.set_printoptions(precision=3)

    if debug:
        print("\n\nBegin Fitting SGD\n")

    sgd_clf = sgd_clf.fit(X_train, y_train)
    y_pred = sgd_clf.predict(X_test)

    if debug:
        print("\n{:*^30}\n".format("Training complete"))
    score = "{:.3f}".format(sgd_clf.score(X_test, y_test))
   
    if debug:
       print("Test score:", score)

    cm, cm_p = print_classifier_metrics(y_test, y_pred)
    return sgd_clf, score, cm, cm_p


def trainGB(X_train, X_test, y_train, y_test):
    gbrt = GradientBoostingClassifier(random_state=0)

    np.set_printoptions(precision=3)

    print("\n\nBegin Fitting GBC\n")

    gbrt = gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_test)

    score = "{:.3f}".format(gbrt.score(X_test, y_test))
    print("Test score:", score)
    cm = print_classifier_metrics(y_test, y_pred)

    return gbrt, score, cm


def train_all_variations(df):
    """Rudimentary test run of a SGD classifier with all of the possible
        iterations with the given functionality. Pass a data_frame, the
        script will save a png representation of each class with the highest
        performace on the raw data. Variations include: Training on Sentinel2
        data, Landsat8 data, or both at the same time; Normalizing the data or
        not normalizing; Undersampling or oversampling; and each of the 9
        classes.
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
                    clf, score, cm = train(X, y)

                    # Save the top scored plot
                    if max_score < float(score):
                        max_score = float(score)
                        plot_confusion_matrix_image(
                            df, clf, class_, image_type=i)


def calculate_mean_metrics(cm_list, n_folds, total_score, percentage=False):
    print("Calculating Mean Metrics")
    TN = 0.0
    FP = 0.0
    FN = 0.0
    TP = 0.0
    for conf_matrix in cm_list:
        TN = TN + conf_matrix[0, 0]
        FP = FP + conf_matrix[0, 1]
        FN = FN + conf_matrix[1, 0]
        TP = TP + conf_matrix[1, 1]
    if not percentage:
        n_folds = 1
    TNstr = "{:.3f}".format(TN/n_folds)
    FPstr = "{:.3f}".format(FP/n_folds)
    FNstr = "{:.3f}".format(FN/n_folds)
    TPstr = "{:.3f}".format(TP/n_folds)

    mean_score = "{:.3f}".format(total_score/n_folds)
    print("TN:", TNstr)
    print("FP:", FPstr)
    print("FN:", FNstr)
    print("TP:", TPstr)
    print("MeanScore:", mean_score)
    return TNstr, FPstr, FNstr, TPstr, mean_score


def build_folds(X_true, X_false, n_folds):
    # https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation
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

        # Delete the data from the original dataframes
        X_true.drop(index=X_tsample.index.values.tolist(), inplace=True)
        X_false.drop(index=X_fsample.index.values.tolist(), inplace=True)

        fold = concatenate_dataframes(X_tsample, X_fsample)
        fold.sample(frac=1)  # shuffle fold
        folds.append(fold)

    # Append the left over data from using the floor method
    fold = concatenate_dataframes(X_true, X_false)
    folds[0] = folds[0].append(fold)

    print("{:/^30}".format("Data Folded"))
    print("Number of Folds:", n_folds)
    print("Size of each fold:", str(fold_size))
    print("Leftover datapoints added to fold 0:", len(fold))

    return folds


def fold(data_frame, class_, n_folds=5, disjoint=False):
    """
    """
    if disjoint:
        data_frame = remove_intersections(data_frame, class_)

    cd = create_class_dictionary(data_frame)

    # Retrieve the original data. Does .copy() do anything after the slice?
    class_ = cd[class_]
    X_true = data_frame[data_frame[class_]].copy()
    X_false = data_frame[~ data_frame[class_]].copy()

    # Decide which class needs to take a subset
    if len(X_true) < len(X_false):
        X_false = X_false.sample(len(X_true))
    else:
        X_true = X_true.sample(len(X_false))

    return build_folds(X_true, X_false, n_folds)


def train_test_split_folded_data(fd, test_data_idx, class_, image_type="all",
                                 normalize=False):
    cd = create_class_dictionary(fd[0])

    train_is_empty = True

    for f in range(len(fd)):
        if f == test_data_idx:  # This is our test set
            X_test = get_x_data(fd[test_data_idx], image_type)
            y_test = fd[test_data_idx][cd[class_]]

        else:
            if train_is_empty:
                X_train = get_x_data(fd[test_data_idx], image_type)
                y_train = fd[test_data_idx][cd[class_]]
                train_is_empty = False
            else:
                X_train = concatenate_dataframes(get_x_data(fd[test_data_idx],
                                                            image_type),
                                                 X_train)
                y_train = concatenate_dataframes(fd[test_data_idx][cd[class_]],
                                                 y_train)
    if normalize:
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)

    return X_train, X_test, y_train, y_test


def train_folded(folded_data, class_, n_folds=5, image_type='all',
                 normalize=True):
    cm_list = list()
    cm_p_list = list()
    total_score = 0

    for idx in range(len(folded_data)):
        X_train, X_test, y_train, y_test = \
          train_test_split_folded_data(folded_data, idx, class_,
                                       image_type=image_type,
                                       normalize=normalize)
        clf, score, cm, cm_p = train(X_train, X_test, y_train, y_test)

        cm_list.append(cm)
        cm_p_list.append(cm_p)
        total_score = total_score + float(score)

    TN, FP, FN, TP, mean_score = calculate_mean_metrics(cm_list,
                                                        len(folded_data),
                                                        total_score)
    TN_p, FP_p, FN_p, TP_p, mean_score = \
        calculate_mean_metrics(cm_p_list, len(folded_data), total_score,
                               True)
    return [TN, FP, FN, TP, TN_p, FP_p, FN_p, TP_p, mean_score, clf]


# execute this in parallel
def train_variation(params):
    global data_frame

    # unpack input parameters
    class_, nf, d, n, i = params

    # selection
    folded_data = fold(data_frame, class_, n_folds=nf, disjoint=d)

    # training
    TN, FP, FN, TP, TN_p, FP_p, FN_p, TP_p, mean_score, clf = \
        train_folded(folded_data, class_,
                     n_folds=nf, image_type=i,
                     normalize=n)
    result = [TN, FP, FN, TP, \
            TN_p, FP_p, FN_p, TP_p, \
            mean_score, clf, params]
    if len(result) != 11:
        err("result")

    return(result)

def train_all_variations_folded(df, n_f=[2, 5, 10], disjoint=[True, False],
                                norm=[True], it=[all]):
    newpath = "log/folded_classifier_" + get_date_string()

    # recursively create folder if not exist yet
    w = newpath.split(os.path.sep)
    for i in range(1, len(w)):
        p = (os.path.sep).join(w[0: i])
        if not os.path.exists(p):
            os.mkdir(p)
    cd = create_class_dictionary(df)
    f = open(newpath + "results.csv", "wb")
    f.write(("Class,N_Folds,Image_Type,Disjoint,Normalize," +
             "TN,FP,FN,TP,TN%,FP%,FN%,TP%,Mean_Accuracy").encode())

    runs = []   # input combinations
    for class_ in cd.keys():
        for nf in n_f:
            for d in disjoint:
                for n in norm:
                    for i in it:
                        runs.append([class_, nf, d, n, i]) # folded_data, class_, nf, i, n])

    # outputs, one for each input combo
    data = parfor(train_variation, runs)

    for result in data:
        print("result", result)
        [TN, FP, FN, TP, \
         TN_p, FP_p, FN_p, TP_p, \
         mean_score, clf, params] = result

        # unpack params
        class_, nf, d, n, i = params

        # one line of log output
        line_to_write = ('\n' + class_ + "," + str(nf) + "," +
                         i + "," + str(d) + "," + str(n) + "," + TN +
                         "," + FP + "," + FN + "," + TP + "," + TN_p +
                         "," + FP_p + "," + FN_p + "," + TP_p + "," +
                         mean_score)
        print(line_to_write)
        f.write(line_to_write.encode())
    return data

"""
        LIST OF THE AVAILABLE DATA NOT INVESTIGATED

    # vri_s2_objid2.tif_project_4x.bin_sub.bin
    # S2A.bin_4x.bin_sub.bin
    # vri_s3_objid2.tif_project_4x.bin_sub.bin
"""

if __name__ == "__main__":


    data_folder = "data_bcgw"

    if not os.path.exists(data_folder) or not os.path.isdir("data_bcgw"):
        err("please run from fuel-for-fire/ folder")
    
    data_frame = populate_data_frame(get_data(["data_img/",
                                               "data_bcgw/",
                                               "data_vri/binary/"]),
                                               showplots=False)
    
    '''
    data = train_all_variations_folded(data_frame,
                                       n_f=range(2, 21),
                                        disjoint=[True],
                                        norm=[True],
                                        it=['all'])
    '''
    data = train_all_variations_folded(data_frame,
                                       n_f=[2, 5, 10], #range(2, 21),
                                        disjoint=[False],
                                        norm=[True],
                                        it=['all'])
    
    sys.exit(1)
    for d in data:
        print("d", d)
        [TN, FP, FN, TP, TN_p, FP_p, FN_p, TP_p, mean_score, clf] = d
        print("clf", clf)
    
    # fd = fold(data_frame, 'water', n_folds=5, disjoint=False)
    # for idx in range(len(fd)):
    #     X_train, X_test, y_train, y_test = \
    #       train_test_split_folded_data(fd, idx, 'water',
    #                                    image_type='all',
    #                                    normalize=True)
    #     clf, score, cm, cm_p = train(X_train, X_test,
    #                                  y_train, y_test)
    #     plot_confusion_matrix_image(data_frame, clf, 'water',
    #                                 image_type='all')

    """     Single fold training
    
        X, y = get_sample(data_frame, "water", image_type='l'
                          undersample=False, normalize=True)
        X_train, X_test, y_train, y_test = \
                train_test_split(X, y, random_state=0, test_size=0.2)
        clf, score, cm = train(X_train, X_test, y_train, y_test)
    """

    """     Single fold training of union classes

        X, y = get_sample(data_frame, ["water", "river"],
                          image_type='all' undersample=False,
                          normalize=True)
        X_train, X_test, y_train, y_test = \
                train_test_split(X, y, random_state=0, test_size=0.2)
        clf, score, cm = train(X_train, X_test, y_train, y_test)
    """

    """     Folded data training

        folded_data = fold(data_frame, "water", n_folds=10)
        TN, FP, FN, TP, mean_score = train_folded(folded_data, class_,
                                                  n_folds=5,
                                                  image_type='all',
                                                  normalize=True)
    """

    """ Testing for folding and training all data with number of fold
            options built in

        train_all_variations(data_frame)
    """

    """     Other useful functions
        plot_confusion_matrix_image(data_frame, clf, "water", image_type="all")
        get_intersection_indices(data_frame, "water", "river")
        folded_data = fold(data_frame, "water", n_folds=10)
    """
