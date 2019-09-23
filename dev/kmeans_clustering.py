import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
from sklearn.cluster import KMeans
import numpy as np
from osgeo import gdal, gdal_array
import sys
import math
import time

gdal.UseExceptions()
gdal.AllRegister()

rawImagePath = "../images/raw/"
colorMapImagePath = "../images/colormap"
imageName = "sentinel2"
imageExtension = ".bin"

image = rawImagePath + imageName + imageExtension

# Params
K = 10
MAX_K = 20
init = 'k-means++'
n_init = 10
# number of processors to use (default 1, -1 uses all processors)
n_jobs = 1


def showImage(filepath):
    image = cv2.imread(filepath)
    plt.show(image)
    plt.show()


def readRasterImage(image):
    print("Reading Raster Image")
    image_ds = gdal.Open(image, gdal.GA_ReadOnly)
    return image_ds


def getInputMatrix(image_ds):
    image = np.zeros((image_ds.RasterYSize, image_ds.RasterXSize, image_ds.RasterCount),
                     gdal_array.GDALTypeCodeToNumericTypeCode(image_ds.GetRasterBand(1).DataType))
    print("Creating input matrix")
    for band in range(image.shape[2]):
        image[:, :, band] = image_ds.GetRasterBand(band + 1).ReadAsArray()

    new_shape = (image.shape[0] * image.shape[1], image.shape[2])
    print("New Shape:", new_shape)
    X = image[:, :, :13].reshape(new_shape)

    return X, image


def runKMeans(K, X, image):
    start = time.time()
    print("Running K Means", "\nK =", K)
    k_means = KMeans(n_clusters=K, init=init, n_init=n_init, n_jobs=n_jobs)
    print("Fitting K Means")
    k_means.fit(X)
    print("Creating clusters")
    X_cluster = k_means.labels_
    X_cluster = X_cluster.reshape(image[:, :, 0].shape)
    print("Clusters created")
    stop = time.time()
    totalProcessTime = stop - start
    print("Time: " + str(totalProcessTime))
    return X_cluster


def createColorMap(X_cluster, K):
    plt.figure(figsize=(20, 20))
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 0),
              (0, 1, 1), (0.1, 0.2, 0.5), (0.8, 0.1, 0.3)]
    print("Creating Color Map")
    cm = LinearSegmentedColormap.from_list("Map", colors, N=K)
    plt.imshow(X_cluster, cmap=cm)
    plt.colorbar()
    plt.show
    time.sleep(5)
    print("Saving color map image")
    plt.imsave(colorMapImagePath + imageName +
               "_colorMap.png",  X_cluster, cmap=cm)


def elbow_method(image_2d, max_k):
    wcss = []
    for i in range(2, max_k):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        random_state=42)
        kmeans.fit(image_2d)
        wcss.append(kmeans.inertia_)

    x = [i for i in range(2, max_k)]
    plt.plot(x, wcss, '--bo')
    plt.xticks(x, x)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()


def run():
    img_ds = readRasterImage(image)
    X, img = getInputMatrix(img_ds)
    X_cluster = runKMeans(K, X, img)
    createColorMap(X_cluster, K)
    print("done")


run()
