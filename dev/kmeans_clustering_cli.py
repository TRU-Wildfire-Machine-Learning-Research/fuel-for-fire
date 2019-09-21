"""
    @author: franarama

"""


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import imageio
from sklearn.cluster import KMeans
import numpy as np
from osgeo import gdal, gdal_array
import sys
import math

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()


"""
Displays provided satellite image
"""


def show_image(filepath):
    image = imageio.imread(filepath)
    plt.imshow(image)
    plt.show()


"""
Runs the k-means algorithm on a provided image
"""


def run_kmeans(filepath, k):
    image = imageio.imread(filepath)
    x, y, z = image.shape
    image_2d = get_image_2d(filepath)

    kmeans_cluster = KMeans(random_state=0, n_clusters=k)
    kmeans_cluster.fit(image_2d)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_

    # plt.imshow(cluster_centers[cluster_labels].reshape(x, y, z).astype(np.uint8))
    # plt.show()

    return cluster_labels, cluster_centers


"""
Reshapes provided image to 2D
"""


def get_image_2d(filepath):
    image = imageio.imread(filepath)
    x, y, z = image.shape
    image_2d = image.reshape(x*y, z)
    return image_2d


"""
Plots number of clusters vs. within cluster sum of squares
(which we aim to minimize)
"""


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


if __name__ == "__main__":

    # parse command line arg
    try:
        image = sys.argv[1]

    # catch no file given
    except IndexError:
        print("Must provide a filename")
        sys.exit(0)

    # show_image(image)
    # number of clusters
    K = 10
    # labels, centroids = run_kmeans(image, K)

    # max number of clusters to plot with elbow method
    MAX_K = 20

    # image_2d = get_image_2d(image)
    # elbow_method(image_2d, (MAX_K + 1))

    # Read in raster image
    img_ds = gdal.Open(image, gdal.GA_ReadOnly)
    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
                   gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))

    for b in range(img.shape[2]):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    X = img[:, :, :13].reshape(new_shape)

    k_means = KMeans(n_clusters=K)
    k_means.fit(X)

    X_cluster = k_means.labels_
    X_cluster = X_cluster.reshape(img[:, :, 0].shape)

    plt.figure(figsize=(20, 20))
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 0),
              (0, 1, 1), (0.1, 0.2, 0.5), (0.8, 0.1, 0.3)]
    # Create the colormap
    cm = LinearSegmentedColormap.from_list(
        "my map", colors, N=10)
    plt.imshow(X_cluster, cmap=cm)
    print(type(X_cluster))
    print(type(cm))

    plt.colorbar()
    plt.show()
