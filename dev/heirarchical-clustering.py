"""
@author: gagandeepbajwa
    based on sattelite-clustering.py by @franama

"""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

if __name__ == "__main__":

    # parse command line arg
    try:
        image = sys.argv[1]

    # catch no file given
    except IndexError:
        print("Must provide a filename")
        sys.exit(0)

    #with the same image create multiple images
    


    # Read in raster image
    img_ds = gdal.Open(image, gdal.GA_ReadOnly)



    # allocate memory to reshape image
    img = np.zeros((img_ds.RasterYSize,  # number of rows
                    img_ds.RasterXSize,  # number of cols
                    img_ds.RasterCount),  # number of bands
                    gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType)) # data type code

    # reshape the image band by band
    for b in range(img.shape[2]):
        print("reading band", b + 1, "of", img.shape[2])
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

    #Printing the shape of the image
    print(img.shape[0]*img.shape[1])
    print('Image Raster Count: ', img.shape[2])


    #Downsampling the image before reshaping
   #img = img[::2, ::2] + img[1::2, ::2] + img[::2, 1::2] + img[1::2, 1::2]

    # reshape image again to match expected format for scikit-learn
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    X = img[:, :, :img.shape[2]].reshape(new_shape)
    print (X.shape)

    # set parameters for clustering
    n_clusters_desired = 7 # need to experiment with this
    print('Going to run the heirarchical clustering')
    hierarchical_clustering = AgglomerativeClustering(n_clusters = n_clusters_desired, linkage='ward')

    # do the clustering
    hierarchical_clustering.fit(X)

    # extract cluster labels and reshape for plotting
    X_cluster = hierarchical_clustering.labels_
    X_cluster = X_cluster.reshape(img[:, :, 0].shape)

    plt.figure(figsize=(20, 20))
    #colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 0), (0, 1, 1), (0.1, 0.2, 0.5), (0.8, 0.1, 0.3)]
    # Create the colormap
    #cm = LinearSegmentedColormap.from_list("my map", colors, N=10)
    plt.imshow(X_cluster) #, cmap=cm)
    plt.colorbar()
    plt.show()

"""
Run hierarchical clustering
"""
'''
def run_hierarchical_clustering(filepath, k):
    img = get_image_2d(filepath)
    img = img.astype(np.int32)
    img = img[::2, ::2] + img[1::2, ::2] + img[::2, 1::2] + img[1::2, 1::2]
    X = np.reshape(img, (-1, 1))
    # Define the structure A of the data. Pixels connected to their neighbors.
    #connectivity = grid_to_graph(*img.shape)
    # Compute clustering
    print("Compute structured hierarchical clustering...")
    st = time.time()
    n_clusters = k
    ward = AgglomerativeClustering(n_clusters=n_clusters,
            linkage='ward',
            #connectivity=connectivity
            ).fit(X)
    label = np.reshape(ward.labels_, img.shape)
    print("Elapsed time: ", time.time() - st)
    print("Number of pixels: ", label.size)
    print("Number of clusters: ", np.unique(label).size)
    ###############################################################################
    # Plot the results on an image
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap=plt.cm.gray)
    for l in range(n_clusters):
        plt.contour(label == l, contours=1,
                    colors=[plt.cm.get_cmap("Spectral")(l / float(n_clusters)), ])
    plt.xticks(())
    plt.yticks(())
    plt.show()
'''

""""
Reshapes provided image to 2D
"""
'''
def get_image_2d(filepath):
    image = imageio.imread(filepath)
    x, y, z = image.shape
    image_2d = image.reshape(x*y, z)
    return image_2d
'''
