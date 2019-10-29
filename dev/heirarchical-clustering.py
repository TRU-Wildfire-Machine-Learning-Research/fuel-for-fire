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
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.feature_extraction.image import grid_to_graph
import cv2

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()



'''
Function Description: Running heirarchical Clustering on the image

'''
def hierarchical_clustering(image, img, desired_clusters):

    print('Starting executing the heirarchical clustering')
    
      # set parameters for clustering
    n_clusters_desired = desired_clusters # need to experiment with this
    print('Going to run the heirarchical clustering')
    hierarchical_clustering = AgglomerativeClustering(n_clusters = n_clusters_desired, linkage='ward')

    # do the clustering
    hierarchical_clustering.fit(image)

    # extract cluster labels and reshape for plotting
    X_cluster = hierarchical_clustering.labels_
    X_cluster = X_cluster.reshape(img[:, :, 0].shape)

    plt.figure(figsize=(1, 1))
    #colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 0), (0, 1, 1), (0.1, 0.2, 0.5), (0.8, 0.1, 0.3)]
    # Create the colormap
    #cm = LinearSegmentedColormap.from_list("my map", colors, N=10)
    #plt.imshow(X_cluster) #, cmap=cm)
    #plt.colorbar()
    #plt.show()

    # returning the following values: number of clusters, lables, number of trees in the hierarchical tree
    #           number of connected components in the graph
    return hierarchical_clustering.n_clusters, hierarchical_clustering.labels_, hierarchical_clustering.n_leaves_, hierarchical_clustering.n_connected_components_



'''
Function Description: Running the DBSCAN
'''
def dbscan_clustering(image, img):
    
    dbscan_clustering = DBSCAN(eps=0.5, min_samples=10, algorithm= 'ball_tree')

    dbscan_clustering.fit(image)

    X_cluster = dbscan_clustering.labels_
    X_cluster = X_cluster.reshape(img[:, :, 0].shape)

    plt.figure(figsize=(20, 20))
    #colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 0), (0, 1, 1), (0.1, 0.2, 0.5), (0.8, 0.1, 0.3)]
    # Create the colormap
    #cm = LinearSegmentedColormap.from_list("my map", colors, N=10)
    plt.imshow(X_cluster) #, cmap=cm)
    plt.colorbar()
    plt.show()




'''
Function Description: Splitting the main image
'''
def image_splitter(image, number):
    
    print('Starting the image splitting')
    #showing the original image
    img=cv2.imread(image)
    cv2.imshow('output', img)
    #cv2.waitKey(0)

    #Getting the size of the image
    img_height = img.shape[0]
    img_width = img.shape[1]
    
    #Size per cropped image
    cropped_size = int(img.shape[0]/number) # as we have square images, this need to be changed according to the image dimensions

    # intitializing the array to store all the required information
    data = [];

    #crop starter
    cs = 0
    for i in range(number):
        #cropping down the image
        crop_img = img[cs:cs+cropped_size, cs:cs+cropped_size]

        #informative
        print('Number of rows in the cropped image: ',crop_img.shape[0])
        print('Numebr of columns in the cropped image: ', crop_img.shape[1])
        print('Number of bands in the cropped image: ', crop_img.shape[2])


        #initialize an empty numpy array of cropped image size
        np_crop_img =  np.zeros((crop_img.shape[0],  # number of rows
                        crop_img.shape[1],  # number of cols
                        crop_img.shape[2])  # number of bands
                        )


        #store the image data in the numpy array for reshaping
        for b in range(crop_img.shape[2]):
            np_crop_img[:,:,b] = crop_img[:,:,b]   

        #convert the image to single to numpy array
        new_crop_img_shape = (np_crop_img.shape[0]*np_crop_img.shape[1], np_crop_img.shape[2])
        crop_img_X = np_crop_img[:,:,:np_crop_img.shape[2]].reshape(new_crop_img_shape)
        print(crop_img_X.shape)

        #before running heirarchical clustering find the number of appropriate clusters in the cropped image
        ap_clusters_possible=elbow_method(crop_img_X, 7) #7 is the maximum number of clusters found in the whole image using elbow method

        #Runninng hierarchical clustering on the cropped image and  receiving the 
        (n_cluster, lables, n_leaves, n_connected_components )=hierarchical_clustering(crop_img_X, crop_img,ap_clusters_possible )

        #pushing the data to the data array for further interpretation
        data.append((n_cluster, lables, n_leaves, n_connected_components ))

        print('Number of clusters found: ', n_cluster)
        print('Labels: ', lables)
        print('Number of leaves: ', n_leaves)
        print('Number of connected components: ', n_connected_components )

        

        #increment the crop starter
        cs=cs+cropped_size

        #show the cropped image
        #cv2.imshow("its cropped",crop_img)
        #cv2.waitKey(0)
    

    #printing totla number of clustered cities we get
    print('Total number of different clustered cropped images we got: ', len(data))




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



'''
Function Description: Display an image
'''



'''
Function Description: Main Function
'''

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
        print('Band Type: ', img_ds.GetRasterBand(b+1).DataType)
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

    #Printing the shape of the image
    print(img.shape[0]*img.shape[1])
    print('Image Raster Count: ', img.shape[2])


    #Downsampling the image before reshaping
   #img = img[::2, ::2] + img[1::2, ::2] + img[::2, 1::2] + img[1::2, 1::2]

    # reshape image again to match expected format for scikit-learn
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    X = img[:, :, :img.shape[2]].reshape(new_shape)
    print (X.shape[0])

    #running the hierarchical clustering
    #hierarchical_clustering(X, img)

    #runnig the DBSCAN
    #dbscan_clustering(X, img)

    #running image splitting
    image_splitter(image, 14)

