# A script that essentially compresses an image without any external dependencies
#Written by Milan Kumar Hazra
# The process is as follows
#Read data file 
# Random initiation of centroids
#Loop for finding nearest centroids to each point and depending upon the updated classification update the 
#centroid positions

import numpy as np
import matplotlib.pyplot as plt
orig_image = plt.imread("Files_image_compression/Files/Files/home/jovyan/work/images/bird_small.png")
plt.imshow(orig_image)
#plt.show()

print(orig_image.shape)
orig_image = orig_image/255
x_img=np.reshape(orig_image, (orig_image.shape[0]*orig_image.shape[1], 3))
print(x_img.shape)


#define a function that will be iteratively called for classification of data points
#define a function to compute centroid on updated points classified
def find_closest_centroid(X, centroids):
    m,n = X.shape
    K = centroids.shape[0]
    dist = np.zeros(n)
    sq_dis = np.zeros(K)
    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    ### START CODE HERE ###
    for i in range(m):
        #print(X[i])
        
        for j in range(K):
            sq_dis[j]=0
            #print(centroids[j][0])
            
            dist = X[i]-centroids[j]
            for k in range(n):
                sq_dis[j] += dist[k]**2
        idx[i]=np.argmin(sq_dis)
    return idx


def compute_centroids(X, idx, K):
    m,n = X.shape
    centroids = np.zeros((K,n))
    for i in range(K):
        count_number = 0
        for j in range(m):
            if idx[j] == i:
                centroids[i]=centroids[i]+X[j]
                count_number += 1
        centroids[i]= centroids[i]/count_number
    return centroids

#Run k-means function iteratively
def kmeans_iter(x, init_centroids, max_iterations= 100):
    
    k =init_centroids.shape[0]
    centroids = init_centroids
    
    for i in range(max_iterations):
        print("K-means iteration number=", i)
        idx = find_closest_centroid(x, centroids)
        prev_centroids = centroids
        centroids = compute_centroids(x,idx,k)
    return idx, centroids
def random_initialization(X,K):
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    init_centroids = X[randidx[:K]]
    return init_centroids

k=10
max_iterations=100
init_centroids = random_initialization(x_img,k)
idx, centroids=kmeans_iter(x_img,init_centroids,max_iterations)
x_recovered = centroids[idx, :]
x_recovered = np.reshape(x_recovered, (orig_image.shape[0],orig_image.shape[1],orig_image.shape[2]))
plt.imshow(x_recovered*255)
plt.show()




