# Imports
import numpy as np
import random as r
import matplotlib.pyplot as plt
from scipy.spatial import distance

# We define a function kmeans using k to be the number of clusters and points to be a 2D array containing [x,y] values for the data we are interested in
# We also may input centroids, but if we choose not to, we generate k random points instead.
def kmeans(k, points, centroids = []):
    if centroids == []:
        centroids = [[r.random() for i in range(2)] for j in range(k)]

    # Set the variable cnv, meaning converged, to False
    cnv = False
    # Such that while converged is False, the while loop loops. 
    while not cnv:
        # Generate an empty 2d array of length k
        cluster = [[] for _ in range(k)]
        # For every data point in our set.
        for point in points:
            # We calculate the distance from each centroid.
            distcent = [distance.euclidean(point, centroid) for centroid in centroids]
            # We find the index of the smallest distance to be the centroid this point is clustered with
            assign = distcent.index(min(distcent))
            # We then add this point to the list in the 2D array with the same index as the closest centroid.
            cluster[assign].append(point)

        # We create a new empty list for the new centroids.
        new_centroids = []
        for i in range(k):
            # we define clust as each cluster as a numpy array 
            clust = np.array(cluster[i])
            # We then define x as all the x values of the points and y as all the y values of the points.
            x,y = clust[:, :1], clust[:, 1:2]
            # We add the mean of the x values and y values to the new_centroids list.
            new_centroids.append([np.mean(x),np.mean(y)])

        # We now set cnv to true
        cnv = True
        
        # We now have a for loop to compare each element of the new centroids to the old centroids and check whether they are equal
        for i in range(k):
            for j in range(2):
                converg = round(new_centroids[i][j],3) == round(centroids[i][j],3)
                if not converg:
                    # If they are not equal, we set cnv to false 
                    cnv = False
                    break 
        
        # We know set the new centroids to be our "normal" (used) centroids
        centroids = new_centroids

    # We plot each cluster here after the while loop stops, i.e. after the clusters are decided
    for i in range(k):
        clust = np.array(cluster[i])
        x,y = clust[:, :1], clust[:, 1:2]
        plt.scatter(x, y)
    plt.scatter([j[0] for j in centroids], [k[1] for k in centroids], edgecolors= "cyan")
    
    # Showing the graph
    plt.show()

# Potential mainline code to generate a random set of data, and split into 5 clusters.

# points = [[r.random() for i in range(2)] for j in range(100)]
# kmeans(3,points)