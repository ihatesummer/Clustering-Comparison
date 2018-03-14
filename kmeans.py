# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 13:57:21 2018

@author: user1
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

"""
# Generating the datapoints (2D)
nDatapoints = 100 #const - number of datapoints
X = np.random.random((nDatapoints,2))
"""

# Importing premade datapoints (for comparison)
X = np.load('100RandomDatapoints.npy')
nDatapoints = len(X)

"""
# Plotting the unclustered datapoints
plt.figure(0)
plt.clf()
plt.scatter(X[:,0], X[:,1], s = 10, c = 'black')
plt.title('Randomly Generated Datapoints')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
"""

"""
# Using the elbow method to find the optimal number of clusters
wcss = []
nMaxClusters = nDatapoints #const - max number of clusters

for i in range(1, nMaxClusters+1):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.figure(1)
plt.clf()
plt.plot(range(1, nMaxClusters+1), wcss)
plt.title('Elbow Method Plot: WCSS for 1~%d clusters' % nMaxClusters)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
"""
nClusters = 8 #const - according to the result of the elbow graph

# Fitting K-Means to the datapoints
kmeans = KMeans(n_clusters = nClusters, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.figure(2)
plt.clf()
for i in range(0, nClusters):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s = 10, label = 'Cluster %d' % i)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 500, c = 'silver', alpha=0.5, label = 'Centroids')
plt.title('K-means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
