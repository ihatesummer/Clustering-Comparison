# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:50:40 2018

@author: user1
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation

#plt.close('all')

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

# Fitting Affinity Propagation to the datapoints
af = AffinityPropagation().fit(X)
clusterCenterIndice = af.cluster_centers_indices_
labels = af.labels_
nClusters = len(clusterCenterIndice)

# Visualising the Clusters
plt.figure(1)
plt.clf()
for i in range(0, nClusters):
    class_members = labels == i
    plt.scatter(X[class_members, 0], X[class_members, 1], s = 10, label = 'Cluster %d' % i)
plt.scatter(X[clusterCenterIndice, 0], X[clusterCenterIndice, 1], s = 500, c = 'silver', alpha=0.5, label = 'Centroids')
plt.title('Affinity Propagation')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
