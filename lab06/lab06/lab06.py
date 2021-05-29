import random as rd
import numpy as np
import matplotlib.pyplot as mpl
import math as m
import pandas as pd

rd.seed(1)


def distp(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def random_clusters(X, n):
    index = []
    clusters = []
    for i in range(n):
        rand = rd.randint(1, len(X))
        while (rand in index):
            rand = rd.randint(1, len(X))
        index.append(rand)
        clusters.append(X[rand])
    return clusters

def delete_nan(column1, column2):
    AUTOS = pd.read_csv("autos.csv")
    X = AUTOS[[column1, column2]]
    X.fillna(X.mean(), inplace = True)
    X = np.array(X)
    return X


def calculate_distance(clusters, X):
    distances = np.zeros([len(clusters), len(X)])
    for i, c in enumerate(clusters):
        for j, x in enumerate(X):
            distances[i][j] = distp(c, x)
    return distances



def assingn_points_to_clusters(distances):
   points_clusters = np.argmin(distances, axis = 0)
   return points_clusters


def new_centroids(points_clusters, clusters, X):
    n_centroid = []
    for i,c in enumerate(clusters):
        n_centroid.append(X[i == points_clusters].mean(axis = 0))
    return n_centroid



def k_means(X, max_number_iteration, cluster_count):
    distatnces = []
    c = []
    n = []
    clusters = random_clusters(X, cluster_count)
    neighbours = np.zeros(len(X))
    for i in range(max_number_iteration):
        distatnces = calculate_distance(clusters, X)
        newNeighbours = assingn_points_to_clusters(distatnces)
        n.append(newNeighbours)
        c.append(clusters)
        if (np.array_equal(neighbours, newNeighbours)):
            return c, n
        neighbours = newNeighbours
        clusters = new_centroids(neighbours, clusters, X)

X = delete_nan('stroke', 'bore')
cluster_count = 3
c, n = k_means(X, 200, cluster_count)

    
for iter in range(len(c)):
    mpl.figure(figsize=(9, 6))
    for i in range(cluster_count):
        neighbours = X[n[iter] == i]
        x, y = np.hsplit(neighbours , 2)
        mpl.scatter(x, y)
    centroids = np.array(c[iter])
    x, y = np.hsplit(centroids, 2)
    mpl.scatter(x, y, marker = 'x', c="#ff0000", s = 100)
    mpl.ylabel('stroke')
    mpl.xlabel('bore')

mpl.show()