#   name: wineCluster.py
#   authors: Chris Schulz and Jake Graham
#   purpose: calculates the mean silhouette coecient and entropy
#   of the k clusters

import numpy as np
from PIL import Image

from sklearn import datasets
from matplotlib.image import imread
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler

import argparse


def parseArguments():
    parser = argparse.ArgumentParser(
        description='KMeans compression of images')

    parser.add_argument('--k', action='store',
                        dest='k', default="", type=int, required=True,
                        help='number of clusters')


    return parser.parse_args()


def main():
    parms = parseArguments()
    k = parms.k
    print("Number of clusters: ", k)

    # load in the data wine set.
    wine_data = datasets.load_wine()
    wine_data = wine_data.data

    means = KMeans(init='random', n_init=1, max_iter=100, n_clusters=k, verbose='true')
    means.fit(wine_data)

    cluster_centers = means.cluster_centers_
    # print("Cluster centers shape : ", cluster_centers.shape)
    # print("Cluster centers print: ", cluster_centers)
    # centroids = np.array([list(means.cluster_centers_[label]) for label in means.labels_])

    # print("Centroids shape : ", centroids.shape)

if __name__ == '__main__':
    main()
