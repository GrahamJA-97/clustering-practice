#    name: imgCompKMeans.py
#  author: molloykp (Nov 2019)
# purpose: K-Means compression on an image

import numpy as np
from PIL import Image

from matplotlib.image import imread
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler

import argparse


def parseArguments():
    parser = argparse.ArgumentParser(
        description='KMeans compression of images')

    parser.add_argument('--imageFileName', action='store',
                        dest='imageFileName', default="", required=True,
                        help='input image file')
    parser.add_argument('--k', action='store',
                        dest='k', default="", type=int, required=True,
                        help='number of clusters')

    parser.add_argument('--outputFileName', action='store',
                       dest='outputFileName', default="", required=True,
                       help='output imagefile name')

    return parser.parse_args()


def main():
    parms = parseArguments()
    k = parms.k
    print("\nK value = ", k)
    img = imread(parms.imageFileName)
    img_size = img.shape

    # Reshape it to be 2-dimension
    # in other words, its a 1d array of pixels with colors (RGB)

    X = img.reshape(img_size[0] * img_size[1], img_size[2])

    # Insert your code here to perform
    # -- KMeans clustering
    # -- replace colors in the image with their respective centroid
    print("\n\n")
    means = KMeans(init='random', n_init=1, max_iter=100, n_clusters=k, verbose='true')
    means.fit(X)
    print("num iterations = ", means.n_iter_)
    print("\n\n")

    # replaces colors with their centroid values.
    centroids = np.array([list(means.cluster_centers_[label]) for label in means.labels_])
    centroids = centroids.astype("uint8")

    # save modified image (code assumes new image in a variable
    # called X_compressed)
    # Reshape to have the same dimension as the original image

    X_compressed = np.reshape(centroids, (img_size[0], img_size[1], img_size[2]))

    fig, ax = plt.subplots(1, 1, figsize = (8, 8))

    Rss11 = Image.fromarray(X_compressed)
    Rss11.save(parms.outputFileName + ".jpg")

    ax.imshow(X_compressed)
    for ax in fig.axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(parms.outputFileName,dpi=400,bbox_inches='tight',pad_inches=0.05)

    # Code to create inertia plots
    inertia_plot = plt.subplot()

    # hard coded from terminal output (verbose=true_ until I can figure out how to get them as an attribute :'(
    inertia_vals = [61008498183.83514, 34454563582.18372, 32259669990.29129, 31837458649.92389, 31747068821.582188, 31727383981.42274,
                    31723153271.2726, 31721853052.235714]

    inertia_plot.margins(2, 2)
    inertia_plot.set_xlim(0, 7)
    inertia_plot.set_ylim(31000000000, 61100000000)
    # inertia_plot.axis(xlim=(0, 3), ylim=(3180000000, 5000000000))
    inertia_plot.plot(inertia_vals)
    #inertia_plot.xticks(np.arange(3), ('0', '1', '2'))
    inertia_plot.set_xlabel('Iteration number')
    inertia_plot.set_ylabel('Inertia')
    inertia_plot.set_title('Inertia When k = 2')
    # inertia_plot.legend(loc='lower right')
    inertia_plot.grid(True)
    plt.xticks(np.arange(8), ('0', '1', '2', '3', '4', '5', '6', '7'))

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

