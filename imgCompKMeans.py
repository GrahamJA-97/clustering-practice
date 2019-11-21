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
    # inertia_vals = [61008498183.83514, 34454563582.18372, 32259669990.29129, 31837458649.92389, 31747068821.582188, 31727383981.42274,
    # 31723153271.2726, 31721853052.235714]

    inertia_vals = [6237645904.740321, 4656784194.011244, 3179784127.733261, 2859639517.0464554, 2750365416.003592, 2685101135.2215004, 2631723759.642526,
                    2586824062.7563076, 2546463914.9146276, 2516942592.143171, 2497728707.0214195, 2484586046.7247863, 2473827648.631936, 2465846829.9025483,
                    2459082149.8852477, 2453433287.427545, 2448377507.967031, 2443760740.144903, 2439743106.8966236, 2435766931.001046, 2432394864.2584357,
                    2429136305.4837923, 2426911316.54699, 2424454606.741913, 2422302772.765727, 2419931033.544321, 2418222809.5146747, 2416932070.0217223,
                    2415646371.6312213, 2414187989.9630694, 2413352105.451529, 2412746686.0088468, 2412038090.883278, 2411284053.6917505, 2410733618.2702923,
                    2410290255.373254, 2409751984.7080126, 2409343156.276633, 2409136853.043205, 2409003575.858629, 2408889233.329532, 2408813427.1639404,
                    2408755585.7190814]


    inertia_plot.margins(2, 2)
    inertia_plot.set_xlim(0, 9)
    inertia_plot.set_ylim(2350000000, 6300000000)
    # inertia_plot.axis(xlim=(0, 3), ylim=(3180000000, 5000000000))
    inertia_plot.plot(inertia_vals)
    #inertia_plot.xticks(np.arange(3), ('0', '1', '2'))
    inertia_plot.set_xlabel('Iteration number')
    inertia_plot.set_ylabel('Inertia')
    inertia_plot.set_title('Inertia When k = 15')
    # inertia_plot.legend(loc='lower right')
    inertia_plot.grid(True)
    plt.xticks(np.arange(10), ('0', '5', '10', '15', '20', '25', '30', '35', '40', '43'))

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

