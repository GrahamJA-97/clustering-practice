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

    inertia_vals = [7646846702.642784, 5925110571.685779, 4638475170.2409935, 3891677600.0655146, 3826421232.5139966, 3795275495.1904445, 3778710497.5260215,
                    3767789726.555915, 3759513220.0340676, 3751443591.7462764, 3745313279.6860867, 3739679868.815274, 3734865735.0996118, 3730428223.0800366,
                    3726567832.6716456, 3723357860.3446064, 3720187469.3059216, 3717430043.8355384, 3714948168.948071, 3713291429.0182266, 3711474522.0578766,
                    3709753186.030366, 3708398673.9275656, 3707151781.9107304, 3706059106.95169, 3705165254.785444, 3704103616.3516493, 3703427922.0262847,
                    3702769564.4635115, 3702142241.980936, 3701441052.1758285, 3700783336.2324862, 3700140253.7965393, 3699546121.6321173, 3699110975.631962,
                    3698801790.2586646, 3698326144.422715, 3697865481.052916, 3697441232.308713, 3697100259.2346163, 3696742012.4450684, 3696528700.0919504,
                    3696387867.179233, 3696257818.035525]

    inertia_plot.margins(2, 2)
    inertia_plot.set_xlim(0, 9)
    inertia_plot.set_ylim(3500000000, 7800000000)
    # inertia_plot.axis(xlim=(0, 3), ylim=(3180000000, 5000000000))
    inertia_plot.plot(inertia_vals)
    #inertia_plot.xticks(np.arange(3), ('0', '1', '2'))
    inertia_plot.set_xlabel('Iteration number')
    inertia_plot.set_ylabel('Inertia')
    inertia_plot.set_title('Inertia When k = 10')
    # inertia_plot.legend(loc='lower right')
    inertia_plot.grid(True)
    plt.xticks(np.arange(10), ('0', '5', '10', '15', '20', '25', '30', '35', '40', '45'))

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

