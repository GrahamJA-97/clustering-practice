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

    inertia_vals = [4891975975.070419, 2788107380.114996, 2146564999.67942, 2010813843.790583, 1890970290.153531, 1829777507.6152644,
                    1807819204.8122683, 1794590676.5246887, 1784551009.9828196, 1777326406.0143735, 1770918013.7896895, 1765417225.5850847,
                    1761572208.7818835, 1757948183.8318393, 1755003242.338388, 1752327155.442484, 1750363100.7989051, 1748386366.2278452,
                    1746793255.8491147, 1745246228.2667713, 1744008313.046172, 1742604956.5377, 1741840129.599502, 1741171056.0459032,
                    1740420448.8413146, 1739717799.912622, 1739111141.6235745, 1738700894.3237705, 1738196189.9786024, 1737525126.1292083,
                    1736978015.677811, 1736661377.0339985, 1736229045.0339496, 1735779625.750715, 1735395495.0838647, 1735137815.567533,
                    1734942051.8391266, 1734752522.5044482, 1734518567.2496505, 1734205629.7079136, 1733931622.777034, 1733614914.9448001,
                    1733479415.4870076, 1733183567.7594562, 1732906676.9809673, 1732663608.431356, 1732492332.0245786, 1732349625.1590977,
                    1732214085.4121923, 1732045379.6766264, 1731842154.4108486, 1731674342.2153716, 1731552110.013666, 1731440972.553985,
                    1731377108.7270117, 1731326716.4446805, 1731268784.8027103, 1731183772.1239505, 1731075751.2901063, 1731003811.8378408,
                    1730946539.3830378, 1730896376.3341656, 1730848209.7971067]


    inertia_plot.margins(2, 2)
    inertia_plot.set_xlim(0, 12)
    inertia_plot.set_ylim(1700000000, 5000000000)
    # inertia_plot.axis(xlim=(0, 3), ylim=(3180000000, 5000000000))
    inertia_plot.plot(inertia_vals)
    #inertia_plot.xticks(np.arange(3), ('0', '1', '2'))
    inertia_plot.set_xlabel('Iteration number')
    inertia_plot.set_ylabel('Inertia')
    inertia_plot.set_title('Inertia When k = 20')
    # inertia_plot.legend(loc='lower right')
    inertia_plot.grid(True)
    plt.xticks(np.arange(13), ('0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '63'))

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

