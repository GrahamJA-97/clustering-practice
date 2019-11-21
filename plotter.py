from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np


def main():

    plot = plt.subplot()
    # entropies for k = 2-6
    points = [4545746.292099379, 2370689.686782968, 1341379.5700647032,
              1010098.1834947375, 684050.7480527544]

    plot.margins(2, 2)
    plot.set_xlim(0, 6)
    plot.set_ylim(600000, 4700000)
    plot.plot(points)
    plot.set_xlabel('K value')
    plot.set_ylabel('Inertia')
    plot.set_title('Inertia by number of clusters')
    plt.xticks(np.arange(5), ('2', '3', '4', '5', '6'))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
