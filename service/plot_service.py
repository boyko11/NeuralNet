from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

class PlotService:

    def __init__(self):
        pass

    @staticmethod
    def plot_line(x, y, x_label, y_label, title=''):

        plt.figure()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.grid()

        plt.plot(x, y)

        plt.show()
