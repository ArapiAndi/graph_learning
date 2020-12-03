import numpy as np
import scipy.io as sio
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import manifold
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from src.weisfeiler_lehman import Weisfeiler_Lehman_Kernel
from src.svm_classifier import fit_n_components, classify

PATH = "rsc/"


def create_distance_matrix(file):
    data_set = sio.loadmat(file)
    graphs = data_set['G'][0]
    labels = data_set['labels'].ravel()

    wl_kernel = Weisfeiler_Lehman_Kernel()
    K = wl_kernel.eval_similarities(graphs[:]['am'], 2)
    D = pairwise_distances(K, metric='euclidean')

    plt.imshow(D, zorder=2, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.style.use("ggplot")
    plt.show()
    return D, labels


if __name__ == '__main__':
    is_ppi = True

    if is_ppi:
        file_name = "PPI.mat"
    else:
        file_name = "SHOCK.mat"

    D, labels = create_distance_matrix(PATH + file_name)
    classify(D, labels)
