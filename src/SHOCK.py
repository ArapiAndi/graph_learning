import scipy.io as sio
from src.shortest_path import ShortestPathKernel
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from src.svm_classifier import fit_n_components, classify
from sklearn.decomposition import PCA
import numpy as np
from sklearn import manifold
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


class SHOCK():
    def distance_matrix_Shortest_SHOCK(self):
        #load dataset
        dataset = sio.loadmat('rsc/SHOCK.mat')
        # get graphs
        graphs = dataset['G'][0]
        labels = dataset['labels'].ravel()
        
        sp_kernel = ShortestPathKernel()
        sp_graph = sp_kernel.compute_multi_shortest_paths(graphs[:]['am'])
        K = sp_kernel.threads_eval_similarities(sp_graph)
        D = pairwise_distances(K, metric='euclidean')
        plt.imshow(D, zorder=2, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        plt.style.use("ggplot")
        plt.show()
        return D, labels

    def initialize(self):
        D, labels = self.distance_matrix_Shortest_SHOCK()
        classify(D, labels)
        PCA_D = PCA(n_components = 2).fit_transform(D)
        plt.plot(np.cumsum(PCA().fit(D).explained_variance_ratio_))
        plt.show()
        np.cumsum(PCA().fit(D).explained_variance_ratio_)[:3]
        return D, labels, PCA_D

    def svm_no_maninfold(self,labels,PCA_D):
        plt.figure(figsize=(8,6))
        n_classes = 10
        for i in range(n_classes):
            plt.scatter(PCA_D[labels == i,0], PCA_D[labels == i,1], s = 155, alpha = 0.65)
    
        plt.axis('tight');
        plt.style.use("ggplot")
        plt.show()

    def svm_maninfoldIsomap(self,labels,D, n_neighbors, n_components):
        iso_prj_D = manifold.Isomap(n_neighbors, n_components).fit_transform(D)

        plt.figure(figsize=(10,6))
        n_classes = 10
        for i in range(n_classes):
            plt.scatter(iso_prj_D[labels == i,0], iso_prj_D[labels == i,1], s = 155, alpha = 0.65)
    
        plt.axis('tight');
        plt.grid(True)
        plt.show()

    def svm_maninfoldIsomapBestComponent(self,labels,D,n_neighbors):
        opt_n_components = fit_n_components(D, labels, manifold.Isomap, n_neighbors)
        opt_iso_prj_D = manifold.Isomap(n_neighbors, opt_n_components).fit_transform(D)
        classify(D,labels)