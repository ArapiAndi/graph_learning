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

class PPI():


    def distance_matrix_Shortest_PPI(self):
        #load dataset
        dataset = sio.loadmat('rsc/PPI.mat')
        # get graphs
        graphs = dataset['G'][0]
        labels = dataset['labels'].ravel()
        
        sp_kernel = ShortestPathKernel()
        sp_graph = sp_kernel.compute_multi_shortest_paths(graphs[:]['am'])
        K = sp_kernel.threads_eval_similarities(sp_graph)
        D = pairwise_distances(K, metric='euclidean', n_jobs=8)
        plt.imshow(D, zorder=2, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        plt.style.use("ggplot")
        plt.show()
        return D, labels

    def initialize(self):
        D, labels = self.distance_matrix_Shortest_PPI()
        classify(D, labels)
        PCA_D = PCA(n_components = 2).fit_transform(D)
        plt.plot(np.cumsum(PCA().fit(D).explained_variance_ratio_))
        plt.show()
        np.cumsum(PCA().fit(D).explained_variance_ratio_)[:3]
        return D, labels, PCA_D
           


    
    def svm_no_maninfold(self,labels,PCA_D,D):
        clf = svm.SVC(kernel="linear", C=1.0)

        acidovorax = PCA_D[labels == 1]
        acidobacteria = PCA_D[labels == 2]

        clf = clf.fit(PCA_D, labels)
        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(np.min(PCA_D), np.max(PCA_D))
        yy = a * xx - (clf.intercept_[0]) / w[1]

        plt.figure(figsize=(10,5))


        ax_av = plt.scatter(acidovorax[:, 0], acidovorax[:, 1], color = "xkcd:red", marker = "^",label = "Acidovorax", s = 455, alpha = 0.65) 
        ax_ab = plt.scatter(acidobacteria[:, 0], acidobacteria[:, 1], color = "green", label = "Acidobacteria",  s = 250, alpha = 0.75)
        svm_line = plt.plot(xx, yy, color = "xkcd:sky blue", linestyle = "--", linewidth = 3.0)

        plt.axis('tight');
   
        plt.legend(prop={'size': 15})

        ax_av.set_facecolor('xkcd:salmon')
        ax_ab.set_facecolor('xkcd:pale green')

        plt.show()

        ###3D###
    
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')

        PCA_D = PCA(n_components = 3).fit_transform(D)

        acidovorax = PCA_D[labels == 1]
        acidobacteria = PCA_D[labels == 2]

        clf = clf.fit(PCA_D, labels)
        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(np.min(PCA_D), np.max(PCA_D))
        yy = a * xx - (clf.intercept_[0]) / w[1]

        #plt.figure(figsize=(10,5))


        ax_av = ax.scatter(acidovorax[:, 0], acidovorax[:, 1], acidovorax[:, 2],c = "xkcd:red", marker = "^",label = "Acidovorax", s = 455, alpha = 0.65) 
        ax_ab = ax.scatter(acidobacteria[:, 0], acidobacteria[:, 1], acidobacteria[:, 2], c = "green", label = "Acidobacteria",  s = 250, alpha = 0.75)
   

        plt.axis('tight');
        plt.legend(prop={'size': 15})

        ax_av.set_facecolor('xkcd:salmon')
        ax_ab.set_facecolor('xkcd:pale green')
        ax.view_init(azim = 30, elev = 25)
        plt.show()


    def svm_maninfoldIsomap(self,labels,D, n_neighbors, n_components):
        iso_prj_D = manifold.Isomap(n_neighbors, n_components).fit_transform(D)
        clf = SVC(kernel="linear", C = 1.0)
        scores_ln = cross_val_score(clf, iso_prj_D, labels, cv = 10, n_jobs= 8)
        np.mean(scores_ln)


        acidovorax = iso_prj_D[labels == 1]
        acidobacteria = iso_prj_D[labels == 2]

        clf = clf.fit(iso_prj_D, labels)
        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(np.min(iso_prj_D), np.max(iso_prj_D))
        yy = a * xx - (clf.intercept_[0]) / w[1]

        plt.figure(figsize=(10,5))


        ax_av = plt.scatter(acidovorax[:, 0], acidovorax[:, 1], color = "xkcd:red", marker = "^",label = "Acidovorax", s = 455, alpha = 0.65) 
        ax_ab = plt.scatter(acidobacteria[:, 0], acidobacteria[:, 1], color = "green", label = "Acidobacteria",  s = 250, alpha = 0.75)
        svm_line = plt.plot(xx, yy, color = "xkcd:sky blue", linestyle = "--", linewidth = 3.0)

        plt.axis('tight');
        plt.legend(prop={'size': 15})

        ax_av.set_facecolor('xkcd:salmon')
        ax_ab.set_facecolor('xkcd:pale green')

        plt.show()


    def svm_maninfoldIsomapBestComponent(self,labels,D,PCA_D,n_neighbors):
        opt_n_components = fit_n_components(D, labels, manifold.Isomap, n_neighbors)
        opt_iso_prj_D = manifold.Isomap(4, 18).fit_transform(D)
        PCA_D = PCA(n_components = 2).fit_transform(opt_iso_prj_D)
        clf = SVC(kernel="linear", C = 1.0)

        acidovorax = PCA_D[labels == 1]
        acidobacteria = PCA_D[labels == 2]

        clf = clf.fit(PCA_D, labels)
        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(np.min(PCA_D), np.max(PCA_D))
        yy = a * xx - (clf.intercept_[0]) / w[1]

        plt.figure(figsize=(10,5))


        ax_av = plt.scatter(acidovorax[:, 0], acidovorax[:, 1], color = "xkcd:red", marker = "^",label = "Acidovorax", s = 455, alpha = 0.65) 
        ax_ab = plt.scatter(acidobacteria[:, 0], acidobacteria[:, 1], color = "green", label = "Acidobacteria",  s = 250, alpha = 0.75)
        svm_line = plt.plot(xx, yy, color = "xkcd:sky blue", linestyle = "--", linewidth = 3.0)

        plt.axis('tight');
        plt.legend(prop={'size': 15})

        ax_av.set_facecolor('xkcd:salmon')
        ax_ab.set_facecolor('xkcd:pale green')

        plt.show()


