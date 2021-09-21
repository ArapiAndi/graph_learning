
from src.PPI import PPI




if __name__ == '__main__':
    ppi_obj = PPI()
    D,labels,PCA_D = ppi_obj.initialize()
    ppi_obj.svm_no_maninfold(labels,PCA_D,D)
    n_neighbors = 15
    n_components = 2
    ppi_obj.svm_maninfoldIsomap(labels,D,n_neighbors,n_components)
    n_neighbors = 4
    ppi_obj.svm_maninfoldIsomapBestComponent(labels,D,PCA_D,n_neighbors)