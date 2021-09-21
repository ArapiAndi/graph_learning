
from src.PPI import PPI
from src.SHOCK import SHOCK



if __name__ == '__main__':

    ## PPI Dataset
    ppi_obj = PPI()
    D,labels,PCA_D = ppi_obj.initialize()  
    ppi_obj.svm_no_maninfold(labels,PCA_D,D)
    n_neighbors = 15
    n_components = 2
    ppi_obj.svm_maninfoldIsomap(labels,D,n_neighbors,n_components)
    n_neighbors = 4
    ppi_obj.svm_maninfoldIsomapBestComponent(labels,D,PCA_D,n_neighbors)
    
    '''
    ## SHOCK Dataset
    shock_obj = SHOCK()
    D,labels,PCA_D = shock_obj.initialize()
    shock_obj.svm_no_maninfold(labels,PCA_D)
    n_neighbors = 15
    n_components = 2
    shock_obj.svm_maninfoldIsomap(labels,D, n_neighbors, n_components)
    n_neighbors = 4
    shock_obj.svm_maninfoldIsomapBestComponent(labels,D,n_neighbors)
    '''
