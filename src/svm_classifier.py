from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
import numpy as np


def fit_n_components(D, Y, manifold_learning, n_neighbors=14, n_iteration=20):
    max_acc = 0.0
    max_idx = 1
    clf = svm.SVC(kernel="linear", C=1.0)
    for i in range(2, n_iteration):
        ml_prj_D = manifold_learning(n_neighbors, i).fit_transform(D)
        scores_ln = cross_val_score(clf, ml_prj_D, Y, cv=10, n_jobs=8)
        if np.mean(scores_ln) > max_acc:
            max_acc = np.mean(scores_ln)
            max_idx = i
    return max_idx


def classify(D, labels):
    start_k_fold = StratifiedKFold(n_splits=10, shuffle=True)
    clf = svm.SVC(kernel="linear", C=1.0)
    scores_ln = cross_val_score(clf, D, labels, cv=start_k_fold, n_jobs = 8)
