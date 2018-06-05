import os
import glob
import tqdm

import numpy as np

from sklearn.externals import joblib
from sklearn.mixture import GaussianMixture

def evaluate_gmms_opt(vec, n_gaussian, covariance = None, step = 3, filename=None):
    if covariance is None:
        covariance = ['spherical', 'tied', 'diag', 'full']
        
    best_option = {}
    best_bic = 1e9
    iter_wo_opt = 0
    for cov in covariance:
        for i in range(1, n_gaussian):
            gmm = GaussianMixture(n_components = i, covariance_type=cov)
            gmm = gmm.fit(vec)
            current_bic = gmm.bic(vec)
            if current_bic < best_bic:
                best_bic = current_bic
                best_option = {"n_components" : i, "covariance_type":cov}
                iter_wo_opt = 0
            else:
                iter_wo_opt += 1

            if iter_wo_opt == step:
                break
            if i == n_gaussian-1:
                print(filename)
    return best_option

for mfcc in tqdm.tqdm(glob.glob("F:/Nicolas/DNUPycharmProjects/machine_learning/audio/FMA/preprocessed_audio/cqt/*.npy")):
    filename = os.path.basename(mfcc)[:-4]
    path = os.path.join("F:/Nicolas/DNUPycharmProjects/machine_learning/audio/FMA/preprocessed_audio/gmm2", filename+".pkl")
    if os.path.exists(path):
        continue
        
    x = np.load(mfcc)
    x = np.swapaxes(x, 0, 1)
    x -= x.min()
    x /= x.max()
    
    best_estimator = evaluate_gmms_opt(vec = x, 
                                       n_gaussian = 100, 
                                       covariance = ["tied"], 
                                       step = 3, 
                                       filename = filename)

#     print(best_estimator)
    
    gmm = GaussianMixture(**best_estimator)
    gmm = gmm.fit(x)
    if gmm.converged_ :
        joblib.dump(gmm, path) 