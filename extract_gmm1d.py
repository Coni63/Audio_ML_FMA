import sys
import os
import glob

import pandas as pd
import numpy as np

from sklearn.mixture import GaussianMixture

def extract_gmm_to_df(start):
    data = {
        "file" : [],
        "mel" : [],
        "weigth" : [],
        "means" : [],
        "covariance" : []
    }

    for mfcc in glob.glob("F:/Nicolas/DNUPycharmProjects/machine_learning/audio/FMA/preprocessed_audio/cqt/*.npy")[start:]:
        filename = os.path.basename(mfcc)[:-4]

        x = np.load(mfcc)
        x = np.swapaxes(x, 0, 1)

        for i in range(84):
            best_bic = 1e6
            for n in range(1, 10):
                vec = x[:, i].reshape(-1, 1)
                gmm = GaussianMixture(n_components = n, covariance_type="full")
                gmm = gmm.fit(vec)
                current_bic = gmm.bic(vec)
                if current_bic < best_bic:
                    best_bic = current_bic
                    best_gmm = gmm
                    n_elem = n
            data["file"] += [filename]*n_elem
            data["mel"] += [i]*n_elem
            data["weigth"] += list(best_gmm.weights_.flatten())
            data["means"] += list(best_gmm.means_.flatten())
            data["covariance"] += list(best_gmm.covariances_.flatten())
    df = pd.DataFrame(data)
    df.to_csv("preprocessed_meta/gmm/{}.csv".format(start//100))
	
if __name__ == "__main__":
	extract_gmm_to_df(int(sys.argv[1]))