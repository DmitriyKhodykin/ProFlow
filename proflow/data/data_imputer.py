import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from proflow import config


class Imputer:

    def __init__(self):
        self.imputer = IterativeImputer(
            max_iter=10, 
            random_state=config.SEED,
        )

    def imput_data(self):
        self.imputer.fit([[1, 2], [3, 6], [4, 8], [np.nan, 3], [7, np.nan]])
        IterativeImputer(random_state=config.SEED)
        X_test = [[np.nan, 2], [6, np.nan], [np.nan, 6]]
        print(np.round(self.imputer.transform(X_test)))
