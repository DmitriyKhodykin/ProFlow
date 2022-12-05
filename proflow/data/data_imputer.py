import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer

from proflow import config


class Imputer:

    def __init__(
        self, 
        type="iterative",
    ):
        if type == "iterative":
            self.imputer = IterativeImputer(
                max_iter=10, 
                random_state=config.SEED,
            )
        
        elif type == "simple":
            self.imputer = SimpleImputer(
                missing_values=np.nan, 
                strategy="mean",
            )
        
        elif type == "knn":
            self.imputer = KNNImputer(
                n_neighbors=2, 
                weights="uniform",
            )

    def imput_data(
        self,
        df: pd.DataFrame,
    ):
        df = df.replace('', np.nan)
        df_imputed = self.imputer.fit_transform(df)
        return df_imputed
