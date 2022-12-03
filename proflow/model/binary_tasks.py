import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from proflow.model import model_config as config


class BinaryTabularModels:

    def __init__(self):
        seed: int = config.SEED
        self.models: dict = config.binary_models
        print("BinaryTabularModels:", self.models.keys())

        self.lr_model = LogisticRegression(
            penalty='l2',
            dual=False, 
            tol=0.0001, 
            C=1.0, 
            fit_intercept=True, 
            intercept_scaling=1, 
            class_weight=None,
            random_state=seed,
            solver='lbfgs', 
            max_iter=100, 
            multi_class='auto', 
            verbose=0, 
            warm_start=False, 
            n_jobs=None, 
            l1_ratio=None,
        )

        self.sgd_model = SGDClassifier(
            loss='hinge',
            penalty='l2', 
            alpha=0.0001, 
            l1_ratio=0.15, 
            fit_intercept=True, 
            max_iter=1000, 
            tol=0.001, 
            shuffle=True, 
            verbose=0, 
            epsilon=0.1, 
            n_jobs=-1, 
            random_state=seed, 
            learning_rate='optimal', 
            eta0=0.0, 
            power_t=0.5, 
            early_stopping=False, 
            validation_fraction=0.1, 
            n_iter_no_change=5, 
            class_weight=None, 
            warm_start=False, 
            average=False,
        )

    def fit(
        self, 
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        label: str,
    ):
        for model in [self.lr_model, self.sgd_model]:
            model.fit(
                train_df.drop([label], axis=1),
                train_df[label],
            )

    def score(self):
        pass

    def predict(self):
        pass

    def predict_proba(self):
        pass
