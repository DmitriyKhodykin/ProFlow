"""AutoML Module."""

import pandas as pd
from prettytable import PrettyTable

from proflow.model.binary_models import BinaryTabularModels

class FlowML:

    def __init__(self, task: str):
        self.task = task
        self.score = None
        self.fitted_model = None

    def fit(
        self,
        train_df: pd.DataFrame,
        label: str,
    ) -> None:
        if self.task == "binary":
            btm = BinaryTabularModels()
            fitted_binary_model = btm.fit(
                train_df,
                label
            )
            self.fitted_model = fitted_binary_model

    def score(self):
        score_table = PrettyTable()

        if self.run_model_results is not None:
            score_table.field_names = ["Score", "Value"]
            for i in range(len(self.score)):
                score_table.add_row(["train_df", self.score[i]])
            print(score_table)

    def predict(self, test_df: pd.DataFrame):
        if self.fitted_model is not None:
            y_predicted = self.fitted_model.predict(test_df)
            return y_predicted

    def predict_proba(self):
        pass

    def save_best_model(self):
        pass
