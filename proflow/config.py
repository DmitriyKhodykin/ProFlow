# COMMON
SEED = 23

# DATA
TEST_SIZE =  0.3
SAMPLING = 50_000
BUCKETS_FOR_HASH = 100

# MODELS
task_type = [
    "regression",
    "binary",
    "multiclass"
]

metrics = {
    "regression": ["mae", "mape", "r2", "rmse"],
    "binary": ["accuracy", "roc_auc", "precision", "recall", "f1"],
    "multiclass": ["accuracy", "f1_weighted"]
}

regression_models ={
    "LinearRegression": "reg_model"
}

binary_models = {
    "LogisticRegression": "lr_model",
    "SGDClassifier": "sgd_model"
}

multiclass_models = {

}

regularization = {
    "Ridge": "ridge",
    "Lasso": "lasso"
}

params = {
    "regression": {},
    "binary": {},
    "multiclass": {}
}