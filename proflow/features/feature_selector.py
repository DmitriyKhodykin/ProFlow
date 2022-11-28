import pandas as pd


def set_feature_selection_method(
    df: pd.DataFrame, 
    target: str,
) -> dict:
    """
    The logic of method selection.

    1) pearson_corr - Both independent (X) and dependent (y) variable 
    are all numerical variables.

    2) spearmans_rank - Independent variable is numerical and 
    dependent variable is categorical.

    3) chi_square - Both independent and dependent variables 
    are all categorical variables.

    4) anova - Independent variable is categorical (>= 3 cat.) and 
    dependent variable is numerical.

    5) t-test (only 2 categories).
    """
    # pearson_corr
    # spearmans_rank
    # chi_square
    # ANOVA
    result = {}
    return result