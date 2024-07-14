# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
import pandas as pd
import numpy as np

from sklearn import set_config

from scipy.stats import spearmanr

import warnings
set_config(transform_output="pandas")


class SpearmanCorrelationCheck(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self, factor=1.5):
        self.factor = factor

    def check_correlation(self, X):
        for i in range(len(X.columns)):
            for j in range(i+1, len(X.columns)):
                col1 = X.columns[i]
                col2 = X.columns[j]
                correlation_coefficient, p_value = spearmanr(X[col1], X[col2])

                if np.abs(correlation_coefficient) >= 0.98:
                    print("**************************************")
                    print(f"Possible correlation between columns:\n\n-> {col1} ;; {col2}\n\nCheck for duplicates.")
                    print("**************************************")
                    # raise Exception(f"Possible correlation between columns:\n {col1} ;; {col2}\n -> Check for duplicates.")


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.check_correlation(X)
        return X

    def get_feature_names(self, input_features=None):
        return [col for col in input_features]