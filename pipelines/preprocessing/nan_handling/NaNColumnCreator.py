# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
import pandas as pd
import numpy as np

from sklearn import set_config

set_config(transform_output="pandas")


class NaNColumnCreator(BaseEstimator, TransformerMixin):
    """
    NaNColumnCreator that creates new columns and marks NaN-Entries with 1

    Steps:
        (1) Checks if X has NaNs.\n
        (2) Marks the corresponding NaN-values with 1, else 0\n
    """

    def __init__(self):
        pass

    def filter_nans(self, X):
        nan_filter = X.isna()
        return nan_filter.map({False: 0, True: 1})

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.apply(self.filter_nans)
        self.column_names = X_transformed.columns

        return X_transformed

    def get_feature_names(self, input_features=None):
        return self.column_names



# from sklearn.impute import MissingIndicator
# preprocessing = ColumnTransformer(
#     [
# #         ("cat", prep_cat, make_column_selector(dtype_include=np.object_)),
# #         ("num", prep_num, make_column_selector(dtype_include=np.number)),
# #         ("tukey", TukeyTransformer(), make_column_selector(dtype_include=np.number)),
# #         ("z", Z_Transformer(threshold=3, z_scores_output=True), make_column_selector(dtype_include=np.number))
#         ("nan_marker_num", NaNColumnCreator(), make_column_selector(dtype_include=np.number)),
#         ("nan_marker_cat", NaNColumnCreator(), make_column_selector(dtype_include=np.object_))
#         # ("nan_marker_num", MissingIndicator(features="all"), make_column_selector(dtype_include=np.number)),
#         # ("nan_marker_cat", MissingIndicator(features="all"), make_column_selector(dtype_include=np.object_))

#     ],
#      remainder='passthrough' # this will pass through any columns not specified in the transformers
# )

# test_data = pd.DataFrame({"Example_num_column":np.array([1,2,3,4,5,100000, np.nan]),
#                           "Example_cat_column":["Katze","Hund","Hund","Katze",np.nan, np.nan, np.nan],
#                           "Example_cat_column_no_nans":["Katze","Hund","Hund","Katze","Hund","Hund","Katze"],
#                           "Example_nans":np.array([1,2,3,4,5,100000, np.nan]),
#                          "Example_no_nan":np.array([1,2,3,4,5,100000, 500])})

# preprocessed_data = preprocessing.fit_transform(test_data)
# preprocessed_data

# %%
