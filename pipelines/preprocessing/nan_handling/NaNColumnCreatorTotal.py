# %%


from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

from bitstring import BitArray


from sklearn import set_config

set_config(transform_output="pandas")


class NaNColumnCreatorTotal(BaseEstimator, TransformerMixin):
    """
    NaNColumnCreator that creates new columns and marks NaN-Entries with 1.

    Steps:
        (1) The numerical values are concatenated row-wise and converted into a string.\n
        (2) Subsequently, they are inverted so that binary sorting is done from left to right (Simpler interpretation of the numbers).\n
        (3) A 0 is added to the beginning due to the two's complement to obtain a positive number.\n
        !! Attention: There is no (Z-)scaling necessary here, as an ordinal order makes sense!\n
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.astype(int)
        try:
            X_Transformed = pd.DataFrame()

            transformed_rows = []

            for row in X.values:
                # Casten zum String + Inverse (Da man pro Zeile in der Spalte nur 0 und 1 hat)
                binary_str = "".join(map(str, row))[::-1]
                binary_str = "0" + binary_str
                integer_representation = BitArray(bin=binary_str).int
                transformed_rows.append(integer_representation)

            X_Transformed["NaNs-Binary"] = transformed_rows

            self.column_names_missing = X_Transformed.columns

            return X_Transformed
        except Exception as e:
            print(f"Error: {e}")
            return X

    def get_feature_names(self, input_features=None):
        return self.column_names_missing


# from NaNColumnCreator import NaNColumnCreator
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer, make_column_selector
# from sklearn.preprocessing import OneHotEncoder
# from category_encoders import BinaryEncoder
# from sklearn.impute import MissingIndicator
# categorical_preprocessor = Pipeline(steps=[
#     ('Preprocessing_Categorical',
#      ColumnTransformer(transformers=[
#          ("nan_marker_cat",
#           Pipeline(
#               steps=[
#                 #   ("nan_marker_cat", NaNColumnCreator()),
#                 #   ("nan_marker_cat", MissingIndicator(features="all")),
#                   ("nan_marker_cat_total_nan", NaNColumnCreatorTotal()),
#                 #   ("OneHotEncoder", BinaryEncoder(handle_unknown="indicator")),
#                 #   ("OneHotEncoder", OneHotEncoder(sparse_output=False)),
#               ]), make_column_selector(dtype_include=np.object_)
#           ),
#      ], remainder='drop', n_jobs=-1, verbose=True))
# ])

# test_data = pd.DataFrame({
#     #  "Example_num_column":np.array([1,2,3,4,5,100000, np.nan]),
#     "Example_cat_column": ["Tiger", np.nan, "Elefant", "Elefant"],
#     "Example_cat2_column": ["Katze", np.nan, "Hund", np.nan],
#     #   "Example_cat_column_no_nans":["Katze","Hund","Hund","Katze","Hund","Hund",np.nan],
#     #   "Example_nans":np.array([1,2,3,4,5,100000, np.nan]),
#     #  "Example_no_nan":np.array([1,2,3,4,5,100000, 500])
# })

# preprocessed_data = categorical_preprocessor.fit_transform(
#     test_data)
# preprocessed_data


# %%
