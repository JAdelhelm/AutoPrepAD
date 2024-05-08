# %%
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np
import re

from bitstring import BitArray
from collections import Counter

from sklearn import set_config

set_config(transform_output="pandas")


class CategoricalPatterns(BaseEstimator, TransformerMixin):
    """
    PatternTransformer to extract certain patterns from categorical data.

    This transformer is designed to extract patterns from features and represent them in a binary format:
        - Big Letters: "00" / Small Letters: "01" / Non-Alphanumeric: "10" / Remaining: "11"\n
    Example "Dog":
        -"00 01 01" translates to 40.

    Steps:
        (1) Iterate over every single Column in X.\n
        (2) Create two lists: temp_word_encoded and count_len\n
            (2.1) temp_word_encoded is used to save the corresponding decimals w.r.t. the binary transformation\n
        (3) Iterate over every row in X.\n
            (3.1) Convert that row into a binary format.\n
            (3.2) Convert that binary into an int/decimal.\n
        (4) Parse the corresponding values to a new DataFrame.\n
        (5) Return the new DataFrame with the corresponding length of every value entry and binary representation.\n

    """

    def __init__(self):
        self.new_feature_names = None

    def extract_patterns(self, X):
        X_Transformed = pd.DataFrame()

        for col in X.columns:
            # print("Actual column: ", col)

            temp_word_encoded = []
            count_len = []
            # count_unique = []
            for value in X[col].values:
                str_w_encoded = []
                count_len.append(len(value))
                # count_unique.append(len(set(value)))

                for character in value:
                    if character.isupper():
                        str_w_encoded.append("00")
                    elif character.islower():
                        str_w_encoded.append("01")
                    elif character.isdigit():
                        str_w_encoded.append("10")
                    else:
                        str_w_encoded.append("11")
                # Keine Invertierung notwendig, da bereits hinten angehängt
                str_w_encoded = "0" + "".join(str_w_encoded)
                # print(str_w_encoded)
                # print(count_len)
                # Has to be str, else the BinaryEncoder cant handle it.
                str_w_encoded = str(BitArray(bin=str_w_encoded))
                temp_word_encoded.append(str_w_encoded)

            X_Transformed[col] = temp_word_encoded
            X_Transformed[col + "_len"] = count_len
            # X_Transformed[col + "_unique"] = count_unique

        self.new_feature_names = X_Transformed.columns

        return X_Transformed

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X.eq(0).all().all():
            return X
        else:
            return self.extract_patterns(X)

    def get_feature_names(self, input_features=None):
        # Rückgabe der Featurenamen
        return self.new_feature_names


test_daten = pd.DataFrame(
    {
        "COLTestCAT1": np.array(["Hund","Hund", "Hund123", "hund"]),
        "COLTestCAT2": np.array(["K*atze","K*atze", np.nan, "Hund$"])
        # "timestamp": np.array(["2023-02-08 06:58:14.017000+00:00", "2023-02-08 15:54:13.693000+00:00", np.nan])
    })

# from category_encoders import BinaryEncoder
# from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# preprocessing = Pipeline(steps=[
#     ('PatternExtraction',
#      ColumnTransformer(transformers=[

#          ("pattern",
#           Pipeline(
#               steps=[
#                          # ("C", SimpleImputer(strategy="most_frequent")),
#                          # ("OHE", OneHotEncoder(handle_unknown="ignore",sparse=False)),
#                          # ("OrdinalEnc",OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
#                         ("Pattern", CategoricalPatterns()),
#                         # ("OHE",OneHotEncoder(sparse_output=False)),
#                         # ("OrdinalEnc",OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
#                         ("BinaryEnc",BinaryEncoder(handle_unknown="indicator")),
#                          ]), make_column_selector(dtype_include=np.object_)
#           )
#      ], remainder='passthrough', n_jobs=-1, verbose=True))
# ])


# preprocessed_data = preprocessing.fit_transform(test_daten)
# preprocessed_data

# %%
