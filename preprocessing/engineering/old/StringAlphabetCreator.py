# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
import pandas as pd
import numpy as np
import re

from sklearn import set_config
set_config(transform_output="pandas")

class StringAlphabetCreatorBig(BaseEstimator, TransformerMixin):
    """
    Klasse, welche die Anzahl an Großbuchstaben zählt.
    """
    def __init__(self):
        self.column_names = []

    def num_alphabet(self,X):
        return X.apply(lambda x: -1 if str(x).lower() == "nan" else len(re.findall("[A-Z]", str(x))))


    def fit(self, X, y=None):
        # Gibt für jede einzelne Spalte noch zusätzlich ein Integer-Wert mit, damit zwichen ihnen differenziert werden kann
        for i in range(len(X.columns)):
            self.column_names.append(X.columns[i]+"_LEN_ALPHABET_BIG_"+str(i))

        X.columns = self.column_names
        return self

    def transform(self, X):
        return X.apply(self.num_alphabet)

    def get_feature_names(self, input_features=None):
        # Rückgabe der Featurenamen
        return [col for col in input_features]

class StringAlphabetCreatorSmall(BaseEstimator, TransformerMixin):
    """
    Klasse, welche die Anzahl an Kleinbuchstaben zählt.
    """
    def __init__(self):
        self.column_names = []

    def num_alphabet(self,X):
        return X.apply(lambda x: -1 if str(x).lower() == "nan" else len(re.findall("[a-z]", str(x))))


    def fit(self, X, y=None):
        # Gibt für jede einzelne Spalte noch zusätzlich ein Integer-Wert mit, damit zwichen ihnen differenziert werden kann
        for i in range(len(X.columns)):
            self.column_names.append(X.columns[i]+"_LEN_ALPHABET_SMALL_"+str(i))

        X.columns = self.column_names
        return self

    def transform(self, X):
        return X.apply(self.num_alphabet)

    def get_feature_names(self, input_features=None):
        # Rückgabe der Featurenamen
        return self.column_names
        # return [col for col in input_features]


# preprocessing = ColumnTransformer(
#     [
#         ("str_alphabet_creator_big", String_Alphabet_Creator_Big(), make_column_selector(dtype_include=np.object_)),
#         ("str_alphabet_creator_small", String_Alphabet_Creator_Small(), make_column_selector(dtype_include=np.object_))

#     ],
#      remainder='passthrough'
# )

# test_data = pd.DataFrame({"Example_num_column":np.array([1,2,3,4,5,100000, np.nan]),
#                           "Example_cat_column":["456","Hund","123","Kat12",np.nan, np.nan, np.nan],
#                          "Example_no_nan":np.array([1,2,3,4,5,100000, 500])})

# preprocessed_data = pd.DataFrame(preprocessing.fit_transform(test_data), columns=preprocessing.get_feature_names_out())
# preprocessed_data

# %%
