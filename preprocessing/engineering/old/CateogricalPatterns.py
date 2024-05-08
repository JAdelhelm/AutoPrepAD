# %%
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np
import re


from collections import Counter

from sklearn import set_config
set_config(transform_output="pandas")

def count_big_letters(x):
    if str(x).lower() == "nan":
        return 0
    else:
        return len(re.findall("[A-Z]", str(x)))

def count_small_letters(x):
    if str(x).lower() == "nan":
        return 0
    else:
        return len(re.findall("[a-z]", str(x)))
    
def len_string(x):
    if str(x).lower() == "nan":
        return 0
    else:
        return len(str(x))
    
def count_numbers(x):
    return len(re.findall("[0-9]", str(x)))

def count_operator(x):
    return len(re.findall("[+:=|&<>^*-]", str(x)))

def count_special_char(x):
    return len(re.findall("[$,;?@#'.()%!]", str(x)))

def count_unqiue_char(x):
    return len(Counter(str(x)))

class CategoricalPatterns(BaseEstimator, TransformerMixin):
    """
    Features werden extrahiert pro Spalte und fließen dann
    in den AutoEncoder ein.
    """
    def __init__(self):
        """
        Schritte:
            (1) Es wird in die extract_patterns Methode gesprungen.
            (2) Dort werden verschiedene Muster aus den Daten extrahiert und in eine eigene Spalte geschrieben.
        """ 
        self.new_feature_names = None

    def extract_patterns(self,X):
        X_transformed_pattern = pd.DataFrame()
        for col in X.columns:
            
            CounterBigLetters = pd.Series(X[col].apply(count_big_letters), name=col + "Big")
            CounterSmallLetters = pd.Series(X[col].apply(count_small_letters), name=col + "Small")
            LenString = pd.Series(X[col].apply(len_string), name=col + "Len")
            # CounterNan = pd.Series(X[col].apply(lambda x: 1 if str(x).lower() == "nan" or pd.isna(x) else 0))
            
            CounterNumbers = pd.Series(X[col].apply(count_numbers), name=col+"Numbers")
            CounterOperator = pd.Series(X[col].apply(count_operator), name=col+"Operator")
            
            CounterSpecialCharacters = pd.Series(X[col].apply(count_special_char), name=col+"Special")
            CounterUniqueCharacters = pd.Series(X[col].apply(count_unqiue_char), name=col+"Unique")


            
            X_features = pd.concat([CounterBigLetters,CounterSmallLetters,LenString,
                                    CounterNumbers,CounterOperator,CounterSpecialCharacters,
                                    CounterUniqueCharacters],axis=1)
            
            # Verwerfen von Spalten, in denen alle Werte 0 sind - Hier nicht sinnvoll, da kein AutoEncoder verwendet wird
            # und somit Trainings- und Testteil gleich sein muss.
            # X_features = X_features.loc[:, (X_features != 0).any(axis=0)]


            X_transformed_pattern = pd.concat([X_transformed_pattern, X_features], axis=1)


        self.new_feature_names = X_transformed_pattern.columns

        return X_transformed_pattern

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.extract_patterns(X)

    def get_feature_names(self, input_features=None):
        # Rückgabe der Featurenamen
        return self.new_feature_names
    


# from sklearn.compose import ColumnTransformer
# from sklearn.compose import make_column_selector
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
# from sklearn.impute import SimpleImputer


# preprocessing = Pipeline(steps=[
#         ('PatternExtraction',
#             ColumnTransformer(transformers=[

#             ("_pattern_",
#                 Pipeline(
#                     steps=[
#                     # ("C", SimpleImputer(strategy="most_frequent")),
#                     # ("OHE", OneHotEncoder(handle_unknown="ignore",sparse=False)),
#                     # ("OrdinalEnc",OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
#                     ("Pattern", CategoricalPatterns())
#                     ]), make_column_selector(dtype_include=np.object_)
#             )
#             ], remainder='passthrough', n_jobs=-1, verbose=True))
#         ])


# test_daten = pd.read_csv('./train_min_february_1_to_7.csv')

# preprocessed_data = preprocessing.fit_transform(test_daten)
# preprocessed_data
