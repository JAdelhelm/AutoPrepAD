# %%
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np
import re

import torch
from torch.utils.data import DataLoader

from pyod.models.auto_encoder_torch import AutoEncoder
# from pyod.models.auto_encoder import AutoEncoder
from collections import Counter

import os
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

def count_nan_entries(x):
    if str(x).lower() == "nan" or pd.isna(x) or x == np.nan:
        return 1
    else:
        return 0


class CategoricalPatternsAutoEncoder(BaseEstimator, TransformerMixin):
    """
    Features werden extrahiert pro Spalte und fließen dann
    in den AutoEncoder ein.
        - Die Rückgabe ist der Rekonstruktionsfehler für jede Spalte.
            - Pro Zeile für den entsprechenden Wert wird zurückgegeben, wie gut der
            Wert auf die Spalte passt. 
    """
    def __init__(self):
        self.new_feature_names = None


    def apply_AutoEncoder(self,X):
        """
        Schritte:
            (1) Merkmale aus Spalte extrahieren: Länge des Strings, Großbuchstaben, etc.
            (2) Erzeugen eines Rekonstruktionsfehlers basierend auf deren Merkmale
            (3) Rückgabe des Rekonstruktionsfehlers für die jeweilige Spalte.
            (4) Wiederholen für jede kategorische Spalte.
            !!! Achtung: Die Spalten, welche nur 0 Werte enthalten, enthalten keine Varianz und werden verworfen!
            !!! Anpassen der Batch-Size nach Größe des Datensatzes. Kann die Rechengeschwindigkeit enorm verbessern!
            batch_size = https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
            
        """ 
        X_transformed_ae = pd.DataFrame()

        results_list = []
        for col in X.columns:
            
            CounterBigLetters = pd.Series(X[col].apply(count_big_letters), name=col + "Big")
            CounterSmallLetters = pd.Series(X[col].apply(count_small_letters), name=col + "Small")
            LenString = pd.Series(X[col].apply(len_string), name=col + "Len")
            CounterNan = pd.Series(X[col].apply(count_nan_entries), name=col + "NaN")
            
            CounterNumbers = pd.Series(X[col].apply(count_numbers), name=col+"Numbers")
            CounterOperator = pd.Series(X[col].apply(count_operator), name=col+"Operator")
            
            CounterSpecialCharacters = pd.Series(X[col].apply(count_special_char), name=col+"Special")
            CounterUniqueCharacters = pd.Series(X[col].apply(count_unqiue_char), name=col+"Unique")


            
            X_features = pd.concat([CounterBigLetters,CounterSmallLetters,LenString,
                                    CounterNumbers,CounterOperator,CounterSpecialCharacters,
                                    CounterUniqueCharacters, CounterNan],axis=1)
            
            # Verwerfen von Spalten, in denen alle Werte 0 sind
            X_features = X_features.loc[:, (X_features != 0).any(axis=0)]

            # Verwerfen von Spalten mit einer Standardabweichung von 0 (überall die gleichen Werte)
            X_features = X_features.loc[:, X_features.std() != 0]

            print(X_features)

            
            X_features_as_tensor = torch.tensor(X_features.copy().values, dtype=torch.float)
            # print(torch.isnan(X_features_as_tensor).sum())

            features_shape = X_features_as_tensor.shape[1]
            # print(features_shape)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f'Using {device} device')
            X_features_as_tensor = X_features_as_tensor.to(device)

            batch_size_user = 2048

            print("Batch size: %i" % (batch_size_user))
            print("Column to analyze: %s" % col)

            try:
                num_cores = os.cpu_count() -1
                torch.set_num_threads(num_cores)
                print(f"{num_cores} cores will be used...")
                           
                clf_ae = AutoEncoder(epochs=50, preprocessing=True, batch_size=batch_size_user, learning_rate=0.01)

                clf_results = clf_ae.fit(X_features_as_tensor)

            except:
                print("Could not use all CPU Threads...")
                clf_ae = AutoEncoder(epochs=50, preprocessing=True,  batch_size=batch_size_user, learning_rate=0.01)
                clf_results = clf_ae.fit(X_features_as_tensor)


            X_results_temp = pd.DataFrame(data=clf_results.decision_scores_, columns=[col+"_reconstructionE"])

            results_list.append(X_results_temp)

        X_transformed_ae = pd.concat(results_list, axis=1)

        self.new_feature_names = X_transformed_ae.columns

        return X_transformed_ae

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.apply_AutoEncoder(X)

    def get_feature_names(self, input_features=None):
        # Rückgabe der Featurenamen
        return self.new_feature_names
    


# from sklearn.compose import ColumnTransformer
# from sklearn.compose import make_column_selector
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
# from sklearn.impute import SimpleImputer


# preprocessing = Pipeline(steps=[
#         ('AutoEncoder Similarity',
#             ColumnTransformer(transformers=[

#             ("_similarity_autoencoder",
#                 Pipeline(
#                     steps=[
#                     # ("C", SimpleImputer(strategy="most_frequent")),
#                     # ("OHE", OneHotEncoder(handle_unknown="ignore",sparse=False)),
#                     # ("OrdinalEnc",OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
#                     ("AutoEncoder", CategoricalPatternsAutoEncoder())
#                     ]), make_column_selector(dtype_include=np.object_)
#             )
#             ], remainder='passthrough', n_jobs=-1, verbose=True))
#         ])


# test_daten = pd.DataFrame(
#     {
#     "COLTestCAT1":np.array(["Hund","Hund123","hund"]),
#     "COLTestCAT2":np.array(["K**atze",np.nan,"Hu*nd$"]),
#     "timestamp":np.array(["2023-02-08 06:58:14.017000+00:00","2023-02-08 15:54:13.693000+00:00", np.nan])
#      })

# preprocessed_data = preprocessing.fit_transform(test_daten)
# preprocessed_data



# def set_all_cores(self):
#     """
#     Versuch, alle Kerne der CPU für das Modelltraining zu verwenden.
#     """
#     try:
#         # # Anzahl der verfügbaren CPU-Kerne ermitteln und setzen
#         num_cores = os.cpu_count()
#         os.environ['OMP_NUM_THREADS'] = str(num_cores)

#         num_cores = torch.get_num_threads()
#         torch.set_num_threads(num_cores)
#         print(f"{num_cores} will be used...")
#     except:
#         print("An error occured. Can not use all CPU-Threads")