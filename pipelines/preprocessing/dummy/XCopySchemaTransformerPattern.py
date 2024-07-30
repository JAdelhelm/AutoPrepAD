# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline

import warnings

# Warnungen vom Versuch des castens ignorieren
warnings.filterwarnings("ignore")

from sklearn import set_config

set_config(transform_output="pandas")


class XCopySchemaTransformerPattern(BaseEstimator, TransformerMixin):
    """
    SchemaTransformer for a certain Pandas DataFrame input.

    Steps:
        (1) Attempt to convert object columns into a better data type format.\n
        (2) Attempt to convert columns with a time series schema into the correct data type.\n
        (3) Attempt to convert numerical data with the incorrect data type into the correct data type.\n
            Example: "col1": [1, "2", 3]  to "col1": [1, 2, 3]\n
        (4) NaN values are formatted correctly for subsequent processing.\n
        (5) Return of the adjusted dataframe.\n


    Parameters
    ----------
    datetime_columns : list
        List of certain Time-Columns that should be converted in timestamp data types.

    exclude_columns : list
        List of Columns that will be dropped.

    name_transformer : list
        Is used for the output, so the enduser can check what Columns are used for a certain Transformation.

    """

    def __init__(
        self, datetime_columns=None, exclude_columns: list = None, name_transformer=""
    ):
        self.datetime_columns = datetime_columns

        self.exclude_columns = exclude_columns
        self.feature_names = None
        self.name_transformer = name_transformer

    def convert_schema_nans(self, X):
        X_Copy = X.copy()

        for col in X_Copy.columns:
            X_Copy[col] = X_Copy[col].replace("NaN", np.nan)
            X_Copy[col] = X_Copy[col].replace("nan", np.nan)
            X_Copy[col] = X_Copy[col].replace(" ", np.nan)
            X_Copy[col] = X_Copy[col].replace("", np.nan)
        return X_Copy

    def infer_schema_X(self, X_copy):
        try:
            X_copy = X_copy.infer_objects()
        except:
            pass

        for col in X_copy.columns:
            if X_copy[col].dtype == "object":
                if self.datetime_columns is not None and col in self.datetime_columns:
                    try:
                        X_copy[col] = pd.to_datetime(
                            X_copy[col], infer_datetime_format=True, errors="coerce"
                        )
                        # print("\nColumns to time dtype:", col, "\n")
                    except:
                        pass
                else:
                    try:
                        X_copy[col] = pd.to_datetime(
                            X_copy[col], infer_datetime_format=True
                        )
                        # print("\nColumns to time dtype:", col, "\n")
                    except:
                        pass

                try:
                    X_copy[col] = X_copy[col].astype(np.float64)
                    # print("\nColumns to numeric dtype:", col, "\n")
                except (ValueError, TypeError):
                    pass

        X_copy.convert_dtypes()

        return X_copy

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> pd.DataFrame:
        if self.exclude_columns is not None:
            for col in self.exclude_columns:
                try:
                    X.drop([col], axis=1, inplace=True)
                except:
                    print(f"Column {col} could not be dropped.")

        self.feature_names = X.columns

        X_copy = X.copy()
        X_copy = self.convert_schema_nans(X_copy)

        X_copy = self.infer_schema_X(X_copy=X_copy)

        print(f"\n\nDtypes-Schema / Columns for {self.name_transformer}:\n")
        print(X_copy.dtypes, "\n")

        return X_copy

    def get_feature_names(self, input_features=None):
        return self.feature_names

    # def convert_column_to_naive_timestamp(self, X, column_name):
    #     """
    #     Konvertieren einer 'datetime64[ns, UTC]' Spalte zu 'datetime64[ns]'
    #     """
    #     try: return X[column_name].dt.tz_convert(None)
    #     except: pass


# Beispielverwendung
# data = pd.DataFrame({
#     'evseid': [
#         'IT*DUF*DAXS20*2', 'ANOMALY', 'RO*RNVED166*01*1', # evseid zweite Zeile
#         'SE*CLE*E20665*1', 'SE*CLE*E2349*1', 'PT*HRZ*E*PRT*00123*02'
#     ],
#     'locationId': [196133.0, 224509.0, -9999.0, 225551.0, 148382.0, 228302.0], # locationId 3 Zeile
#     'uuid': [
#         'INVALID_UUID', # Anomalous uuid in first row
#         'd08f6f93-cea0-4a6f-a067-de6612e443b7', '69bf02ba-8530-4352-a6b1-31ae9f8c570e',
#         'b755e455-2bff-42d7-83c9-c4f1d37ba5b3', 'c2621085-7f31-4c64-bf8a-b632ecee0fa3',
#         'be999c75-71bf-4b68-906e-31ea42a3a824'
#     ],
#     'platform': ['HUBJECT', 'ANOMALY_PLATFORM', 'HUBJECT', 'HUBJECT', 'ANOMALY_PLATFORM', 'HUBJECT'], # Platform in zweiter und vierter Zeile
#     'oldAvailability': ['AVAILABLE', 'AVLAILABLE', 'AVAILABLE', 'OCCUPIED', 'AVAILABLE', 'AVAILABLE'], # Zweite Zeile falsch geschrieben
#     'oldTimestamp': [
#         '2023-02-08T13:36:14.342Z', '2023-01-09T14:28:13.529Z', '2023-02-09T14:26:12.992Z',
#         '2023-02-10T12:38:15.322Z', '2023-02-09T14:26:13.185Z', '2023-02-09T14:28:13.408Z' # Falsches Datum in zweiter Zeile
#     ],
#     'availability': ['OCCUPIED', 'OCCUPIED', 'OCCUPIED', 'AVAILABLE', 'OCCUPIED', 'OCCUPIED'],
#     'timestamp': [
#         '2023-02-09 15:38:14.954000+00:00', '2023-02-09 15:30:12.996000+00:00',
#         '2023-02-09 14:28:13.381000+00:00', '2023-02-09 13:28:13.568000+00:00',
#         '2023-02-09 15:32:14.362000+00:00', '2022-02-08 14:30:16.858000+00:00' # Falscher Zeitstempel in letzter Zeile Monat
#     ],
#         'y_true': [
#         1, 1, 1, 0, 1, 1
#     ]
# })

# preprocessor = make_pipeline(
#             ColumnTransformer(transformers=[
#                 ("XCopy", XCopySchemaTransformer(), make_column_selector(dtype_include=None))
#             ], remainder="passthrough", n_jobs=-1),
#             ColumnTransformer(transformers=[
#                 ("C_imputed", SimpleImputer(strategy="most_frequent", missing_values=np.nan), make_column_selector(dtype_include=np.object_)),
#                 ("N_imputed", SimpleImputer(strategy="median"), make_column_selector(dtype_include=np.number)),
#             ], remainder="passthrough", n_jobs=-1)
#             )

# transformed_data = preprocessor.fit_transform(data)
