
import pandas as pd
from json import dump
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn import set_config
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import pyod.models
from functools import reduce

from pipelines.experiment.experiment import Experiment

class PipelineRuns(Experiment):
    def __init__(self,
                 PipelineStructure: object,
                 remove_columns_with_no_variance = False,
                 exclude_columns = None,
                 mark_anomalies_pct_data=0.1) -> None:
        self.PipelineStructure = PipelineStructure
        self.remove_columns_with_no_variance = remove_columns_with_no_variance
        self.exclude_columns = exclude_columns
        self.mark_anomalies_pct_data = mark_anomalies_pct_data




        self.fitted_model = None
        self.fitted_pipeline = None


        self._X_train = None
        self._X_test = None

        self.X_train_prep = None
        self.X_test_prep = None

        self._X_train_transformed = None
        self._X_test_transformed = None

        self.X_train_transformed_model = None
        self.X_test_transformed_model = None

        self.no_variance_columns = None

    @property
    def X_train_transformed(self):
        return self._X_train_transformed

    @property
    def X_test_transformed(self):
        return self._X_test_transformed
    
    @property
    def X_train(self):
        return self._X_train
    
    @property
    def X_test(self):
        return self._X_test
    

    def fit_pipeline(
            self, 
            X_train: pd.DataFrame,
            clf: pyod.models = None,
            dump_model: bool = False,
    ):
        """
        Fitting of Pipeline and anomaly detection algorithm.

            1. Remove Columns that should be excluded.
            2. Remove Columns with no variance.
                --> Before checking variance, data has to be transformed to numeric
            3. Fit Pipeline based on input data (train data).
            4. Fit Anomaly Detection algorithm with transformed input data.


        """    

        print("Fitting Pipeline and train anomaly detection model...")
        self.X_train_prep = self.remove_excluded_columns(X_train)

        try:
            self.fitted_pipeline = self.PipelineStructure.fit(self.X_train_prep)
        except Exception as e:
            print(self.X_train_prep.isna().sum(),"\n",e,"\n")
            raise

        self.X_train_transformed_model = self.fitted_pipeline.transform(self.X_train_prep)

        self.X_train_transformed_model = self.remove_no_variance_columns(
            X_train=self.X_train_transformed_model,
            remove_no_variance=self.remove_columns_with_no_variance,
        )



        self.fitted_model = clf.fit(self.X_train_transformed_model)

        if dump_model == True:
            try:
                dump(self.fitted_model, f"clf_{type(self.fitted_model).__name__}.joblib")
            except:
                print("Could not dump the model.")




        

    def predict_pipeline(
            self,
            X_test: pd.DataFrame
    ):
        """
        Transformation, based on fitted Pipeline + Prediction.

            1. Remove Columns that should be excluded.
            2. Remove Columns with no variance.
            3. Predict Anomalies on transformed input data (test data)
        """
        print("Prediction of Anomalies, based on fitted Pipeline...")
        self.X_test_prep = self.remove_excluded_columns(X_test)


        self.X_test_transformed_model = self.fitted_pipeline.transform(self.X_test_prep)

        self.X_test_transformed_model = self.remove_no_variance_columns(
            X_train=self.X_test_transformed_model,
            remove_no_variance=self.remove_columns_with_no_variance,
        )



        y_pred_decision_score = self.fitted_model.decision_function(self.X_test_transformed_model)
        X_test["AnomalyScore"] = y_pred_decision_score
        scaler = MinMaxScaler()
        X_test[["AnomalyScore"]] = scaler.fit_transform(X_test[["AnomalyScore"]])

        try:
            column_name_mad_total = [
                col for col in self.X_test_transformed_model.columns if col.endswith("MAD_Total")
            ][0]
            X_test["MAD_Total"] = self.X_test_transformed_model[column_name_mad_total]
            column_name_tukey_total = [
                col for col in self.X_test_transformed_model.columns if col.endswith("Tukey_Total")
            ][0]
            X_test["Tukey_Total"] = self.X_test_transformed_model[column_name_tukey_total]

            X_test = X_test.sort_values(
                ["AnomalyScore", "MAD_Total", "Tukey_Total"], ascending=False
            )


            first_column = X_test.pop("AnomalyScore")
            X_test.insert(0, "AnomalyScore", first_column)

            threshold_AD = np.percentile(first_column, 100 * (1 - self.mark_anomalies_pct_data))
            y_pred_array = (first_column > threshold_AD).astype(int)

            X_test["AnomalyLabel"] = y_pred_array

            
            self._X_train = X_test

            return X_test
        except Exception as e:
            threshold_AD = np.percentile(first_column, 100 * (1 - self.mark_anomalies_pct_data))
            y_pred_array = (first_column > threshold_AD).astype(int)
            X_test["AnomalyLabel"] = y_pred_array
            return X_test.sort_values("AnomalyScore", ascending=False)




    def remove_excluded_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes columns that are predefined in order to exclude them from the input data.
        """
        df_modified = df.copy()
        if self.exclude_columns is not None:
                for col in self.exclude_columns:
                    try:
                        df_modified.drop([col], axis=1, inplace=True)
                    except Exception as e:
                        print(e)
        return df_modified



    def remove_no_variance_columns(
        self, X_train, X_test=None,       remove_no_variance=False
    ) -> (pd.DataFrame, pd.DataFrame):

        """
        Can only remove no variance columns on transformed data --> numeric format
        --> Have to transform both (Train and test), because we need to remove both sides of columns for concistency.
        """

        if X_test is None:
            X_train_cols_no_variance = X_train.loc[:, X_train.std() == 0].columns
            print("No Variance in follow Train Columns: ", X_train_cols_no_variance)
            X_train_cols_only_nans = X_train.columns[X_train.isna().any()]
            print("Only NaNs in follow Train Columns: ", X_train_cols_only_nans)


            drop_columns_no_variance = []

            lists_to_combine = [
                X_train_cols_no_variance,
                X_train_cols_only_nans,
            ]

            for lst in lists_to_combine:
                if isinstance(lst, pd.Index):
                    if not lst.empty:
                        drop_columns_no_variance.extend(lst.tolist())
                else:
                    if lst:
                        drop_columns_no_variance.extend(lst)
 
            # print(drop_columns_no_variance)

            print(f"Remove columns with zero variance: {remove_no_variance}")

            print(
                f"Shape Train before drop:{X_train.shape} "
            )

            if remove_no_variance == True:
                X_train_dropped = X_train.drop(drop_columns_no_variance, axis=1)

                print(
                    f"Shape Train after drop: {X_train_dropped.shape} \n"
                )
                print(
                    f"Check NaN Train: {X_train_dropped.columns[X_train_dropped.isna().any()].tolist()} "
                )
                print(
                    f"Check inf Train: {X_train_dropped.columns[np.isinf(X_train_dropped).any()].tolist()} "
                )
                return X_train_dropped
            else:
                return X_train

        else:
            X_train_cols_no_variance = X_train.loc[:, X_train.std() == 0].columns
            X_test_cols_no_variance = X_test.loc[:, X_test.std() == 0].columns
            print("No Variance in follow Train Columns: ", X_train_cols_no_variance)
            print("No Variance in follow Test Columns: ", X_test_cols_no_variance)

            X_train_cols_only_nans = X_train.columns[X_train.isna().any()]
            X_test_cols_only_nans = X_test.columns[X_test.isna().any()]
            print("Only NaNs in follow Train Columns: ", X_train_cols_only_nans)
            print("Only NaNs in follow Test Columns: ", X_test_cols_only_nans)



            drop_columns_no_variance = []

            lists_to_combine = [
                X_train_cols_no_variance,
                X_test_cols_no_variance,
                X_train_cols_only_nans,
                X_test_cols_only_nans
            ]

            for lst in lists_to_combine:
                if isinstance(lst, pd.Index):
                    if not lst.empty:
                        drop_columns_no_variance.extend(lst.tolist())
                else:
                    if lst:
                        drop_columns_no_variance.extend(lst)
            # print(drop_columns_no_variance)

            print(f"Remove columns with zero variance: {remove_no_variance}")

            print(
                f"Shape Train before drop:{X_train.shape} / Shape Test before drop: {X_test.shape} "
            )

            if remove_no_variance == True:
                X_train_dropped = X_train.drop(drop_columns_no_variance, axis=1)
                X_test_dropped = X_test.drop(drop_columns_no_variance, axis=1)
                print(
                    f"Shape Train after drop: {X_train_dropped.shape} / Shape Test after drop: {X_test_dropped.shape}\n"
                )
                print(
                    f"Check NaN Train: {X_train_dropped.columns[X_train_dropped.isna().any()].tolist()} "
                )
                print(
                    f"Check inf Train: {X_train_dropped.columns[np.isinf(X_train_dropped).any()].tolist()} "
                )
                return X_train_dropped, X_test_dropped
            else:
                return X_train, X_test


