
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
                 exclude_columns = None) -> None:
        self.PipelineStructure = PipelineStructure
        self.remove_columns_with_no_variance = remove_columns_with_no_variance
        self.exclude_columns = exclude_columns

        self.X_train_prep = None
        self.X_test_prep = None


        self._X_train = None
        self._X_test = None

        self._X_train_transformed = None
        self._X_test_transformed = None

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
    


    def unsupervised_run(self, X_train: pd.DataFrame, clf, dump_model) -> Pipeline:
        print("Running only Input-Data-Pipeline...")
        self.X_train_prep = self.remove_excluded_columns(X_train)

        try:
            X_train_transformed = self.PipelineStructure.fit_transform(self.X_train_prep)
        except Exception as e:
            print(self.X_train_prep.isna().sum(),"\n",e,"\n")
            raise


        X_train_transformed = self.remove_no_variance_columns(
            X_train=X_train_transformed,
            remove_no_variance=self.remove_columns_with_no_variance,
        )

        self._X_train_transformed = X_train_transformed

        clf_no_injection = clf
        clf_no_injection.fit(X_train_transformed)

        if dump_model == True:
            try:
                dump(clf, f"clf_{type(clf).__name__}.joblib")
            except:
                print("Could not dump the model.")

        y_pred_decision_score = clf_no_injection.decision_function(X_train_transformed)
        X_train["AnomalyScore"] = y_pred_decision_score
        scaler = MinMaxScaler()
        X_train[["AnomalyScore"]] = scaler.fit_transform(X_train[["AnomalyScore"]])

        try:
            column_name_mad_total = [
                col for col in X_train_transformed.columns if col.endswith("MAD_Total")
            ][0]
            X_train["MAD_Total"] = X_train_transformed[column_name_mad_total]
            column_name_tukey_total = [
                col for col in X_train_transformed.columns if col.endswith("Tukey_Total")
            ][0]
            X_train["Tukey_Total"] = X_train_transformed[column_name_tukey_total]

            X_train = X_train.sort_values(
                ["AnomalyScore", "MAD_Total", "Tukey_Total"], ascending=False
            )


            first_column = X_train.pop("AnomalyScore")
            X_train.insert(0, "AnomalyScore", first_column)

            
            self._X_train = X_train

            return X_train
        except Exception as e:
            return X_train.sort_values("AnomalyScore", ascending=False)



    def unsupervised_train_test_run(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        clf: pyod.models,
        dump_model: bool,
    ):
        print("Running Pipeline without injected Anomalies...")
        """
        Processes the data through the pipeline and returns the test dataset with corresponding anomaly values.
        
        Purpose:
        - Used for monitoring purposes.
        
        Optional:
        - The model can be temporarily stored using the `dump_model` parameter.
        """
        self.X_train_prep = self.remove_excluded_columns(X_train)
        self.X_test_prep = self.remove_excluded_columns(X_test)

        self.PipelineStructure.fit(self.X_train_prep)

        X_train_transformed = self.PipelineStructure.transform(self.X_train_prep)

        X_test_transformed = self.PipelineStructure.transform(self.X_test_prep)

        X_train_transformed, X_test_transformed = self.remove_no_variance_columns(
            X_train=X_train_transformed,
            X_test=X_test_transformed,
            remove_no_variance=self.remove_columns_with_no_variance,
        )



        self._X_train_transformed = X_train_transformed
        self._X_test_transformed = X_test_transformed

        clf_no_injection = clf
        clf_no_injection.fit(X_train_transformed)

        if dump_model == True:
            try:
                dump(clf, f"clf_{type(clf).__name__}.joblib")
            except:
                print("Could not dump the model.")

        y_pred_decision_score = clf_no_injection.decision_function(X_test_transformed)
        X_test["AnomalyScore"] = y_pred_decision_score
        scaler = MinMaxScaler()
        X_test[["AnomalyScore"]] = scaler.fit_transform(X_test[["AnomalyScore"]])

        try:
            column_name_mad_total = [
                col for col in X_test_transformed.columns if col.endswith("MAD_Total")
            ][0]
            X_test["MAD_Total"] = X_test_transformed[column_name_mad_total]
            column_name_tukey_total = [
                col for col in X_test_transformed.columns if col.endswith("Tukey_Total")
            ][0]
            X_test["Tukey_Total"] = X_test_transformed[column_name_tukey_total]

            X_test = X_test.sort_values(
                ["AnomalyScore", "MAD_Total", "Tukey_Total"], ascending=False
            )

            first_column = X_test.pop("AnomalyScore")
            X_test.insert(0, "AnomalyScore", first_column)

            self._X_test = X_test

            return X_test
        except Exception as e:
            return X_test.sort_values("AnomalyScore", ascending=False)




    def unsupervised_train_test_anomalies_run(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        clf: pyod.models,
        inject_anomalies: pd.DataFrame,
        timeframe_random_state_experiment: str,
    ) -> pd.DataFrame:
        """
        Method to conduct the experiment.

        Steps:
            1. Reassignment of the index for injected anomaly values (end of the test data).
            2. Merging of the test data with the injected anomalies (without their y_true label).
            3. Resetting the index for the test and training data to avoid having two index columns.
            4. Data is processed through the pipeline (structure provided in the method).
            5. Evaluation of the injected anomalies and extraction of quality metrics.
        """
        print("Running Pipeline with injected Anomalies...")
        # self.X_train_prep = self.remove_excluded_columns(X_train)
        # self.X_test_prep = self.remove_excluded_columns(X_test)

        if isinstance(inject_anomalies, pd.DataFrame):
            self.X_test_output_injected = super().start_experiment(
                X_train=X_train,
                X_test=X_test,
                injected_anomalies=inject_anomalies,
                pipeline_structure=self.PipelineStructure,
                clf=clf,
                remove_columns_with_no_variance=self.remove_columns_with_no_variance,
                timeframe_random_state_experiment=timeframe_random_state_experiment,
            )
            # self.feature_importances = super().get_feature_importances()
            return self.X_test_output_injected
        else:
            print("Invalid Injection of Anomalies.")



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





