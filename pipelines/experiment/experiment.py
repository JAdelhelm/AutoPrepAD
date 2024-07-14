# %%
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn import set_config
import pandas as pd
import numpy as np

# import tensorflow as tf
# import torch
# from pyod.models.auto_encoder_torch import AutoEncoder
from sklearn.preprocessing import MinMaxScaler

import pyod
import os



import time

from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)


class Experiment:
    """
    The Experiment class represents is used for an experimentals setup to evaluate outliers.

    Methods
    ----------
        - collect_feature_importances:
            - Attempt to extract the feature importances from trained model.

        - create_new_index:\n
            Steps:\n
                1. Extracts the index length of ``X_1``
                2. Returns the index starting from the end of the ``X_1`` index length.

        - start_experiment:
            Conducts the experimental procedure with the following steps:

                1. Reassigns the index for the injected anomaly values (at the end of the test data).
                2. Merges the test data with the injected anomalies (excluding their label ``y_true``).
                3. Resets the index for both the test and training data to avoid having two index columns.
                4. The data passes through the pipeline (with its structure provided by the method).
                            - The pipeline is fitted using the training data, and then the transformation is applied to both the training and test data.
                5. Evaluates the injected anomalies and retrieves the quality metrics.

        - start_evaluation_experiment:
            Steps:\n
                1. Train the model using the training data and record the training time.
                2. Use the model to predict on the test data.
                3. Evaluate the injected anomalies. Perform an inner-join based on the index.
                4. Return the labeled anomalies with a threshold of 10% along with the anomaly scores.
        
        - check_variance:\n
            Removes columns without variance, except for 'MAD_Total' and 'Tukey_Total',
            which represent statistical methods.

            Relevant only for the experiment evaluation. Can be omitted in monitoring.




    Parameters
    ----------
    time_column_names : list
        List of certain Time-Columns that should be converted in timestamp data types.


    Attributes
    ----------
    X_train_transformed: pd.DataFrame
        This attribute returns the transformed Train-Dataframe

    X_test_anomalies_transformed
        This attribute returns the transformed Test-Dataframe

    X_test_anomalies_transformed
        This attribute returns all indices of the injected anomalies

    feature_importances
        This attribute returns the feature importances of the model (if possible)

    results_experiment
        This attribute returns the results of the experiment
    """

    def __init__(self) -> None:
        self._index_of_anomalies = None
        self.X_test_with_anomalies = None

        self.pipeline_time_total = None
        self.training_time_total = None
        self.testing_time_total = None

        self._X_train_transformed = None
        self._X_test_anomalies_transformed = None

        self.y_pred = None
        self.y_pred_score = None

        self.injected_anomalies = None

        self._results_metrics = None
        self._feature_importances = None


    @property
    def index_of_anomalies(self):
        if not hasattr(self, "_index_of_anomalies"):
            self._index_of_anomalies = None
        return self._index_of_anomalies.to_numpy().reshape(-1)

    @property
    def feature_importances(self):
        if not hasattr(self, "_feature_importances"):
            self._feature_importances = None
        return self._feature_importances

    @property
    def X_train_transformed(self):
        if not hasattr(self, "_X_train_transformed"):
            self._X_train_transformed = None
        return self._X_train_transformed

    @X_train_transformed.setter
    def X_train_transformed(self, new_value):
        self._X_train_transformed = new_value

    @property
    def X_test_anomalies_transformed(self):
        if not hasattr(self, "_X_test_anomalies_transformed"):
            self._X_test_anomalies_transformed = None
        return self._X_test_anomalies_transformed

    @X_test_anomalies_transformed.setter
    def X_test_anomalies_transformed(self, new_value):
        self._X_test_anomalies_transformed = new_value

    @property
    def results_experiment(self):
        if not hasattr(self, "_results_metrics"):
            print("No anomalies were injected.")
            self._results_metrics = None
        return self._results_metrics

    def collect_feature_importances(self, clf, X_transformed):
        try:
            clf_fitted_feature_importances = clf.feature_importances_
            clf_fitted_feature_importances = pd.DataFrame(
                {"importance": clf_fitted_feature_importances},
                index=X_transformed.columns,
            )
            self._feature_importances = clf_fitted_feature_importances.sort_values(
                "importance", ascending=False
            )
        except (
            Exception
        ) as e:  
            print(f"Couldn't get feature importances. Error: {e}")

    def create_new_index(self, X_1, X_2):
        index_len_start = len(X_1.index)
        index_len_end = len(X_1.index) + len(X_2.index)

        new_index_starts_at_end_of_index_len_start = [
            val for val in range(index_len_start, index_len_end)
        ]

        return new_index_starts_at_end_of_index_len_start

    def start_experiment(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        injected_anomalies: pd.DataFrame,
        pipeline_structure: Pipeline,
        clf: pyod.models,
        remove_columns_with_no_variance: bool = False,
        timeframe_random_state_experiment: str = "42"
    ) -> pd.DataFrame:
        self.timeframe_random_state_experiment = timeframe_random_state_experiment

        self.remove_columns_with_no_variance = remove_columns_with_no_variance

        X_train_experiment = X_train.copy()
        X_test_experiment = X_test.copy()
        self.injected_anomalies = injected_anomalies

        self.injected_anomalies.index = self.create_new_index(
            X_1=X_test_experiment, X_2=self.injected_anomalies
        )
        self._index_of_anomalies = self.injected_anomalies.index

        X_test_experiment_with_anomalies = pd.concat(
            [X_test_experiment, self.injected_anomalies.drop("y_true", axis=1)], axis=0
        )
        self._X_test_with_anomalies = X_test_experiment_with_anomalies

        pipeline_time_start = time.time()

        pipeline_structure.fit(X_train_experiment)

        self._X_train_transformed = pipeline_structure.transform(X_train_experiment)
        self._X_test_anomalies_transformed = pipeline_structure.transform(
            X_test_experiment_with_anomalies
        )

        pipeline_time_end = time.time()

        self.pipeline_time_total = round(
            (abs(pipeline_time_start - pipeline_time_end) / 60), 2
        )
        print(f"Pipeline-Transformation took: {self.pipeline_time_total} mins...")

        try:
            column_name_mad_total = [
                col for col in self._X_train_transformed if col.endswith("MAD_Total")
            ][0]
            self._X_train_transformed["MAD_Total"] = self._X_train_transformed[
                column_name_mad_total
            ]

            column_name_mad_total = [
                col for col in self._X_train_transformed if col.endswith("Tukey_Total")
            ][0]
            self._X_train_transformed["Tukey_Total"] = self._X_train_transformed[
                column_name_mad_total
            ]

            column_name_tukey_total = [
                col
                for col in self._X_test_anomalies_transformed.columns
                if col.endswith("MAD_Total")
            ][0]
            self._X_test_anomalies_transformed[
                "MAD_Total"
            ] = self._X_test_anomalies_transformed[column_name_tukey_total]

            column_name_tukey_total = [
                col
                for col in self._X_test_anomalies_transformed.columns
                if col.endswith("Tukey_Total")
            ][0]
            self._X_test_anomalies_transformed[
                "Tukey_Total"
            ] = self._X_test_anomalies_transformed[column_name_tukey_total]
        except Exception as e:
            print(
                """Could not get the statistical columns from transformed data.\n
Please consider to check you experiment class and your naming of statistical columns.\n
-> Probably no numerical columns in DataFrame. \n"""
            )

        self.y_pred, self.y_pred_score = self.start_evaluation_experiment(
            X_train_transformed=self._X_train_transformed,
            X_test_anomalies_transformed=self._X_test_anomalies_transformed,
            clf=clf,
        )

        self._X_test_with_anomalies["AnomalyLabel"] = self.y_pred["y_pred"]
        self._X_test_with_anomalies["AnomalyScore"] = self.y_pred_score
        # Scale AnomalyScore in Range 0-1 with MinMaxScaler
        scaler = MinMaxScaler()
        self._X_test_with_anomalies[["AnomalyScore"]] = scaler.fit_transform(
            self._X_test_with_anomalies[["AnomalyScore"]]
        )

        try:
            self._X_test_with_anomalies[
                "MAD_Total"
            ] = self._X_test_anomalies_transformed["MAD_Total"]
            self._X_test_with_anomalies[
                "Tukey_Total"
            ] = self._X_test_anomalies_transformed["Tukey_Total"]

            first_column = self._X_test_with_anomalies.pop("AnomalyScore")
            self._X_test_with_anomalies.insert(0, "AnomalyScore", first_column)

            return self._X_test_with_anomalies.sort_values(
                ["AnomalyScore", "MAD_Total", "Tukey_Total"], ascending=False
            )
        except Exception as e:
            return self._X_test_with_anomalies.sort_values(
                "AnomalyScore", ascending=False
            )

    def start_evaluation_experiment(
        self, X_train_transformed: pd.DataFrame = None, X_test_anomalies_transformed: pd.DataFrame = None, clf: pyod.models = None
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Steps:\n
            1. Get parameters of model for later evaluation, e.g. name of model.
            2. Check Arrays X_train and X_test for variance in columns + parse to attributes
            3. Fitting Model by train data + try to get the feature importances
            4. Predict test data by model
            5. Calculate different y_pred with different thresholds, e.g. y_pred_array_THRESHOLD
                - Labels the top X percent, based on the decision score from test data
            6. Save metrics for standard evaluation (10% Threshhold) with and without statistical methods
            7. Calculate precision and recall for all thresholds + save them in an extra csv (no statistical methods)
        """
        try:
            params = clf.get_params()
            print(f"Paramters of Anomaly Detection Model - {type(clf).__name__}: ")
            print(params, "\n")
        except:
            pass

        X_train_transformed, X_test_anomalies_transformed = self.check_variance(
            X_train=X_train_transformed,
            X_test=X_test_anomalies_transformed,
            remove_cols=self.remove_columns_with_no_variance,
        )

        self._X_train_transformed = X_train_transformed
        self._X_test_anomalies_transformed = X_test_anomalies_transformed

        start_train_time = time.time()
        clf_fitted = clf.fit(X_train_transformed)
        end_train_time = time.time()
        self.training_time_total = abs(start_train_time - end_train_time)

        print(
            f"Model-Training took: {round((abs(self.training_time_total) / 60), 2)} mins..."
        )

        self.collect_feature_importances(
            clf=clf_fitted, X_transformed=X_train_transformed
        )

        start_test_time = time.time()

        y_pred_decision_score_array = clf_fitted.decision_function(
            X_test_anomalies_transformed
        )


        end_test_time = time.time()
        self.testing_time_total = abs(start_test_time - end_test_time)
        print(f"\nModel-Prediction took: {round((abs(self.testing_time_total) / 60), 2)} mins...")

        if (X_test_anomalies_transformed.select_dtypes(include=np.number).shape[0] == 0): print("\nNo numerical columns in DataFrame...");


        # # Calculate different decision scores of different thresholds
        # # 1%
        # threshold_AD_1 = np.percentile(y_pred_decision_score_array, 100 * (1 - 0.01))
        # y_pred_array_1 = (y_pred_decision_score_array > threshold_AD_1).astype(int)
        # # 5%
        # threshold_AD_5 = np.percentile(y_pred_decision_score_array, 100 * (1 - 0.05))
        # y_pred_array_5 = (y_pred_decision_score_array > threshold_AD_5).astype(int)
        # 10%
        threshold_AD_10 = np.percentile(y_pred_decision_score_array, 100 * (1 - 0.1))
        y_pred_array_10 = (y_pred_decision_score_array > threshold_AD_10).astype(int)
        # # 25%       
        # threshold_AD_25 = np.percentile(y_pred_decision_score_array, 100 * (1 - 0.25))
        # y_pred_array_25 = (y_pred_decision_score_array > threshold_AD_25).astype(int)
        # # 50%  
        # threshold_AD_50 = np.percentile(y_pred_decision_score_array, 100 * (1 - 0.5))
        # y_pred_array_50 = (y_pred_decision_score_array > threshold_AD_50).astype(int)
        # # 75%  
        # threshold_AD_75 = np.percentile(y_pred_decision_score_array, 100 * (1 - 0.75))
        # y_pred_array_75 = (y_pred_decision_score_array > threshold_AD_75).astype(int)
        # # 100% 
        # # threshold_AD_100 = np.percentile(y_pred_decision_score_array, 100)
        # y_pred_array_100 = np.ones(len(y_pred_decision_score_array), dtype=int)


        unique, counts = np.unique(y_pred_array_10, return_counts=True)
        unique_counts = dict(zip(unique, counts))




        y_pred_statistical_methods_10, y_pred_no_statistical_methods_10 = self.calculate_y_pred_of_threshold_10(y_pred_array_10, X_test_anomalies_transformed=X_test_anomalies_transformed)
        try:
            self.save_metrics_standard(clf_fitted=clf_fitted, y_pred=y_pred_statistical_methods_10, X_test_anomalies_transformed=X_test_anomalies_transformed, mode_pandas="w",include_stat_methods=True)

            try: self.save_metrics_standard(clf_fitted=clf_fitted, y_pred=y_pred_no_statistical_methods_10, X_test_anomalies_transformed=X_test_anomalies_transformed, mode_pandas="a", include_stat_methods=False)
            except: print("\nCould not calculate metrics without statistical methods.")
        except Exception as e:
            print(f"\nCould not save metrics: {e}")


        # Calculation of multiple Thresholds
        for threshold in np.arange(0.01,1.01,0.01):
            if threshold == 0.01:
                val_threshold =  np.percentile(y_pred_decision_score_array, 100 * (1 - threshold))
                val_pred = (y_pred_decision_score_array > val_threshold).astype(int)
                self.save_metrics_pr_curve_no_statistical_methods(val_pred, threshold, X_test_anomalies_transformed=X_test_anomalies_transformed, clf_fitted=clf_fitted, mode_pandas="w")
            elif threshold == 100:
                val_pred = np.ones(len(y_pred_decision_score_array), dtype=int)
                self.save_metrics_pr_curve_no_statistical_methods(val_pred, threshold, X_test_anomalies_transformed=X_test_anomalies_transformed, clf_fitted=clf_fitted, mode_pandas="a")
            else:
                val_threshold =  np.percentile(y_pred_decision_score_array, 100 * (1 - threshold))
                val_pred = (y_pred_decision_score_array > val_threshold).astype(int)
                self.save_metrics_pr_curve_no_statistical_methods(val_pred, threshold, X_test_anomalies_transformed=X_test_anomalies_transformed, clf_fitted=clf_fitted, mode_pandas="a")


        # for val, threshold in zip([y_pred_array_1, y_pred_array_5, y_pred_array_10, y_pred_array_25, y_pred_array_50,y_pred_array_75, y_pred_array_100], ["1%","5%","10%","25%","50%","75%","100%"]):
        #     if threshold == "1%":
        #         self.save_metrics_pr_curve_no_statistical_methods(val, threshold, X_test_anomalies_transformed=X_test_anomalies_transformed, clf_fitted=clf_fitted, mode_pandas="w")
        #     else:
        #         self.save_metrics_pr_curve_no_statistical_methods(val, threshold, X_test_anomalies_transformed=X_test_anomalies_transformed, clf_fitted=clf_fitted, mode_pandas="a")


        y_pred_decision_score = pd.DataFrame(
            y_pred_decision_score_array,
            index=X_test_anomalies_transformed.index,
            columns=["y_pred_decision_score"],
        )

        return y_pred_statistical_methods_10, y_pred_decision_score
    

    def calculate_y_pred_of_threshold_10(self, y_pred_array: list = [], X_test_anomalies_transformed: pd.DataFrame() = None):
        """
        Merges y_pred with statistical methods and without statistical methods.
        This used for the standard setting with 10% contamination.
        """
        y_pred_statistical_methods = pd.DataFrame()
        y_pred_no_statistical_methods = pd.DataFrame()

        y_pred_no_statistical_methods = pd.DataFrame({"y_pred": y_pred_array}, index=X_test_anomalies_transformed.index)
        try:
            y_pred_statistical_methods = pd.DataFrame(
                        {
                            "y_pred": y_pred_array,
                            "MAD_Total": X_test_anomalies_transformed["MAD_Total"],
                            "Tukey_Total": X_test_anomalies_transformed["Tukey_Total"],
                        },
                        index=X_test_anomalies_transformed.index)
            y_pred_statistical_methods["y_pred"] = np.where(
                        (y_pred_statistical_methods["MAD_Total"] == 1) | (y_pred_statistical_methods["Tukey_Total"] == 1),
                        1,
                        y_pred_statistical_methods["y_pred"],
                    )
            print("\nStatistical methods will be merged with AnomalyLabel.")

            return y_pred_statistical_methods, y_pred_no_statistical_methods

        except Exception as e:
            print(f"\nCould not include statistical methods: {e}")
            y_pred_no_statistical_methods = pd.DataFrame({"y_pred": y_pred_array}, index=X_test_anomalies_transformed.index)

            return y_pred_statistical_methods, y_pred_no_statistical_methods
        


    
    def save_metrics_standard(self,clf_fitted, y_pred, X_test_anomalies_transformed,mode_pandas, include_stat_methods):
        model_name = type(clf_fitted).__name__
        y_true = self.injected_anomalies[["y_true"]]

        df_evaluation = y_true.join(y_pred, how="inner")

        precision = round(
            precision_score(df_evaluation["y_true"], df_evaluation["y_pred"]), 3
        )
        recall = round(
            recall_score(df_evaluation["y_true"], df_evaluation["y_pred"]), 3
        )
        f1 = round(f1_score(df_evaluation["y_true"], df_evaluation["y_pred"]), 3)

        results = {
            "Modell": [model_name],
            "Precision": [precision],
            "Recall": [recall],
            "F1-Score": [f1],
            "Statistische Methoden": [include_stat_methods],
            "Anzahl Anomalien injiziert": [len(y_true)],
            "Trainingszeit in min": [
                round((abs(self.training_time_total) / 60), 2)
            ],
            "Trainingszeit in sec pro Zeile": [
                self.training_time_total / X_test_anomalies_transformed.shape[0]
            ],
            "Testzeit in min": [round((abs(self.testing_time_total) / 60), 2)],
            "Testzeit in sec pro Zeile": [
                self.testing_time_total / X_test_anomalies_transformed.shape[0]
            ],
            "Anzahl Zeilen/Spalten Testdaten": [X_test_anomalies_transformed.shape],
            "Anzahl Zeilen/Spalten Trainingsdaten": [self._X_train_transformed.shape],
            "Pipeline Zeit für Transformation": [self.pipeline_time_total],
        }
        results_df = pd.DataFrame(results)
        from pathlib import Path
        Path("./RESULTS").mkdir(parents=True, exist_ok=True)
        if mode_pandas == "a":
            results_df.to_csv(f"./RESULTS/{model_name}_{self.timeframe_random_state_experiment}_injected.csv", index=False, mode=mode_pandas, header=False)
        else:
            results_df.to_csv(f"./RESULTS/{model_name}_{self.timeframe_random_state_experiment}_injected.csv", index=False, mode=mode_pandas)

        self._results_metrics = pd.read_csv(f"./RESULTS/{model_name}_{self.timeframe_random_state_experiment}_injected.csv")

    def save_metrics_pr_curve_no_statistical_methods(self, y_pred_array: list = [], threshold: int = None, X_test_anomalies_transformed: pd.DataFrame() = None, clf_fitted: pyod.models = None, mode_pandas: str="a"):
        """
        Get the predictions for injected anomalies at different thresholds.
        Evaluation will be without statistical methods.
        """
        model_name = type(clf_fitted).__name__
        y_pred_no_statistical_methods = pd.DataFrame({"y_pred": y_pred_array}, index=X_test_anomalies_transformed.index)
        y_true = self.injected_anomalies[["y_true"]]

        df_evaluation = y_true.join(y_pred_no_statistical_methods, how="inner")

        precision = round(
            precision_score(df_evaluation["y_true"], df_evaluation["y_pred"]), 3
        )
        recall = round(
            recall_score(df_evaluation["y_true"], df_evaluation["y_pred"]), 3
        )

        f1 = round(
            f1_score(df_evaluation["y_true"], df_evaluation["y_pred"]) , 3
        )

        results = {
        "Modell": [model_name],
        "Precision": [precision],
        "Recall": [recall],
        "F1": [f1],
        "Threshold": [str(threshold)+"%"],
    }
        results_df = pd.DataFrame(results)
        from pathlib import Path
        Path(f"./RESULTS/PR_CURVE_{model_name}").mkdir(parents=True, exist_ok=True)
        if mode_pandas == "a":
            results_df.to_csv(f"./RESULTS/PR_CURVE_{model_name}/thresholds_{model_name}_{self.timeframe_random_state_experiment}_injected.csv", index=False, mode=mode_pandas, header=False)
        else:
            results_df.to_csv(f"./RESULTS/PR_CURVE_{model_name}/thresholds_{model_name}_{self.timeframe_random_state_experiment}_injected.csv", index=False, mode=mode_pandas)


    def check_variance(
        self, X_train, X_test, remove_cols=False
    ) -> (pd.DataFrame, pd.DataFrame):
        X_train_cols_no_variance = X_train.loc[:, X_train.std() == 0].columns
        X_test_cols_no_variance = X_test.loc[:, X_test.std() == 0].columns

        X_train_cols_no_variance = X_train_cols_no_variance.tolist()
        X_test_cols_no_variance = X_test_cols_no_variance.tolist()

        if "MAD_Total" in X_train_cols_no_variance:
            X_train_cols_no_variance.remove("MAD_Total")
        if "Tukey_Total" in X_train_cols_no_variance:
            X_train_cols_no_variance.remove("Tukey_Total")

        if "MAD_Total" in X_test_cols_no_variance:
            X_test_cols_no_variance.remove("MAD_Total")
        if "Tukey_Total" in X_test_cols_no_variance:
            X_test_cols_no_variance.remove("Tukey_Total")

        print("\nNo Variance in follow Train Columns: ", X_train_cols_no_variance, "\n")
        print("No Variance in follow Test Columns: ", X_test_cols_no_variance)

        drop_columns_no_variance = X_train_cols_no_variance + X_test_cols_no_variance

        print(f"Remove columns with zero variance: {remove_cols}")

        print(
            f"\nShape Train before drop:{X_train.shape} / Shape Test before drop: {X_test.shape} "
        )
        print(
            f"Check NaN Train: {X_train.isna().any().sum()} / Check Nan Test: {X_test.isna().any().sum()}"
        )
        print(
            f"Check inf Train: {np.isinf(X_train).any().sum()} / Check inf Test: {np.isinf(X_test).any().sum()}\n"
        )

        if remove_cols == True:
            X_train_dropped = X_train.drop(drop_columns_no_variance, axis=1)
            X_test_dropped = X_test.drop(drop_columns_no_variance, axis=1)
            print(
                f"Shape Train after drop: {X_train_dropped.shape} / Shape Test after drop: {X_test_dropped.shape}\n"
            )
            print(
                f"Check NaN Train: {X_train_dropped.isna().any().sum()} / Check Nan Test: {X_test_dropped.isna().any().sum()}\n"
            )
            print(
                f"Check inf Train: {np.isinf(X_train_dropped).any().sum()} / Check inf Test: {np.isinf(X_test_dropped).any().sum()}\n"
            )
            return X_train_dropped, X_test_dropped
        else:
            return X_train, X_test
