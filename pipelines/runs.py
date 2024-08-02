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
    """
    A class to run machine learning pipelines with anomaly detection.

    Attributes
    ----------
    PipelineStructure : object
        The structure of the pipeline to be fitted.
    remove_columns_with_no_variance : bool, optional
        Whether to remove columns with no variance (default is False).
    exclude_columns : list, optional
        List of columns to exclude from the pipeline (default is None).
    mark_anomalies_pct_data : float, optional
        Percentage of data to mark as anomalies (default is 0.1).
    
    Methods
    -------
    fit_pipeline(X_train, clf=None, dump_model=False)
        Fits the pipeline and trains the anomaly detection model.
    predict_pipeline(X_test)
        Predicts anomalies based on the fitted pipeline.
    remove_excluded_columns(df)
        Removes specified columns from the dataframe.
    remove_no_variance_columns(X_train, X_test=None, remove_no_variance=False, name="Train")
        Removes columns with no variance from the dataframe.
    check_and_rename_statistical_outliers()
        Checks for statistical outliers and renames columns accordingly.
    """

    def __init__(self,
                 PipelineStructure: object,
                 remove_columns_with_no_variance=False,
                 exclude_columns=None,
                 mark_anomalies_pct_data=0.1) -> None:
        """
        Constructs all the necessary attributes for the PipelineRuns object.

        Parameters
        ----------
        PipelineStructure : object
            The structure of the pipeline to be fitted.
        remove_columns_with_no_variance : bool, optional
            Whether to remove columns with no variance (default is False).
        exclude_columns : list, optional
            List of columns to exclude from the pipeline (default is None).
        mark_anomalies_pct_data : float, optional
            Percentage of data to mark as anomalies (default is 0.1).
        """
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

        self.X_preprocessed = None

        self.no_variance_columns = None

    @property
    def X_train_transformed(self):
        """Returns the transformed training data."""
        return self._X_train_transformed

    @property
    def X_test_transformed(self):
        """Returns the transformed test data."""
        return self._X_test_transformed
    
    @property
    def X_train(self):
        """Returns the original training data."""
        return self._X_train
    
    @property
    def X_test(self):
        """Returns the original test data."""
        return self._X_test

    @X_test.setter
    def X_test(self, value):
        """Sets the original test data."""
        self._X_test = value
    

    def fit_pipeline(
            self, 
            X_train: pd.DataFrame,
            clf: pyod.models = None,
            dump_model: bool = False,
    ):
        """
        Fits the pipeline and trains the anomaly detection model.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training data.
        clf : pyod.models, optional
            The anomaly detection model (default is None).
        dump_model : bool, optional
            Whether to dump the model to a file (default is False).
        """
        print("Fitting Pipeline and train anomaly detection model...")
        self.X_train_prep = self.remove_excluded_columns(X_train)

        try:
            self.fitted_pipeline = self.PipelineStructure.fit(self.X_train_prep)
        except TypeError as e:
            message = (
                "Please check if your data contains datetime columns.\n"
                "If it does, ensure they are specified using the 'datetime_columns' parameter.\n"
                "when initializing the AutoPrepAD object.\n"
            )

            raise DatetimeException(f"{e}\n\n\n{message}")


        except Exception as e:
            print(self.X_train_prep.isna().sum(), "\n", e, "\n")
            raise

        self.X_train_transformed_model = self.fitted_pipeline.transform(self.X_train_prep)

        self.X_train_transformed_model = self.remove_no_variance_columns(
            X_train=self.X_train_transformed_model,
            remove_no_variance=self.remove_columns_with_no_variance,
            name="Train"
        )

        self.fitted_model = clf.fit(self.X_train_transformed_model)

        if dump_model:
            try:
                dump(self.fitted_model, f"clf_{type(self.fitted_model).__name__}.joblib")
            except:
                print("Could not dump the model.")

    def predict_pipeline(
            self,
            X_test: pd.DataFrame
    ):
        """
        Predicts anomalies based on the fitted pipeline.

        Parameters
        ----------
        X_test : pd.DataFrame
            The test data.

        Returns
        -------
        pd.DataFrame
            The test data with anomaly scores and labels.
        """
        print("Prediction of Anomalies, based on fitted Pipeline...")
        self.X_test_prep = self.remove_excluded_columns(X_test)

        self.X_test_transformed_model = self.fitted_pipeline.transform(self.X_test_prep)

        self.X_test_transformed_model = self.remove_no_variance_columns(
            X_train=self.X_test_transformed_model,
            remove_no_variance=self.remove_columns_with_no_variance,
            name="Test"
        )

        y_pred_decision_score = self.fitted_model.decision_function(self.X_test_transformed_model)
        X_test["AnomalyScore"] = y_pred_decision_score
        scaler = MinMaxScaler()
        X_test[["AnomalyScore"]] = scaler.fit_transform(X_test[["AnomalyScore"]])

        self.X_test = X_test

        try:
            self.check_and_rename_statistical_outliers()

            first_column = self.X_test.pop("AnomalyScore")
            self.X_test.insert(0, "AnomalyScore", first_column)

            threshold_AD = np.percentile(first_column, 100 * (1 - self.mark_anomalies_pct_data))
            y_pred_array = (first_column > threshold_AD).astype(int)

            self.X_test["AnomalyLabel"] = y_pred_array

            return self.X_test
        
        except Exception as e:
            threshold_AD = np.percentile(first_column, 100 * (1 - self.mark_anomalies_pct_data))
            y_pred_array = (first_column > threshold_AD).astype(int)
            self.X_test["AnomalyLabel"] = y_pred_array
            return self.X_test.sort_values("AnomalyScore", ascending=False)
        


    def preprocess_pipeline(
            self, 
            df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a preprocessing pipeline to the input DataFrame.

        This method performs several preprocessing steps on the given DataFrame, 
        including removing excluded columns, fitting the specified pipeline 
        structure, transforming the data, and removing columns with no variance. 
        It ensures that the data is properly transformed and ready for further 
        analysis or modeling.

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame that needs to be preprocessed.

        Returns:
        --------
        pd.DataFrame
            The preprocessed DataFrame with all necessary transformations applied.
        """

        print("Transforming Input Dataframe with Pipeline...")
        self.X_preprocessed =  self.remove_excluded_columns(df)

        try:
            self.fitted_pipeline = self.PipelineStructure.fit(self.X_preprocessed)
        except TypeError as e:
            message = (
                "Please check if your data contains datetime columns.\n"
                "If it does, ensure they are specified using the 'datetime_columns' parameter.\n"
                "when initializing the AutoPrepAD object.\n"
            )

            raise DatetimeException(f"{e}\n\n\n{message}")


        except Exception as e:
            print(self.X_preprocessed.isna().sum(), "\n", e, "\n")
            raise

        self.X_preprocessed = self.fitted_pipeline.transform(self.X_preprocessed)

        self.X_preprocessed = self.remove_no_variance_columns(
            X_train=self.X_preprocessed,
            remove_no_variance=self.remove_columns_with_no_variance,
            name="Preprocessed"
        )

        return self.X_preprocessed










    def remove_excluded_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes specified columns from the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            The input dataframe.

        Returns
        -------
        pd.DataFrame
            The dataframe with specified columns removed.
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
        self, X_train, X_test=None, remove_no_variance=False, name="Train"
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Removes columns with no variance from the dataframe.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training data.
        X_test : pd.DataFrame, optional
            The test data (default is None).
        remove_no_variance : bool, optional
            Whether to remove columns with no variance (default is False).
        name : str, optional
            The name of the dataset (default is "Train").

        Returns
        -------
        pd.DataFrame or tuple of pd.DataFrame
            The modified training data, and optionally the modified test data.
        """
        if X_test is None:
            X_train_cols_no_variance = X_train.loc[:, X_train.std() == 0.0].columns
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

            print(f"Remove columns with zero variance: {remove_no_variance}")

            print(f"Shape {name} before drop: {X_train.shape}")

            if remove_no_variance:
                X_train_dropped = X_train.drop(drop_columns_no_variance, axis=1)

                print(f"Shape {name} after drop: {X_train_dropped.shape}\n")
                print(f"Check NaN {name}: {X_train_dropped.columns[X_train_dropped.isna().any()].tolist()}")
                print(f"Check inf {name}: {X_train_dropped.columns[np.isinf(X_train_dropped).any()].tolist()}")
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

            print(f"Remove columns with zero variance: {remove_no_variance}")

            print(f"Shape Train before drop: {X_train.shape} / Shape Test before drop: {X_test.shape}")

            if remove_no_variance:
                X_train_dropped = X_train.drop(drop_columns_no_variance, axis=1)
                X_test_dropped = X_test.drop(drop_columns_no_variance, axis=1)
                print(f"Shape Train after drop: {X_train_dropped.shape} / Shape Test after drop: {X_test_dropped.shape}\n")
                print(f"Check NaN Train: {X_train_dropped.columns[X_train_dropped.isna().any()].tolist()}")
                print(f"Check inf Train: {X_train_dropped.columns[np.isinf(X_train_dropped).any()].tolist()}")
                return X_train_dropped, X_test_dropped
            else:
                return X_train, X_test

    def check_and_rename_statistical_outliers(self):
        """
        Checks for statistical outliers and renames columns accordingly.
        """
        try:
            column_name_mad_total = [
                col for col in self.X_test_transformed_model.columns if col.endswith("MAD_Total")
            ]
            if column_name_mad_total:
                self.X_test["MAD_Total"] = self.X_test_transformed_model[column_name_mad_total[0]]
            else:
                print("No Median Absolute Deviation outliers found in data....")

            column_name_tukey_total = [
                col for col in self.X_test_transformed_model.columns if col.endswith("Tukey_Total")
            ]
            if column_name_tukey_total:
                self.X_test["Tukey_Total"] = self.X_test_transformed_model[column_name_tukey_total[0]]
            else:
                print("No Tukey outliers found in data....")

            sort_columns = ["AnomalyScore"]
            if "MAD_Total" in self.X_test.columns:
                sort_columns.append("MAD_Total")
            if "Tukey_Total" in self.X_test.columns:
                sort_columns.append("Tukey_Total")

            self.X_test = self.X_test.sort_values(sort_columns, ascending=False)

        except Exception as e:
            print(f"An error occurred while checking and renaming statistical outliers: {e}")

class DatetimeException(Exception):
    """
    Exception raised for errors in the datetime handling in the pipeline.

    Attributes
    ----------
    message : str
        Explanation of the error.
    """
    pass
