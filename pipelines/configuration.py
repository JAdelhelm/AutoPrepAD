import numpy as np
import pandas as pd

# import pandas_profiling as pp


from pipelines.preprocessing.nan_handling.NaNColumnCreator import NaNColumnCreator
from pipelines.preprocessing.nan_handling.NaNColumnCreatorTotal import NaNColumnCreatorTotal

from pipelines.preprocessing.statistical.TukeyTransformer import TukeyTransformer
from pipelines.preprocessing.statistical.TukeyTransformerTotal import TukeyTransformerTotal

# from pipelines.preprocessing.statistical.ZTransformerMean import ZTransformerMean
# from pipelines.preprocessing.statistical.ZTransformerMeanTotal import ZTransformerMeanTotal

from pipelines.preprocessing.statistical.MedianAbsolutDeviation import MedianAbsolutDeviation
from pipelines.preprocessing.statistical.MedianAbsolutDeviationTotal import (
    MedianAbsolutDeviationTotal,
)

from pipelines.preprocessing.statistical.SpearmanCheck import SpearmanCorrelationCheck

# from pipelines.preprocessing.engineering.CategoricalPatternsAutoEncoder import CategoricalPatternsAutoEncoder

from pipelines.preprocessing.engineering.CategoricalPatterns import CategoricalPatterns

from pipelines.preprocessing.dummy.XCopySchemaTransformer import XCopySchemaTransformer
from pipelines.preprocessing.dummy.XCopySchemaTransformerNominal import (
    XCopySchemaTransformerNominal,
)
from pipelines.preprocessing.dummy.XCopySchemaTransformerOrdinal import (
    XCopySchemaTransformerOrdinal,
)
from pipelines.preprocessing.dummy.XCopySchemaTransformerPattern import (
    XCopySchemaTransformerPattern,
)

from pipelines.preprocessing.timeseries.DateEncoder import DateEncoder

from pipelines.preprocessing.timeseries.TimeSeriesImputer import TimeSeriesImputer

# from pipelines.preprocessing.estimator.KMedian import KMedianEstimator


from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    OrdinalEncoder,
    RobustScaler,
)
from sklearn.impute import MissingIndicator

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

from category_encoders import BinaryEncoder
from sklearn.base import BaseEstimator, TransformerMixin

import warnings



import itertools
from sklearn import set_config
set_config(transform_output="pandas")


class PipelinesConfiguration():
    """
    The PipelinesConfiguration class represents the class to configure pipelines.

    Methods
    ----------
        - pre_pipeline:
            A transformer that executes the schema and other steps to prepare data for the transformation pipeline.
                - If ``self.exclude_columns`` has entries, the corresponding columns will be dropped.

        - nan_marker_pipeline:\n
            Pipeline that marks columns with NaN-Values.
            Steps: \n
                1. Attempt to infer schema of columns.
                2. Mark corresponding columns with Nan-Values

        - numeric_pipeline:\n
            Pipeline for preprocessing of numerical data.
            Steps:\n
                1. Impute numerical data.
                2. The resulting columns are input to the statistical methods.

        - categorical_pipeline:\n
            Pipeline for preprocessing of categorical data.
            Steps:\n
                1. Impute categorical data.
                2. BinaryEncode categorical data.

        - timeseries_pipeline:\n
            Pipeline for preprocessing of timeseries data.
            Steps:\n
                1. Impute timeseries data.
                2. Extract time units from corresponding timeseries columns.

        - pattern_extraction:\n
            Pipeline to extract patterns from categorical data.
            Includes columnDropperTransformer that drops the columns,
            pattern_extraction is deactivated.
            Steps:\n
                1. Infer Schema of data.
                2. Impute Data.
                3. Extract patterns.
                4. Binary encode the extracted patterns.
            
        
        - nominal_pipeline:\n
            Pipeline for separate preprocessing of nominal data.

        - ordinal_pipeline:\n
            Pipeline for separate preprocessing of ordinal data.        
        
        - get_profiling:\n
            Method to get a profiling of the corresponding input DataFrame.
        


    Parameters
    ----------
    time_column_names : list
        List of certain Time-Columns that should be converted in timestamp data types.

    exclude_columns : list
        List of Columns that should be dropped.

    """
    def __init__(self):
        self.exclude_columns = None
        self.time_column_names = None

    def pre_pipeline(self, time_column_names=None, exclude_columns=None):
        self.exclude_columns = exclude_columns
        self.time_column_names = time_column_names

        original_preprocessor = Pipeline(
            steps=[
                (
                    "Preprocessing",
                    ColumnTransformer(
                        transformers=[
                            (
                                "X",
                                XCopySchemaTransformer(
                                    time_column_names=self.time_column_names,
                                    exclude_columns=self.exclude_columns,
                                    name_transformer="Schema Standard Pipeline",
                                ),
                                make_column_selector(dtype_include=None),
                            ),
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                        verbose=True,
                    ),
                )
            ]
        )

        return original_preprocessor

    def nan_marker_pipeline(self):
        nan_marker_preprocessor = Pipeline(
            steps=[
                (
                    "NanMarker",
                    ColumnTransformer(
                        transformers=[
                            (
                                "nan_marker_columns",
                                Pipeline(
                                    steps=[
                                        (
                                            "X_nan",
                                            XCopySchemaTransformer(
                                                time_column_names=self.time_column_names,
                                                name_transformer="Schema NaNMarker",
                                            ),
                                        ),
                                        # ("nan_marker_total", NaNColumnCreator()),
                                        ("nan_marker", MissingIndicator(features="all")),
                                        # Markiert nur die Werte, die außerhalb des Wertebreichs liegen
                                        # ("tukey_missing", TukeyTransformer(factor=1.5)),   
                                        # ("tukey_total_missing", TukeyTransformerTotal()),                                     
                                        # (
                                        #     "nan_marker_total_nan",
                                        #     NaNColumnCreatorTotal(),
                                        # ),
                                        # (
                                        #     # "OneHotEncoder", OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False),
                                        #     "BinaryEnc",
                                        #     BinaryEncoder(handle_unknown="indicator"),
                                        # ),
                                    ]
                                ),
                                make_column_selector(dtype_include=None),
                            ),
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                        verbose=True,
                    ),
                )
            ]
        )

        return nan_marker_preprocessor

    def numeric_pipeline(self):
        numeric_preprocessor = Pipeline(
            steps=[
                (
                    "Preprocessing_Numerical",
                    ColumnTransformer(
                        transformers=[
                            (
                                "numeric",
                                Pipeline(
                                    steps=[
                                        ("N", IterativeImputer()),
                                        (
                                            "robust_scaler",
                                            RobustScaler(),
                                        ),
                                    ]
                                ),
                                make_column_selector(dtype_include=np.number),
                            ),
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                        verbose=True,
                    ),
                ),
                (
                    "Statistical methods",
                    ColumnTransformer(
                        transformers=[
                            (
                                "stat_tukey",
                                Pipeline(
                                    steps=[
                                        # ("Z-Transformation", StandardScaler().set_output(transform="pandas")),
                                        # (
                                        #     "impute_num",
                                        #     SimpleImputer(strategy="median"),
                                        # ),
                                        ("Tukey_impute", IterativeImputer()),
                                        ("tukey", TukeyTransformer(factor=1.5)),
                                        ("tukey_total", TukeyTransformerTotal())
                                        # ("impute_num", SimpleImputer(strategy="median"))
                                    ]
                                ),
                                make_column_selector(dtype_include=np.number),
                            ),
                            #  ("stat_z",
                            #      Pipeline(
                            #          steps=[
                            #          ("impute_num", SimpleImputer(strategy="median")),
                            #          ("z", ZTransformerMean(threshold=3, z_scores_output=False)),
                            #          # ("iterative_num", IterativeImputer()),
                            #          ("z_total",ZTransformerMeanTotal())
                            #          # ("impute_num", SimpleImputer(strategy="median"))
                            #          ]), make_column_selector(dtype_include=np.number)
                            #  ),
                            (
                                "z_mod",
                                Pipeline(
                                    steps=[
                                        # (
                                        #     "impute_num",
                                        #     SimpleImputer(strategy="median"),
                                        # ),
                                        ("z_mod_impute", IterativeImputer()),
                                        ("z_mod", MedianAbsolutDeviation()),
                                        # ("iterative_num", IterativeImputer()),
                                        ("z_mod_total", MedianAbsolutDeviationTotal())
                                        # ("impute_num", SimpleImputer(strategy="median"))
                                    ]
                                ),
                                make_column_selector(dtype_include=np.number),
                            ),
                            (
                                "pass_cols",
                                Pipeline(
                                    steps=[
                                        (
                                            "_pass_cols_",
                                            "passthrough",
                                        ),
                                    ]
                                ),
                                make_column_selector(dtype_include=np.number),
                            ),
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                        verbose=True,
                    ),
                ),
            ]
        )

        return numeric_preprocessor

    def categorical_pipeline(self):
        return Pipeline(
            steps=[
                (
                    "Preprocessing_Categorical",
                    ColumnTransformer(
                        transformers=[
                            (
                                "categorical",
                                Pipeline(
                                    steps=[
                                        (
                                            "C",
                                            SimpleImputer(strategy="most_frequent"),
                                        ),
                                        (
                                            "BinaryEnc",
                                            BinaryEncoder(handle_unknown="indicator"),
                                        ),
                                    ]
                                ),
                                make_column_selector(dtype_include=np.object_),
                            ),
                        ],
                        remainder="drop",
                        n_jobs=-1,
                        verbose=True,
                    ),
                )
            ]
        )

    def timeseries_pipeline(self):
        timeseries_preprocessor = Pipeline(
            steps=[
                (
                    "Preprocessing_Timeseries",
                    ColumnTransformer(
                        transformers=[
                            (
                                "timeseries",
                                Pipeline(
                                    steps=[
                                        ("T", TimeSeriesImputer(impute_method="ffill")),
                                        # ("num_time_dates", TimeTransformer())
                                        ("num_time_dates", DateEncoder()),
                                        ("robust_scaler", RobustScaler()),
                                        # ("BinaryEnc",BinaryEncoder(handle_unknown="indicator")),
                                    ]
                                ),
                                make_column_selector(
                                    dtype_include=(
                                        np.dtype("datetime64[ns]"),
                                        np.datetime64,
                                        "datetimetz",
                                    )
                                ),
                            ),
                        ],
                        remainder="drop",
                        n_jobs=-1,
                        verbose=True,
                    ),
                )
            ]
        )

        return timeseries_preprocessor

    def pattern_extraction(
        self,
        pattern_recognition_exclude_columns: list = None,
        time_column_names_pattern: list = None,
        deactivate_pattern_recognition: bool = False,
    ):
        if deactivate_pattern_recognition == True:
            return    Pipeline(
            steps=[
                (
                    "PatternRecognition Deactivated",
                    ColumnTransformer(
                        transformers=[
                            (
                                "drop_columns",
                                Pipeline(
                                    steps=[
                                        (
                                            "DropColumns", columnDropperTransformer(),
                                        ),
                                    ]
                                ),
                                make_column_selector(dtype_include=None),
                            ),
                        ],
                        remainder="drop",
                        n_jobs=-1,
                        verbose=True,
                    ),
                )
            ]
        )
  
        elif pattern_recognition_exclude_columns is not None:
            return Pipeline(
                steps=[
                    (
                        "X_pattern",
                        XCopySchemaTransformerPattern(
                            exclude_columns=pattern_recognition_exclude_columns,
                            time_column_names=time_column_names_pattern,
                            name_transformer="Schema PatternExtraction",
                        ),
                    ),
                    (
                        "pattern_processing",
                        ColumnTransformer(
                            transformers=[
                                (
                                    "pattern_processing_inner",
                                    Pipeline(
                                        steps=[
                                            (
                                                "impute_pattern",
                                                SimpleImputer(strategy="most_frequent"),
                                            ),
                                            (
                                                "pattern_extraction",
                                                CategoricalPatterns(),
                                            ),
                                            (
                                                "BinaryEnc",
                                                BinaryEncoder(
                                                    handle_unknown="indicator"
                                                ),
                                            ),
                                        ]
                                    ),
                                    make_column_selector(dtype_include=np.object_),
                                )
                            ],
                            remainder="drop",
                        ),
                    ),
                ]
            )

        else:
            return Pipeline(
                steps=[
                    (
                        "X_pattern",
                        XCopySchemaTransformerPattern(
                            time_column_names=time_column_names_pattern
                        ),
                    ),
                    (
                        "pattern_processing",
                        ColumnTransformer(
                            transformers=[
                                (
                                    "pattern_processing_inner",
                                    Pipeline(
                                        steps=[
                                            (
                                                "impute_pattern",
                                                SimpleImputer(strategy="most_frequent"),
                                            ),
                                            (
                                                "pattern_extraction",
                                                CategoricalPatterns(),
                                            ),
                                            (
                                                "BinaryEnc",
                                                BinaryEncoder(
                                                    handle_unknown="indicator"
                                                ),
                                            ),
                                        ]
                                    ),
                                    make_column_selector(dtype_include=np.object_),
                                )
                            ],
                            remainder="drop",
                        ),
                    ),
                ]
            )

    def nominal_pipeline(
        self,
        nominal_columns: list = None,
        time_column_names: list = None,
    ):
        return Pipeline(
            steps=[
                (
                    "X_nominal",
                    XCopySchemaTransformerNominal(
                        nominal_columns=nominal_columns,
                        time_column_names=time_column_names,
                        name_transformer="Schema Nominal",
                    ),
                ),
                (
                    "nominal_preprocessing",
                    ColumnTransformer(
                        transformers=[
                            (
                                "nominal_processing_inner",
                                Pipeline(
                                    steps=[
                                        (
                                            "impute_nominal",
                                            SimpleImputer(strategy="most_frequent"),
                                        ),
                                        (
                                            "BinaryEnc",
                                            BinaryEncoder(handle_unknown="indicator"),
                                        ),
                                    ]
                                ),
                                make_column_selector(dtype_include=None),
                            )
                        ],
                        remainder="drop",
                    ),
                ),
            ]
        )

    def ordinal_pipeline(
        self,
        ordinal_columns: list = None,
        time_column_names: list = None,
    ):
        """
        Separate Behandlung von Ordinalen Spalten
        """
        return Pipeline(
            steps=[
                (
                    "X_ordinal",
                    XCopySchemaTransformerOrdinal(
                        ordinal_columns=ordinal_columns,
                        time_column_names=time_column_names,
                        name_transformer="Schema Ordinal",
                    ),
                ),
                (
                    "ordinal_preprocessing",
                    ColumnTransformer(
                        transformers=[
                            (
                                "ordinal_processing_inner",
                                Pipeline(
                                    steps=[
                                        (
                                            "impute_ordinal",
                                            SimpleImputer(strategy="most_frequent"),
                                        ),
                                        (
                                            "OrdinalEncoder",
                                            OrdinalEncoder(
                                                handle_unknown="use_encoded_value",
                                                unknown_value=-1,
                                            ),
                                        ),
                                    ]
                                ),
                                make_column_selector(dtype_include=None),
                            )
                        ],
                        remainder="drop",
                    ),
                ),
                (
                    "SpearmanCorrelationCheck",
                    SpearmanCorrelationCheck(),
                ),
            ]
        )


    def get_profiling(self, X: pd.DataFrame, deeper_profiling=False):
        from ydata_profiling import ProfileReport
        if deeper_profiling == False:
            profile = ProfileReport(X, title="Profiling Report")
            profile.to_file("DQ_report.html")
        else:
            profile = ProfileReport(X, title="Profiling Report", explorative=True)
            profile.to_file("DQ_report_deep.html")


class columnDropperTransformer:
    def __init__(self, exclude_columns):
        """
        Es wird eine Rückgabe eines DataFrames von mindestens 1 Spalte erwartet.
        Aus diesem Grund werden die verworfenen Spalten mit 0 befüllt.
        """
        self.exclude_columns = exclude_columns
        self.feature_names = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Falls ein DataFrame nur eine Spalte enthält, dann
        wird diese mit 0'en befüllt und zurückgegeben.
        """

        if self.exclude_columns is None:
            return X

        X_dropped = X.copy()
        columns_to_drop = []
        for exclude_column in self.exclude_columns:
            columns_to_drop.extend([col for col in X.columns if exclude_column in col])

        X_dropped = X.drop(columns_to_drop, axis=1)

        if X_dropped.empty:
            print(
                f"""\nYou drop all columns of dtype: {X.iloc[:,0].dtypes}
-> Only NaN-values of this columns will be marked.\n"""
            )

            X_dropped = X.copy()
            X_dropped[:] = 0

        return X_dropped

    # def get_feature_names(self, input_features=None):
    #     return self.feature_names
    
class columnDropperTransformer():
    def __init__(self):
        pass

    def transform(self, X, y=None):
        return X[[]]

    def fit(self, X, y=None):
        return self
 