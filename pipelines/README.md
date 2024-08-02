## Key Methods (``control.py``)

### `fit(self, X_train: pd.DataFrame, clf: pyod.models = None, dump_model: bool = False) -> Pipeline`
Fits a pipeline to the provided training data using the specified anomaly detection algorithm.

**Parameters:**
- `X_train` (pd.DataFrame): A DataFrame containing training data that the pipeline and anomaly detection algorithm will be fitted to.
- `clf` (pyod.models, optional): An instance of an anomaly detection model from the pyod library to be used for fitting. If not provided, a default model will be used.
- `dump_model` (bool, optional): A flag indicating whether the fitted model should be saved to disk. Default is False.

**Returns:**
- `Pipeline`: The fitted pipeline.

### `predict(self, X_test: pd.DataFrame) -> pd.DataFrame`
Predict anomalies in the test data using the fitted pipeline.

This method takes a DataFrame, processes it through the fitted pipeline, and returns the original DataFrame enriched with additional columns that represent the anomaly scores.

**Parameters:**
- `X_test` (pd.DataFrame): A DataFrame potentially containing anomalies to be predicted.

**Returns:**
- `pd.DataFrame`: The original DataFrame enriched with columns representing anomaly scores.

**Examples:**
```
>>> model = AutoPrepAD()
>>> model.fit(X_train, clf)
>>> X_test_with_scores = model.predict(X_test)
>>> print(X_test_with_scores.head())
```

### `preprocess(self, df: pd.DataFrame) -> pd.DataFrame`
Preprocesses the given DataFrame.

This method applies a preprocessing pipeline to the input DataFrame, which may include operations such as encoding columns and other transformations necessary for the dataset to be in a suitable form for further analysis or modeling.

**Parameters:**
- `df` (pd.DataFrame): The input DataFrame that needs to be preprocessed.

**Returns:**
- `pd.DataFrame`: The preprocessed DataFrame with all necessary transformations applied.