# Automated (Unsupervised) Anomaly Detection Preprocessing Pipeline
---

### I used sklearn's Pipeline and Transformer concept to create this preprocessing pipeline
- Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
- Transformer: https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html

---

## How to use the pipeline

### <a href="https://html-preview.github.io/?url=https://github.com/JAdelhelm/Automated-Anomaly-Detection-Preprocessing-Pipeline/blob/main/visualization/PipelineDQ.html" target="_blank">Structure of Pipeline (Click)</a>

```python
import numpy as np
import pandas as pd
from dataqualitypipeline import initialize_autoencoder, initialize_autoencoder_modified
from pyod.models.iforest import IForest
from pyod.models.lof import LOF

df_data = pd.read_csv("./HOWTO/players_20.csv")
clf_lof = LOF(n_jobs=-1)

# Init Preprocessing Pipeline
from dataqualitypipeline import DQPipeline
dq_pipe = DQPipeline(
    nominal_columns=["player_tags","preferred_foot",
                     "work_rate","team_position","loaned_from"],

    exclude_columns=["player_url","body_type","short_name", "long_name", 
                     "team_jersey_number","joined","contract_valid_until",
                     "real_face","nation_position","player_positions","nationality","club"],

    time_column_names=["dob"],
    deactivate_pattern_recognition=True,
    remove_columns_with_no_variance=True,
)


# Run Preprocessing-Pipeline (Named dq_pipe)
X_output = dq_pipe.run_pipeline(
    X_train=df_data.iloc[:,0:37],
# Add Anomaly Detection Model (clf)
    clf=clf_lof,
    dump_model=False,
)

X_output.head(40)
```

- Checkout the ``how_to.ipynb`` Notebook to use this pipeline.
    - There is an  example with only train data (unsupervised)

---


## Highlights ⭐

### 📌 BinaryEncoder instead of OneHotEncoder for nominal columns / *Big Data and Performance*
   Newest research shows similar results for encoding nominal columns with significantly fewer dimensions.
   - (John T. Hancock and Taghi M. Khoshgoftaar. "Survey on categorical data for neural networks." In: Journal of Big Data 7.1 (2020), pp. 1–41.)
       - Tables 2, 4
   - (Diogo Seca and João Mendes-Moreira. "Benchmark of Encoders of Nominal Features for Regression." In: World Conference on Information Systems and Technologies. 2021, pp. 146–155.)
       - P. 151


### 📌 Implementation of univariate methods / *Detection of univariate anomalies*
   Both methods (MOD Z-Value and Tukey Method) are resilient against outliers, ensuring that the position measurement will not be biased. They also support multivariate anomaly detection algorithms in identifying univariate anomalies.

### 📌 Transformation of time series data and standardization of data with RobustScaler / *Normalization for better prediction results*

### 📌 Labeling of NaN values in an extra column instead of removing them / *No loss of information*

---


## Abstract View - Project
![alt text](./images/project.png)

---

## Decision rules of the pipeline
![alt text](./images/decision_rules.png)



---


## Feel free to contribute 🙂

### Reference
- https://www.researchgate.net/publication/379640146_Detektion_von_Anomalien_in_der_Datenqualitatskontrolle_mittels_unuberwachter_Ansatze (German Thesis)
