# UAADP - Unsupervised Automated Anomaly Detection Pipeline
<a href="https://html-preview.github.io/?url=https://github.com/JAdelhelm/Automated-Anomaly-Detection-Preprocessing-Pipeline/blob/main/visualization/PipelineDQ.html" target="_blank">Structure of Pipeline (Click)</a>



This pipeline is designed for unsupervised applications where labels are not available. It can be utilized in various fields, for example:

- **Cybersecurity**: Detect potential cyber attacks.
- **Data Quality**: Identify broken or inconsistent data.
- **Marketing**: Discover customers with high revenue and margins.
- **Finance**: Spot unusual transactions that may indicate fraud.
- **Energy**: Identify anomalies in energy consumption patterns that could indicate inefficiencies or issues with the power grid.



## Abstract View - Project
![alt text](./images/project.png)



## Usage 


```python
import numpy as np
import pandas as pd

from pipelines.control import UAADP
from pipelines.defaults import initialize_autoencoder, initialize_autoencoder_modified
from pipelines.defaults import dummy_data
pd.set_option("display.max_columns", None)
# from pyod.models.iforest import IForest
# from pyod.models.lof import LOF
from pyod.models.pca import PCA



if __name__ == "__main__":
    df_data = pd.read_csv("./temperature_USA.csv")

    clf_pca = PCA()

    anomaly_detection_pipeline = UAADP(
        exclude_columns=[],
        deactivate_pattern_recognition=True,
        exclude_columns_no_variance=True,
        mark_anomalies_pct_data=0.01
    )

    anomaly_detection_pipeline.fit(
        X_train=df_data,
        clf=clf_pca,
        dump_model=False,
    )
    
    X_output = anomaly_detection_pipeline.predict(X_test=df_data)


    X_output.to_csv("temperatures_anomalies.csv", index=False)

    # anomaly_detection_pipeline.visualize_pipeline_structure_html()

```


#### Data: https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities
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

## Pipeline - Logic
![alt text](./images/decision_rules.png)



---

## Feel free to contribute 🙂

### Reference
- https://www.researchgate.net/publication/379640146_Detektion_von_Anomalien_in_der_Datenqualitatskontrolle_mittels_unuberwachter_Ansatze (German Thesis)

### Further Information
- Pipeline can also be used to add extra columns (feature engineering)
    - Adds AnomalyLabel to rows
    - Marks univariate outliers
- I used sklearn's Pipeline and Transformer concept to create this preprocessing pipeline
    - Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    - Transformer: https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
