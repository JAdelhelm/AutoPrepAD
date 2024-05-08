# %%
import numpy as np
import pandas as pd

from dataqualitypipeline import DQPipeline
from dataqualitypipeline import initialize_autoencoder, initialize_autoencoder_modified
from dataqualitypipeline import dummy_data
pd.set_option("display.max_columns", None)
from pyod.models.iforest import IForest


if __name__ == "__main__":
    df_data = pd.read_csv("./HOWTO/players_20.csv")

    # import torch
    clf_if = IForest(n_jobs=-1)
    clf_ae = initialize_autoencoder_modified(epochs=120)

    dq_pipe = DQPipeline(
        nominal_columns=["short_name","long_name","nationality","club","nation_position","player_tags","work_rate"],
        exclude_columns=["player_url"],
        time_column_names=["dob"],
        # deactivate_pattern_recognition=False,
        remove_columns_with_no_variance=True,
    )

    X_output = dq_pipe.run_pipeline(
        X_train=df_data.iloc[:,0:35],
        clf=clf_ae,
        dump_model=False,
    )



###############################################################################################
#### Alternative if you have train, test data and anomalies you want to inject and evaluate ###
###############################################################################################

    # train_data, test_data, injected_anomaly_data = dummy_data()
    # clf_if = IForest(n_jobs=-1)
    # clf_ae = initialize_autoencoder_modified()

    # dq_pipe = DQPipeline(
        # nominal_columns=["COLTestCAT1"],
        # ordinal_columns=["COLNUM", "COLNUM2"],
        # exclude_columns=["locationId","evseid","uuid"],
        # time_column_names=["timestamp"],
        # pattern_recognition_exclude_columns=["locationId", "timestamp","oldTimestamp"],

        # deactivate_pattern_recognition=False,
        # remove_columns_with_no_variance=True,
    # )

    # X_output = dq_pipe.run_pipeline(
    #     X_train=train_data,
    #     X_test=test_data, 
    #     clf=clf_if,
    #     dump_model=False,
    #     inject_anomalies=injected_anomaly_data,
    # )
    # print(train_data)
    # print(test_data)
    # print(injected_anomaly_data)

    # X_output.to_csv("output_testrun.csv", index=False)

    # dq_pipe.visualize_pipeline_structure_html()
    # dq_pipe.get_profiling(train_data)

    







# %%


