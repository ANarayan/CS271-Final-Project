from sklearn import preprocessing
import os
import pandas as pd
import ast
import wfdb
import numpy as np
import tensorflow as tf


def load_data(path_to_data):
    sampling_rate = 100
    duration = 10
    nleads = 12
    val_fold = 9
    test_fold = 10
    num_classes = 5

    data_dir = path_to_data

    def load_raw_data(df):
        if sampling_rate == 100:
            data = [
                wfdb.rdsamp(str(os.path.join(data_dir, f)))
                for f in df.filename_lr
            ]
        else:
            data = [
                wfdb.rdsamp(str(os.path.join(data_dir / f)))
                for f in df.filename_hr
            ]
        data = np.array([signal for signal, meta in data])
        return data

    # load and convert annotation data
    Y = pd.read_csv(
        os.path.join(data_dir, "ptbxl_database.csv"),
        index_col="ecg_id",
    )
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(
        os.path.join(data_dir, "scp_statements.csv"), index_col=0
    )
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        lhmax = -1
        superclass = ""
        for key in y_dic.keys():
            if key in agg_df.index and y_dic[key] > lhmax:
                lhmax = y_dic[key]
                superclass = agg_df.loc[key].diagnostic_class
        return superclass

    # Apply diagnostic superclass
    Y["diagnostic_superclass"] = Y.scp_codes.apply(aggregate_diagnostic)
    Y = Y[Y.diagnostic_superclass != ""]

    # Load raw signal data
    X = load_raw_data(Y)

    # Convert labels to multiclass targets
    le = preprocessing.LabelEncoder()
    le.fit(Y.diagnostic_superclass)
    Y["diagnostic_superclass"] = le.transform(Y.diagnostic_superclass)

    # if split == "train":
    X_train = input_values = X[
        np.where((Y.strat_fold != val_fold) & (Y.strat_fold != test_fold))
    ]
    y_train = Y[
        ((Y.strat_fold != val_fold) & (Y.strat_fold != test_fold))
    ].diagnostic_superclass.values

    X_val = X[np.where(Y.strat_fold == val_fold)]
    y_val = Y[(Y.strat_fold == val_fold)].diagnostic_superclass.values

    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[(Y.strat_fold == test_fold)].diagnostic_superclass.values

    def preprocess_signals(X_train, X_validation, X_test):
        # Standardize data such that mean 0 and variance 1
        ss = preprocessing.StandardScaler()
        ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))

        return (
            apply_standardizer(X_train, ss),
            apply_standardizer(X_validation, ss),
            apply_standardizer(X_test, ss),
        )

    def apply_standardizer(X, ss):
        X_tmp = []
        for x in X:
            x_shape = x.shape
            X_tmp.append(
                ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape)
            )
        X_tmp = np.array(X_tmp)
        return X_tmp

    X_train, X_val, X_test = preprocess_signals(X_train, X_val, X_test)

    dataset_train = (
        tf.convert_to_tensor(X_train, dtype=tf.float32),
        tf.convert_to_tensor(y_train),
    )

    dataset_val = (
        tf.convert_to_tensor(X_val, dtype=tf.float32),
        tf.convert_to_tensor(y_val),
    )
    dataset_test = (
        tf.convert_to_tensor(X_test, dtype=tf.float32),
        tf.convert_to_tensor(y_test),
    )

    return dataset_train, dataset_val, dataset_test
