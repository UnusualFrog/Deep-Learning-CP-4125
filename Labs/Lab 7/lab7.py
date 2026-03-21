import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force CPU
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Construct MLP model
def build_mlp(n_hidden=1, units=32, dropout=0.2, lr=1e-3):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(9,)))

    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(units, activation="relu"))
        model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",  # restore loss here
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

def main():
    df = pd.read_csv("plc_sensor_quality.csv")

    # Split the dataset between training features and target
    X = df.drop(columns=["defective"]).values
    y = df["defective"].values

    # 30/70 train/test split with stratification to get train and test set
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # 50/50 validation/test split with stratification to split test set further into test and validation sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Earlying stop if AUC does not improve after 8 epochs
    # Focus on improving (rather than decreasing) auc and restore best model weights upon early stopping, rather than use current ones
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=8,
        restore_best_weights=True
    )

    # Construct pipeline for baseline model
    baseline = Pipeline(steps=[
        ("scaler", StandardScaler()), # scale values
        ("clf", KerasClassifier(        # set classifier model to MLP
            model=build_mlp,
            epochs=40,
            batch_size=16,
            verbose=0,
            loss="binary_crossentropy"
        ))
    ])

    # Fit baseline model to training data with early stopping
    baseline.fit(
        X_train, y_train,
        clf__validation_data=(X_val, y_val),
        clf__callbacks=[early_stop]
    )

    # Make predition, convert from probabilities (0.42, 0.91) to binary based on threshold (0, 1)
    y_pred = (baseline.predict(X_test) > 0.5).astype(int)

    # Display baseline results 
    print(classification_report(y_test, y_pred, digits=3))
    print(confusion_matrix(y_test, y_pred))

    # Build gridsearch pipeline
    grid_pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", KerasClassifier(model=build_mlp, verbose=0, loss="binary_crossentropy"))
    ])

    # Build parameter grid 
    param_grid = {
        "clf__model__n_hidden": [1, 2],
        "clf__model__units": [16, 32, 64],
        "clf__model__dropout": [0.0, 0.2, 0.4],
        "clf__model__lr": [1e-4, 1e-3, 1e-2],
        "clf__epochs": [30, 50],
        "clf__batch_size": [16, 32],
    }

    # Build Gridsearch model with ROC_AUC scoring and 3 fold cv
    gs = GridSearchCV(
        grid_pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=1
    )

    # Fit gridsearch model to training data with early stop
    gs.fit(
        X_train, y_train,
        clf__validation_data=(X_val, y_val),
        clf__callbacks=[early_stop]
    )

    # Show best parameter combination found by the model
    print(gs.best_params_)
    print(gs.best_score_)

    # Build random search pipeline
    rand_pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", KerasClassifier(model=build_mlp, verbose=0, loss="binary_crossentropy"))
    ])

    # Build parameter grid
    param_dist = {
        "clf__model__n_hidden": [1, 2, 3],
        "clf__model__units": [8, 16, 32, 64, 96],
        "clf__model__dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "clf__model__lr": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        "clf__epochs": [25, 35, 50, 70],
        "clf__batch_size": [8, 16, 32, 64],
    }

    # Build random search model
    rs = RandomizedSearchCV(
        rand_pipe,
        param_distributions=param_dist,
        n_iter=18,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=1
    )

    # Fit random search model to training data with early stop
    rs.fit(
        X_train, y_train,
        clf__validation_data=(X_val, y_val),
        clf__callbacks=[early_stop]
    )

    print(rs.best_params_)
    print(rs.best_score_)


if __name__ == "__main__":
    main()


# Results
#              precision    recall  f1-score   support

#            0      0.000     0.000     0.000         8
#            1      0.600     0.800     0.686        15

#     accuracy                          0.522        23
#    macro avg      0.300     0.400     0.343        23
# weighted avg      0.391     0.522     0.447        23

# [[ 0  8]
#  [ 3 12]]
# {'clf__batch_size': 32, 'clf__epochs': 50, 'clf__model__dropout': 0.4, 'clf__model__lr': 0.001, 'clf__model__n_hidden': 2, 'clf__model__units': 16}
# 0.7334042768825378
# {'clf__model__units': 96, 'clf__model__n_hidden': 1, 'clf__model__lr': 0.001, 'clf__model__dropout': 0.4, 'clf__epochs': 25, 'clf__batch_size': 64}
# 0.6384074862335732