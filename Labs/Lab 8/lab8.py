import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, ParameterGrid, ParameterSampler
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

max_features = 10000
maxlen = 200
embedding_dim = 64

# INS. Load original IMDB data
(x_train_full, y_train_full), (x_test_full, y_test_full) = imdb.load_data(num_words=max_features)

# INS. Combine train and test for custom small balanced subset
X_all = np.concatenate([x_train_full, x_test_full], axis=0)
y_all = np.concatenate([y_train_full, y_test_full], axis=0)

# INS. First choose a small subset
subset_size = 6000
indices = np.arange(len(X_all))
np.random.shuffle(indices)
indices = indices[:subset_size]
X_small = [X_all[i] for i in indices]
y_small = y_all[indices]

# INS. Split into train / validation / test
X_temp, X_test, y_temp, y_test = train_test_split(
    X_small, y_small, test_size=1000, random_state=SEED, stratify=y_small
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=1000, random_state=SEED, stratify=y_temp
)

# INS. Pad sequences
X_train = pad_sequences(X_train, maxlen=maxlen)
X_val = pad_sequences(X_val, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

print(X_train.shape, X_val.shape, X_test.shape)


def build_model(learning_rate=1e-3, dropout=0.2, units=64):
    model = Sequential([
        Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=maxlen),
        LSTM(units),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def evaluate_config(config, verbose=0):
    model = build_model(
        learning_rate=config["learning_rate"],
        dropout=config["dropout"],
        units=config["units"]
    )
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True
    )
    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=64,
        verbose=verbose,
        callbacks=[early_stop]
    )
    runtime = time.time() - start
    best_val_acc = max(history.history["val_accuracy"])
    tf.keras.backend.clear_session()
    return best_val_acc, runtime


# INS. Grid search
grid = {
    "learning_rate": [1e-3, 1e-2],
    "dropout": [0.2, 0.4],
    "units": [32, 64]
}

grid_results = []
for config in ParameterGrid(grid):
    score, runtime = evaluate_config(config, verbose=0)
    grid_results.append({
        "method": "grid",
        **config,
        "val_accuracy": score,
        "runtime_sec": runtime
    })
    print(config, score)

# INS. Random search
search_space_random = {
    "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    "units": [32, 48, 64, 96, 128]
}

random_results = []
samples = list(ParameterSampler(search_space_random, n_iter=8, random_state=SEED))
for config in samples:
    score, runtime = evaluate_config(config, verbose=0)
    random_results.append({
        "method": "random",
        **config,
        "val_accuracy": score,
        "runtime_sec": runtime
    })
    print(config, score)

# INS. Bayesian search
bayes_results = []
space = {
    "learning_rate": hp.loguniform("learning_rate", np.log(1e-4), np.log(1e-2)),
    "dropout": hp.uniform("dropout", 0.1, 0.5),
    "units": hp.choice("units", [32, 48, 64, 96, 128])
}


def objective(params):
    config = {
        "learning_rate": float(params["learning_rate"]),
        "dropout": float(params["dropout"]),
        "units": int(params["units"])
    }
    score, runtime = evaluate_config(config, verbose=0)
    bayes_results.append({
        "method": "bayesian",
        **config,
        "val_accuracy": score,
        "runtime_sec": runtime
    })
    return {"loss": -score, "status": STATUS_OK}


trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=10,
    trials=trials,
    rstate=np.random.RandomState(SEED)  # changed from np.random.default_rng(SEED) to accomodate older version of hyperopt used
)
print(best)

# INS. Collect and compare results
all_results = grid_results + random_results + bayes_results
best_grid = max(grid_results, key=lambda x: x["val_accuracy"])
best_random = max(random_results, key=lambda x: x["val_accuracy"])
best_bayes = max(bayes_results, key=lambda x: x["val_accuracy"])

print("Best Grid:", best_grid)
print("Best Random:", best_random)
print("Best Bayesian:", best_bayes)

# INS. Train final model with best overall config
best_overall = max(all_results, key=lambda x: x["val_accuracy"])
final_model = build_model(
    learning_rate=best_overall["learning_rate"],
    dropout=best_overall["dropout"],
    units=best_overall["units"]
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=2,
    restore_best_weights=True
)

final_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=64,
    verbose=1,
    callbacks=[early_stop]
)

test_loss, test_acc = final_model.evaluate(X_test, y_test, verbose=0)
print("Final test accuracy:", test_acc)