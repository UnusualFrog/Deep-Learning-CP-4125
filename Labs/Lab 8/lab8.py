# NOTE: Python Version 3.11.15 had to be used do to version issues with hyperopt
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
# Randomize indexes of samples
indices = np.arange(len(X_all))
np.random.shuffle(indices)
# Pick a subset of samples using randomized indicies
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

# Build 4 Layer LSTM model
def build_model(learning_rate=1e-3, dropout=0.2, units=64):
    model = Sequential([
        Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=maxlen),
        LSTM(units),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])
    # Use Adam optimizer with crossentropy loss and accuracy
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Build, train and evaluate model
def evaluate_config(config, verbose=0):
    # Build Model
    model = build_model(
        learning_rate=config["learning_rate"],
        dropout=config["dropout"],
        units=config["units"]
    )
    # Set early stopping based on validation accuracy with tolerance for 2 runs without improvement
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True
    )
    # Time the model training
    start = time.time()
    # Train model on training data over 5 epochs with early stopping
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

# Run Grid Search with 8 possible combinations of parameters to maximize validation accuracy
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

# Total Grid Search Time
total_grid_time = sum(r["runtime_sec"] for r in grid_results)
print(f"Grid search total runtime: {total_grid_time:.1f}s")

# INS. Random search
search_space_random = {
    "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    "units": [32, 48, 64, 96, 128]
}

# Run random search over 8 iterations
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

# Total Random Search Time
total_random_time = sum(r["runtime_sec"] for r in random_results)
print(f"Random search total runtime: {total_random_time:.1f}s")

# INS. Bayesian search
bayes_results = []
space = {
    "learning_rate": hp.loguniform("learning_rate", np.log(1e-4), np.log(1e-2)),
    "dropout": hp.uniform("dropout", 0.1, 0.5),
    "units": hp.choice("units", [32, 48, 64, 96, 128])
}

# Build, train and evaluate bayesian search
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

# Run Bayesian search with a maximum of 10 iterations
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

# Total Bayesian search time
total_bayes_time = sum(r["runtime_sec"] for r in bayes_results)
print(f"Bayesian search total runtime: {total_bayes_time:.1f}s")

# INS. Collect and compare results
all_results = grid_results + random_results + bayes_results
best_grid = max(grid_results, key=lambda x: x["val_accuracy"])
best_random = max(random_results, key=lambda x: x["val_accuracy"])
best_bayes = max(bayes_results, key=lambda x: x["val_accuracy"])

print("Best Grid:", best_grid)
print("Best Random:", best_random)
print("Best Bayesian:", best_bayes)

print(f"Grid evaluations: {len(grid_results)}")
print(f"Random evaluations: {len(random_results)}")
print(f"Bayesian evaluations: {len(bayes_results)}")

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

# Train best model over 5 epochs with early stopping
final_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=64,
    verbose=1,
    callbacks=[early_stop]
)

# Capture best model test loss and test accuracy
test_loss, test_acc = final_model.evaluate(X_test, y_test, verbose=0)
print("Final test accuracy:", test_acc)

# Plot validation accuracy vs trial number
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# Get results of all three methods
methods = [
    ("Grid Search", grid_results),
    ("Random Search", random_results),
    ("Bayesian Search", bayes_results)
]

# For each method, plot trial number, current accuracy, and best overall accuracy
for ax, (method_name, results) in zip(axes, methods):
    accuracies = [r["val_accuracy"] for r in results]
    trial_numbers = list(range(1, len(accuracies) + 1))
    best_so_far = [max(accuracies[:i+1]) for i in range(len(accuracies))]

    ax.plot(trial_numbers, accuracies, marker='o', linestyle='--',
            color='steelblue', label='Val Accuracy', alpha=0.7)
    ax.plot(trial_numbers, best_so_far, marker='', linestyle='-',
            color='crimson', label='Best So Far')
    ax.set_title(method_name)
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Validation Accuracy")
    ax.set_xticks(trial_numbers)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle("Validation Accuracy vs Trial Number", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("val_accuracy_vs_trial.png", dpi=150)