
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

import joblib
import os

# -----------------------------
# 1. Synthetic Data Generation
# -----------------------------
def generate_predictive_data(n_samples=3000, n_features=6):
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)
    y = (
        3 * X[:, 0]
        + 5 * X[:, 1] ** 2
        - 2 * np.sin(np.pi * X[:, 2])
        + 4 * X[:, 3]
        + np.random.normal(0, 0.5, n_samples)
    )
    df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(n_features)])
    df["target"] = y
    return df


# -----------------------------
# 2. Data Loading & Saving
# -----------------------------
def create_and_save_dataset():
    df = generate_predictive_data()
    df.to_csv("synthetic_predictive_data.csv", index=False)
    print("Synthetic dataset created and saved.")


# -----------------------------
# 3. Data Loading & Inspection
# -----------------------------
def load_and_inspect_data():
    df = pd.read_csv("synthetic_predictive_data.csv")
    print("\nDataset Head:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nDataset Description:")
    print(df.describe())
    return df


# -----------------------------
# 4. Preprocessing
# -----------------------------
def preprocess_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "scaler.pkl")
    print("Scaler saved as scaler.pkl")

    return X_scaled, y


# -----------------------------
# 5. Model Architectures
# -----------------------------
def build_model(model_type, input_dim):
    model = Sequential()

    if model_type == "shallow":
        model.add(Dense(32, activation="relu", input_shape=(input_dim,)))
    elif model_type == "deep":
        model.add(Dense(64, activation="relu", input_shape=(input_dim,)))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
    elif model_type == "dropout":
        model.add(Dense(64, activation="relu", input_shape=(input_dim,)))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation="relu"))
    elif model_type == "batchnorm":
        model.add(Dense(64, input_shape=(input_dim,)))
        model.add(BatchNormalization())
        model.add(Dense(64, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(32, activation="relu"))

    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


# -----------------------------
# 6. Training
# -----------------------------
def train_models(X_scaled, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models = {
        "Shallow_DNN": build_model("shallow", X_scaled.shape[1]),
        "Deep_DNN": build_model("deep", X_scaled.shape[1]),
        "Dropout_DNN": build_model("dropout", X_scaled.shape[1]),
        "BatchNorm_DNN": build_model("batchnorm", X_scaled.shape[1]),
    }

    history_data = {}

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    for name, model in models.items():
        print(f"Training {name}...")
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1,
        )

        model.save(f"{name}.h5")
        history_data[name] = history.history["val_loss"]

    return models, history_data, X_test, y_test


# -----------------------------
# 7. Evaluation
# -----------------------------
def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        predictions = model.predict(X_test).flatten()
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"\n{name}")
        print("MSE:", mse)
        print("R2 Score:", r2)


# -----------------------------
# 8. Visualization
# -----------------------------
def plot_validation_loss(history_data):
    plt.figure(figsize=(10, 6))
    for name, losses in history_data.items():
        plt.plot(losses, label=name)

    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Comparison of DNN Architectures")
    plt.legend()
    plt.tight_layout()
    plt.savefig("validation_loss_comparison.png")
    plt.show()

    print("Validation loss plot saved as validation_loss_comparison.png")


# -----------------------------
# 9. Inference Function
# -----------------------------
def predict_sample(input_features):
    scaler = joblib.load("scaler.pkl")
    model = load_model("Deep_DNN.h5")

    input_features = np.array(input_features).reshape(1, -1)
    input_scaled = scaler.transform(input_features)

    prediction = model.predict(input_scaled)
    return prediction[0][0]


# -----------------------------
# 10. Main Execution
# -----------------------------
def main():
    if not os.path.exists("synthetic_predictive_data.csv"):
        create_and_save_dataset()

    df = load_and_inspect_data()
    X_scaled, y = preprocess_data(df)

    models, history_data, X_test, y_test = train_models(X_scaled, y)
    evaluate_models(models, X_test, y_test)
    plot_validation_loss(history_data)

    sample_input = [0.5, 0.6, 0.2, 0.8, 0.4, 0.7]
    prediction = predict_sample(sample_input)
    print("\nSample Prediction:", prediction)


if __name__ == "__main__":
    main()
