import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import time
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Project root directory
ROOT = Path(__file__).resolve().parents[2]

# Dataset directories
DATA_DIR = ROOT / "data"

NOVPN_DIR = DATA_DIR / "novpn"
VPN_DIR = DATA_DIR / "vpn"


def load_data(context):
    if context == "novpn":
        data_dir = NOVPN_DIR
    elif context == "vpn":
        data_dir = VPN_DIR
    else:
        raise ValueError("Context must be 'novpn' or 'vpn'")

    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")

    X_train_scaled = pd.read_csv(data_dir / "X_train_scaled.csv")
    X_test_scaled = pd.read_csv(data_dir / "X_test_scaled.csv")

    y_train = pd.read_csv(data_dir / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(data_dir / "y_test.csv").iloc[:, 0]

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test

def create_models():
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=7,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        ),

        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    }

    return models

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
def train_and_evaluate(context):
    print(f"\n===== {context.upper()} =====")

    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = load_data(context)

    models = create_models()

    metrics = []
    per_class_results = []

    for model_name, model in models.items():

        # Logistic Regression uses scaled data
        if model_name == "logistic_regression":
            Xtr = X_train_scaled
            Xte = X_test_scaled
        else:
            Xtr = X_train
            Xte = X_test

        print(f"\nTraining {model_name}...")

        start_time = time.perf_counter()

        model.fit(Xtr, y_train)

        training_time = time.perf_counter() - start_time

        y_pred = model.predict(Xte)

        # Overall metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        metrics.append({
            "context": context,
            "model": model_name,
            "training_time_seconds": training_time,
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1
        })

        # Per-class recall
        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        for class_id in sorted(y_test.unique()):
            class_name = str(class_id)

            per_class_results.append({
                "context": context,
                "model": model_name,
                "class": class_id,
                "recall": report[class_name]["recall"]
            })

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        cm_df = pd.DataFrame(cm)

        cm_path = CONFUSION_DIR / f"{context}_{model_name}.csv"
        cm_df.to_csv(cm_path, index=False)

        # Save model
        model_dir = MODELS_DIR / context
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Macro F1 : {f1:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Saved model: {model_path}")

    return metrics, per_class_results

def train_and_evaluate(context):
    print(f"\n===== {context.upper()} =====")

    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = load_data(context)

    models = create_models()

    metrics = []
    per_class_results = []

    for model_name, model in models.items():

        # Logistic Regression uses scaled data
        if model_name == "logistic_regression":
            Xtr = X_train_scaled
            Xte = X_test_scaled
        else:
            Xtr = X_train
            Xte = X_test

        print(f"\nTraining {model_name}...")

        start_time = time.perf_counter()

        model.fit(Xtr, y_train)

        training_time = time.perf_counter() - start_time

        y_pred = model.predict(Xte)

        # Overall metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        metrics.append({
            "context": context,
            "model": model_name,
            "training_time_seconds": training_time,
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1
        })

        # Per-class recall
        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        for class_id in sorted(y_test.unique()):
            class_name = str(class_id)

            per_class_results.append({
                "context": context,
                "model": model_name,
                "class": class_id,
                "recall": report[class_name]["recall"]
            })

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        cm_df = pd.DataFrame(cm)

        cm_path = CONFUSION_DIR / f"{context}_{model_name}.csv"
        cm_df.to_csv(cm_path, index=False)

        # Save model
        model_dir = MODELS_DIR / context
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Macro F1 : {f1:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Saved model: {model_path}")

    return metrics, per_class_results
if __name__ == "__main__":

    all_metrics = []
    all_per_class_results = []

    for context in ["novpn", "vpn"]:

        metrics, per_class_results = train_and_evaluate(context)

        all_metrics.extend(metrics)
        all_per_class_results.extend(per_class_results)

    # Save overall metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(
        RESULTS_DIR / "baseline_metrics.csv",
        index=False
    )

    # Save per-class recall
    per_class_df = pd.DataFrame(all_per_class_results)
    per_class_df.to_csv(
        RESULTS_DIR / "per_class_recall.csv",
        index=False
    )

    print("\n===================================")
    print("BASELINE TRAINING COMPLETE")
    print("===================================")
    print(f"Metrics saved to: {RESULTS_DIR / 'baseline_metrics.csv'}")
    print(f"Per-class recall saved to: {RESULTS_DIR / 'per_class_recall.csv'}")