"""
==============================================================
modelling.py  (MLflow Project entry point)
Digunakan oleh: MLProject/MLproject → entry_point: main
Dipanggil oleh: GitHub Actions CI workflow

Cara menjalankan manual:
    mlflow run MLProject/ -P n_estimators=100 -P max_depth=5

Argumen:
    --n_estimators      Jumlah pohon di Random Forest  (default: 100)
    --max_depth         Kedalaman maksimal pohon        (default: 5)
    --min_samples_split Min sampel untuk split node     (default: 2)
    --random_state      Seed random                     (default: 42)
==============================================================
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)


# ============================================================
# ARGPARSE
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Iris Classifier")
    parser.add_argument("--n_estimators",      type=int, default=100)
    parser.add_argument("--max_depth",         type=int, default=5)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--random_state",      type=int, default=42)
    return parser.parse_args()


# ============================================================
# UTILITAS
# ============================================================
def load_data():
    """Memuat data dari folder data/ (relatif terhadap lokasi script)."""
    base = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base, "..", "data", "train.csv")
    test_path  = os.path.join(base, "..", "data", "test.csv")
    
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test  = test_df.drop(columns=["target"])
    y_test  = test_df["target"]
    
    print(f"✅ Data | Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def save_confusion_matrix_plot(y_true, y_pred):
    """Simpan plot confusion matrix ke /tmp."""
    class_names = ["setosa", "versicolor", "virginica"]
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    path = "/tmp/confusion_matrix.png"
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    
    print("=" * 60)
    print("   MLflow Project — Training Entry Point")
    print(f"   n_estimators     : {args.n_estimators}")
    print(f"   max_depth        : {args.max_depth}")
    print(f"   min_samples_split: {args.min_samples_split}")
    print(f"   random_state     : {args.random_state}")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    with mlflow.start_run():
        
        # ---- Training ----
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=args.random_state,
        )
        model.fit(X_train, y_train)
        
        # ---- Evaluasi ----
        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)
        
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        roc  = roc_auc_score(y_test, y_pred_prob, multi_class="ovr", average="weighted")
        
        # ---- Manual Logging ----
        # Params
        mlflow.log_param("n_estimators",      args.n_estimators)
        mlflow.log_param("max_depth",          args.max_depth)
        mlflow.log_param("min_samples_split",  args.min_samples_split)
        mlflow.log_param("random_state",       args.random_state)
        
        # Metrics
        mlflow.log_metric("accuracy",           acc)
        mlflow.log_metric("precision_weighted", prec)
        mlflow.log_metric("recall_weighted",    rec)
        mlflow.log_metric("f1_weighted",        f1)
        mlflow.log_metric("roc_auc_weighted",   roc)
        
        # Artefak: confusion matrix
        cm_path = save_confusion_matrix_plot(y_test, y_pred)
        mlflow.log_artifact(cm_path, artifact_path="plots")
        
        # Artefak: classification report
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_path = "/tmp/classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        mlflow.log_artifact(report_path, artifact_path="reports")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"\n📊 Accuracy  : {acc:.4f}")
        print(f"   Precision : {prec:.4f}")
        print(f"   Recall    : {rec:.4f}")
        print(f"   F1-Score  : {f1:.4f}")
        print(f"   ROC-AUC   : {roc:.4f}")
        
        print(f"\n✅ Run selesai | Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
