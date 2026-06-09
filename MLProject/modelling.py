import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ===== KONFIGURASI MLFLOW =====
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Diabetes-Classification")

# ===== LOAD DATA =====
def load_data():
    base_path = 'diabetes_preprocessing'
    X_train = pd.read_csv(f'{base_path}/X_train.csv')
    X_test  = pd.read_csv(f'{base_path}/X_test.csv')
    y_train = pd.read_csv(f'{base_path}/y_train.csv').squeeze()
    y_test  = pd.read_csv(f'{base_path}/y_test.csv').squeeze()
    print(f"[INFO] Data loaded. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# ===== SIMPAN CONFUSION MATRIX =====
def save_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("[INFO] Confusion matrix disimpan.")

# ===== TRAINING =====
def train():
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name="RandomForest-Baseline"):
        # Aktifkan autolog
        mlflow.sklearn.autolog()

        # Training model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)

        print(f"\n📊 Hasil Evaluasi:")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1 Score : {f1:.4f}")

        # Simpan dan log confusion matrix
        save_confusion_matrix(y_test, y_pred)
        mlflow.log_artifact('confusion_matrix.png')

    print("\n✅ Training selesai! Cek MLflow UI di http://127.0.0.1:5000")

if __name__ == "__main__":
    train()