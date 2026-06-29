"""
==============================================================
automate_Wildan.py
File otomatisasi preprocessing data untuk proyek MLOps
Kriteria 1 - Skilled: Preprocessing otomatis

Cara menjalankan:
    python automate_Wildan.py

Output:
    data/train.csv - Data latih yang sudah diproses
    data/test.csv  - Data uji yang sudah diproses
==============================================================
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# ============================================================
# KONSTANTA
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_DIR = "data"


# ============================================================
# FUNGSI-FUNGSI PREPROCESSING
# ============================================================

def load_data() -> pd.DataFrame:
    """
    Memuat dataset Iris dari scikit-learn.
    
    Returns:
        pd.DataFrame: DataFrame berisi fitur dan label target.
    """
    print("[1/5] Memuat dataset...")
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    print(f"      ✅ Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Menangani nilai yang hilang (missing values) dengan SimpleImputer.

    Args:
        df (pd.DataFrame): DataFrame input.
        strategy (str): Strategi imputasi ('mean', 'median', 'most_frequent').

    Returns:
        pd.DataFrame: DataFrame tanpa missing values.
    """
    print(f"[2/5] Menangani missing values (strategy='{strategy}')...")
    
    feature_cols = [c for c in df.columns if c != 'target']
    
    imputer = SimpleImputer(strategy=strategy)
    df[feature_cols] = imputer.fit_transform(df[feature_cols])
    
    total_missing = df.isnull().sum().sum()
    print(f"      ✅ Missing values setelah imputasi: {total_missing}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris duplikat dari DataFrame.

    Args:
        df (pd.DataFrame): DataFrame input.

    Returns:
        pd.DataFrame: DataFrame tanpa duplikat.
    """
    print("[3/5] Menghapus duplikat...")
    
    before = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    after = df.shape[0]
    
    print(f"      ✅ Duplikat dihapus: {before - after} baris (sisa: {after} baris)")
    return df


def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Menormalkan fitur menggunakan StandardScaler.

    Args:
        df (pd.DataFrame): DataFrame input (belum dinormalisasi).

    Returns:
        tuple: (DataFrame ternormalisasi, objek scaler yang sudah di-fit)
    """
    print("[4/5] Melakukan feature scaling (StandardScaler)...")
    
    feature_cols = [c for c in df.columns if c != 'target']
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    print(f"      ✅ Scaling selesai pada {len(feature_cols)} fitur")
    return df, scaler


def split_and_save(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Membagi data menjadi train/test dan menyimpannya ke file CSV.

    Args:
        df (pd.DataFrame): DataFrame yang sudah diproses.
        output_dir (str): Direktori output.

    Returns:
        tuple: (DataFrame train, DataFrame test)
    """
    print(f"[5/5] Membagi dan menyimpan data ke '{output_dir}/'...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    train_df = X_train.copy()
    train_df['target'] = y_train.values
    
    test_df = X_test.copy()
    test_df['target'] = y_test.values
    
    train_path = os.path.join(output_dir, "train.csv")
    test_path  = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"      ✅ Data train: {train_df.shape} → disimpan ke '{train_path}'")
    print(f"      ✅ Data test:  {test_df.shape} → disimpan ke '{test_path}'")
    
    return train_df, test_df


def preprocess_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline utama preprocessing: load → handle missing → remove duplicates
    → scale features → split & save.

    Returns:
        tuple: (train_df, test_df) - DataFrame hasil preprocessing
    """
    print("=" * 60)
    print("   PIPELINE PREPROCESSING OTOMATIS")
    print("   Proyek: Membangun Sistem Machine Learning (Dicoding)")
    print("=" * 60)
    
    # Step 1 - Load data
    df = load_data()
    
    # Step 2 - Handle missing values
    df = handle_missing_values(df, strategy='mean')
    
    # Step 3 - Remove duplicates
    df = remove_duplicates(df)
    
    # Step 4 - Scale features
    df, scaler = scale_features(df)
    
    # Step 5 - Split & save
    train_df, test_df = split_and_save(df)
    
    print("\n" + "=" * 60)
    print("   ✅ PREPROCESSING SELESAI!")
    print(f"   Train: {train_df.shape[0]} baris | Test: {test_df.shape[0]} baris")
    print("=" * 60)
    
    return train_df, test_df


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    train_df, test_df = preprocess_pipeline()
