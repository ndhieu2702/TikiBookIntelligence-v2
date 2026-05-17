"""
Huấn luyện mô hình dự đoán hiệu quả sản phẩm.
- So sánh Decision Tree, Random Forest, XGBoost.
- Chọn mô hình tốt nhất dựa trên F1-score weighted.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
# Ưu tiên PhoBERT → sentiment_simple → labeled
DATA_PATH_PHOBERT   = "data/tiki_books_sentiment_phobert.csv"
DATA_PATH           = "data/tiki_books_sentiment.csv"
DATA_PATH_FALLBACK  = "data/tiki_books_labeled.csv"
OUTPUT_CSV = "data/model_comparison.csv"
REPORT_CSV = "data/classification_report.csv"
CM_PATH = "images/confusion_matrix.png"
FI_PATH = "images/feature_importance.png"
MODEL_PATH = "models/product_performance_model.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
GRIDSEARCH_RF_CSV  = "data/gridsearch_rf.csv"
GRIDSEARCH_XGB_CSV = "data/gridsearch_xgb.csv"

# Nhóm gán nhãn (KHÔNG dùng để train — tránh label leakage):
#   sold_count, estimated_revenue, rating, positive_ratio, negative_ratio
# Nhóm train model (độc lập với quy tắc gán nhãn):
BASE_FEATURES = [
    'price',
    'review_count',
    'comment_count',
    'avg_comment_rating',
    'neutral_ratio',
]
TARGET = 'product_label'

# ==================== UTILITY FUNCTIONS ====================
def load_data():
    """Đọc dữ liệu — ưu tiên PhoBERT sentiment, fallback về simple sentiment, rồi labeled."""
    # Xác định file base (có đầy đủ features)
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        source = DATA_PATH
    elif os.path.exists(DATA_PATH_FALLBACK):
        print(f"⚠️  {DATA_PATH} chưa có — dùng fallback {DATA_PATH_FALLBACK}")
        df = pd.read_csv(DATA_PATH_FALLBACK)
        source = DATA_PATH_FALLBACK
    else:
        print(f"❌ Không tìm thấy dữ liệu. Chạy: python label_products.py")
        exit(1)

    # Override sentiment_score bằng PhoBERT nếu có
    if os.path.exists(DATA_PATH_PHOBERT):
        phobert = pd.read_csv(DATA_PATH_PHOBERT)
        before = df.get('sentiment_score', pd.Series(dtype=float)).mean()
        df = df.merge(phobert[['product_id', 'sentiment_score']],
                      on='product_id', how='left', suffixes=('_simple', '_phobert'))
        # Dùng PhoBERT khi có, giữ simple khi PhoBERT NaN
        if 'sentiment_score_phobert' in df.columns:
            df['sentiment_score'] = df['sentiment_score_phobert'].fillna(
                df.get('sentiment_score_simple', 0)
            )
            df.drop(columns=['sentiment_score_simple', 'sentiment_score_phobert'],
                    errors='ignore', inplace=True)
        matched = phobert['product_id'].isin(df['product_id']).sum()
        print(f"✅ Đã đọc {len(df):,} sản phẩm từ {source}")
        print(f"🧠 Override sentiment_score bằng PhoBERT ({matched:,}/{len(df):,} sản phẩm)")
    else:
        print(f"✅ Đã đọc {len(df):,} sản phẩm từ {source}")
        print(f"⚠️  {DATA_PATH_PHOBERT} chưa có — dùng sentiment_simple")

    return df

def prepare_data(df):
    """Chuẩn bị dữ liệu cho huấn luyện."""
    print("\n📋 Chuẩn bị dữ liệu...")

    # Thêm sentiment_score vào features nếu cột tồn tại trong dữ liệu
    features = list(BASE_FEATURES)
    if 'sentiment_score' in df.columns:
        features.append('sentiment_score')
        print(f"   ✓ Phát hiện sentiment_score — đã thêm vào features")
    else:
        print(f"   ⚠️  Không có sentiment_score — chạy sentiment_simple.py để thêm")

    # Kiểm tra cột
    for col in features + [TARGET]:
        if col not in df.columns:
            print(f"❌ Cột {col} không tồn tại")
            exit(1)

    X = df[features].copy()
    y = df[TARGET].copy()

    # Xử lý NaN
    X = X.fillna(0)

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"   ✓ Features: {features}")
    print(f"   ✓ Target: {len(le.classes_)} lớp")
    for i, label in enumerate(le.classes_):
        count = sum(y_encoded == i)
        print(f"      - {label}: {count}")

    return X, y_encoded, y, le, features

def train_test_split_data(X, y_encoded):
    """Chia dữ liệu train/test."""
    print("\n📊 Chia dữ liệu train/test...")
    
    # Kiểm tra stratify
    use_stratify = True
    for label in np.unique(y_encoded):
        if sum(y_encoded == label) < 2:
            use_stratify = False
            print(f"   ⚠️  Cảnh báo: Lớp {label} có quá ít mẫu, không dùng stratify")
            break
    
    if use_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
    
    print(f"   ✓ Train: {len(X_train)} mẫu")
    print(f"   ✓ Test: {len(X_test)} mẫu")
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, X_test, y_train, y_test, le):
    """Huấn luyện các mô hình."""
    print("\n🤖 Huấn luyện các mô hình...")
    
    models_data = []
    trained_models = {}
    
    # 1. Decision Tree
    print("\n   Decision Tree Classifier...")
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_metrics = evaluate_model(dt_pred, y_test, "Decision Tree Classifier")
    models_data.append(dt_metrics)
    trained_models["Decision Tree"] = dt_model
    
    # 2. Random Forest + GridSearchCV
    print("\n   Random Forest Classifier (GridSearchCV)...")
    rf_param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'max_features': ['sqrt', 'log2']
    }
    rf_gs = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_param_grid,
        cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1
    )
    rf_gs.fit(X_train, y_train)
    print(f"      Best params: {rf_gs.best_params_}")
    print(f"      Best CV F1:  {rf_gs.best_score_:.4f}")
    pd.DataFrame(rf_gs.cv_results_).to_csv(GRIDSEARCH_RF_CSV, index=False)
    rf_model = rf_gs.best_estimator_
    rf_pred = rf_model.predict(X_test)
    rf_metrics = evaluate_model(rf_pred, y_test, "Random Forest Classifier")
    models_data.append(rf_metrics)
    trained_models["Random Forest"] = rf_model
    
    # 3. XGBoost + GridSearchCV
    try:
        import xgboost as xgb
        print("\n   XGBoost Classifier (GridSearchCV)...")
        xgb_param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.10, 0.20]
        }
        xgb_gs = GridSearchCV(
            xgb.XGBClassifier(random_state=42, tree_method='hist'),
            xgb_param_grid,
            cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        xgb_gs.fit(X_train, y_train)
        print(f"      Best params: {xgb_gs.best_params_}")
        print(f"      Best CV F1:  {xgb_gs.best_score_:.4f}")
        pd.DataFrame(xgb_gs.cv_results_).to_csv(GRIDSEARCH_XGB_CSV, index=False)
        xgb_model = xgb_gs.best_estimator_
        xgb_pred = xgb_model.predict(X_test)
        xgb_metrics = evaluate_model(xgb_pred, y_test, "XGBoost Classifier")
        models_data.append(xgb_metrics)
        trained_models["XGBoost"] = xgb_model
    except ImportError:
        print("\n   ⚠️  Chưa cài xgboost. Hãy chạy: python -m pip install xgboost")
    
    return models_data, trained_models, y_test

def evaluate_model(y_pred, y_test, model_name):
    """Đánh giá mô hình."""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"      Accuracy: {accuracy:.4f}")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall: {recall:.4f}")
    print(f"      F1-score: {f1:.4f}")
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }

def save_model_comparison(models_data):
    """Lưu bảng so sánh mô hình."""
    df_comparison = pd.DataFrame(models_data)
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    df_comparison.to_csv(OUTPUT_CSV, index=False)
    return df_comparison

def save_classification_report(best_model, X_test, y_test, le):
    """Lưu classification report."""
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, 
                                    target_names=le.classes_,
                                    output_dict=True)
    
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(REPORT_CSV)

def plot_confusion_matrix(best_model, X_test, y_test, le):
    """Vẽ confusion matrix."""
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues', aspect='auto')
    plt.title('Confusion Matrix - Mô hình tốt nhất', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(range(len(le.classes_)), le.classes_, rotation=45, ha='right')
    plt.yticks(range(len(le.classes_)), le.classes_)
    
    # Thêm giá trị vào từng ô
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', 
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    plt.colorbar(label='Count')
    plt.tight_layout()
    os.makedirs(os.path.dirname(CM_PATH) or ".", exist_ok=True)
    plt.savefig(CM_PATH, dpi=150, bbox_inches='tight')
    plt.close()

def plot_feature_importance(best_model, model_name, features):
    """Vẽ feature importance."""
    if not hasattr(best_model, 'feature_importances_'):
        print(f"⚠️  Mô hình {model_name} không có feature_importances_")
        return

    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance - Mô hình tốt nhất', fontsize=14, fontweight='bold')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45, ha='right')
    plt.ylabel('Importance', fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(FI_PATH) or ".", exist_ok=True)
    plt.savefig(FI_PATH, dpi=150, bbox_inches='tight')
    plt.close()

def save_best_model(best_model, le):
    """Lưu mô hình tốt nhất."""
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

def main():
    print(f"\n🚀 Tiki Book Intelligence - Huấn luyện mô hình")
    print(f"   Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Đọc dữ liệu
    df = load_data()
    X, y_encoded, y, le, features = prepare_data(df)
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split_data(X, y_encoded)
    
    # Huấn luyện mô hình
    models_data, trained_models, y_test = train_models(X_train, X_test, y_train, y_test, le)
    
    # Lưu bảng so sánh
    df_comparison = save_model_comparison(models_data)
    
    # Chọn mô hình tốt nhất
    best_idx = df_comparison['F1-score'].idxmax()
    best_model_name = df_comparison.loc[best_idx, 'Model']
    
    if "Decision Tree" in best_model_name:
        best_model = trained_models["Decision Tree"]
    elif "Random Forest" in best_model_name:
        best_model = trained_models["Random Forest"]
    elif "XGBoost" in best_model_name:
        best_model = trained_models["XGBoost"]
    else:
        best_model = list(trained_models.values())[0]
        best_model_name = list(trained_models.keys())[0]
    
    best_metrics = df_comparison.loc[best_idx]
    
    print(f"\n🏆 Mô hình tốt nhất: {best_model_name}")

    # Lưu classification report, confusion matrix, feature importance, model
    save_classification_report(best_model, X_test, y_test, le)
    plot_confusion_matrix(best_model, X_test, y_test, le)
    plot_feature_importance(best_model, best_model_name, features)
    save_best_model(best_model, le)

    print(f"\n📁 Đã lưu:")
    print(f"   {OUTPUT_CSV}")
    print(f"   {REPORT_CSV}")
    print(f"   {CM_PATH}")
    print(f"   {FI_PATH}")
    print(f"   {MODEL_PATH}")
    print(f"   {LABEL_ENCODER_PATH}")

    print(f"\n" + "="*60)
    print("✅ Hoàn thành huấn luyện mô hình!")
    print("="*60)

if __name__ == "__main__":
    main()
