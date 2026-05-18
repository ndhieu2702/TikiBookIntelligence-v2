# TikiBookIntelligence — CLAUDE.md

## Mô tả dự án
Hệ thống dự đoán và phân tích hiệu quả sản phẩm sách trên Tiki kết hợp Data Mining, Machine Learning và Deep Learning NLP (PhoBERT).
- **Sinh viên**: Năm 3, ngành KHMT - AI, mục tiêu tốt nghiệp Xuất Sắc
- **Dashboard**: https://tikibookintelligence.streamlit.app/
- **Repo GitHub**: https://github.com/ndhieu2ce/TikiBookIntelligence-v2

## Pipeline hoàn chỉnh
```
crawl_tiki_books.py           → data/tiki_books_reviews.csv
    → label_products.py       → data/tiki_books_labeled.csv
        → sentiment_simple.py → data/tiki_books_sentiment.csv
            → sentiment_phobert.py → data/tiki_books_sentiment_phobert.csv
                → train_product_classifier.py → models/*.pkl
                    → app.py  → Streamlit dashboard (8 trang)
```

## ⚠️ Thứ tự chạy bắt buộc
```bash
python label_products.py
python sentiment_simple.py
python train_product_classifier.py
python -m streamlit run app.py
```

## Dữ liệu thực tế
- **60,778 dòng** reviews | **15,060 sản phẩm** duy nhất
- **MAX_REVIEWS_PER_PRODUCT = 50**
- Chống trùng review theo **review_id**
- 26,830 bình luận có nội dung | 33,948 bình luận rỗng

## 5 Nhãn sản phẩm (HDBSCAN — 24 cụm tự động)
| Nhãn | Số lượng | Đặc trưng |
|------|----------|-----------|
| Normal | 8,253 | Sản phẩm trung bình, ít review |
| Needs Improvement | 2,832 | Rating thấp hoặc negative_ratio cao |
| Premium / Niche Quality | 2,672 | Giá cao, rating tốt |
| High Potential | 358 | Có tín hiệu bán nhưng chưa đủ đánh giá |
| Best Seller | 945 | sold_count cao (~1417), rating ~4.70 |
| Outlier | 1,898 (12.6%) | Tự động map vào cụm gần nhất |

## Kết quả model (số liệu thật — chạy 2026-05-17)

### Hành trình cải thiện F1
| Giai đoạn | Accuracy | F1 Weighted | Ghi chú |
|-----------|----------|-------------|---------|
| Ban đầu (label leakage) | 99.1% | 0.9910 | ❌ Giả |
| Sau HDBSCAN + fix leakage | 80.3% | 0.7680 | ✅ Nhãn khách quan |
| Sau sentiment_simple | 80.61% | 0.7716 | ✅ +0.0036 |
| Sau PhoBERT | 80.81% | 0.7736 | ✅ Deep Learning |
| **Sau PhoBERT + GridSearch** | **80.11%** | **0.7722** | ✅ Siêu tham số tối ưu |

### So sánh 3 mô hình (test set thực tế — sau GridSearchCV)
| Model | Accuracy | Precision | Recall | F1 Weighted | CV F1 |
|-------|----------|-----------|--------|-------------|-------|
| Decision Tree | 79.32% | 0.7852 | 0.7932 | 0.7552 | — |
| **Random Forest ✅** | **80.11%** | **0.7840** | **0.8011** | **0.7722** | **0.7717** |
| XGBoost | 80.35% | 0.8058 | 0.8035 | 0.7620 | 0.7645 |

→ **Random Forest được chọn** vì F1 Weighted cao nhất (0.7722)
→ XGBoost có Accuracy cao hơn (80.35%) nhưng F1 thấp hơn (0.7620)

### Siêu tham số tối ưu (GridSearchCV — số liệu thật)

**Random Forest** — 18 cấu hình (3×3×2), 5-fold CV:
```python
RandomForestClassifier(
    max_depth=20,
    max_features='sqrt',
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)
# Best CV F1: 0.7717
```

**XGBoost** — 27 cấu hình (3×3×3), 5-fold CV:
```python
XGBClassifier(
    learning_rate=0.2,
    max_depth=8,
    n_estimators=150,
    tree_method='hist',
    random_state=42
)
# Best CV F1: 0.7645
```

### Feature Importance (Random Forest — thực tế)
```
avg_comment_rating : 26%  ← quan trọng nhất
price              : 25%
review_count       : 22%
sentiment_score    : 16%  ← từ PhoBERT
comment_count      :  8%
neutral_ratio      :  2%
```

## Tất cả thay đổi đã làm

### 1. Crawl data
- `MAX_REVIEWS_PER_PRODUCT = 50` (tăng từ 20)
- Chống trùng review theo `review_id`
- Index `product_id → set(review_id)` build trước vòng crawl

### 2. Gán nhãn bằng HDBSCAN
- File: `label_products.py`
- HDBSCAN tự tìm **24 cụm** → map về 5 nhãn
- Chuẩn hóa bằng **StandardScaler** trước clustering
- Features clustering: `sold_count`, `estimated_revenue`, `rating`, `positive_ratio`, `negative_ratio`
- 1,898 outlier → gán vào cụm gần nhất theo Euclidean distance
- Thay thế hoàn toàn ngưỡng quantile tự đặt cũ

### 3. Fix label leakage
- Features train model tách biệt hoàn toàn với features gán nhãn
- File: `train_product_classifier.py`
- Features train model:
```python
FEATURES = [
    'price', 'review_count', 'comment_count',
    'avg_comment_rating', 'neutral_ratio', 'sentiment_score'
]
```

### 4. Sentiment Simple
- File: `sentiment_simple.py`
- ~130 từ tích cực, ~108 từ tiêu cực, bigrams, xử lý phủ định
- Fallback: `rating > 0` → convert [-1,1] | `rating=0` → 0.0
- Output: `data/tiki_books_sentiment.csv`

### 5. PhoBERT Sentiment
- File: `sentiment_phobert.py`
- Model: `wonrax/phobert-base-vietnamese-sentiment` (~135M tham số)
- Chạy trên Google Colab GPU T4 (~8 phút cho 15,060 SP)
- Checkpoint mỗi 1,000 sản phẩm
- `train_product_classifier.py` tự override sentiment_simple bằng PhoBERT
- Output: `data/tiki_books_sentiment_phobert.csv`

### 6. GridSearchCV
- File: `train_product_classifier.py`
- Random Forest: 18 cấu hình × 5-fold = 90 lần train
- XGBoost: 27 cấu hình × 5-fold = 135 lần train
- Kết quả lưu: `data/gridsearch_rf.csv`, `data/gridsearch_xgb.csv`

### 7. Dashboard (app.py) — hoàn chỉnh
- 8 trang hoàn chỉnh
- Trang Mô hình dự đoán: mô tả đúng 6 features thực tế
- Trang Kết luận: bảng hành trình đọc động từ model_comparison.csv (không hardcode)
- Trang Dự đoán: hướng dẫn 3 bước + ví dụ link + card 5 nhãn
- MODEL_FEATURE_COLUMNS sync đúng với features train:
```python
MODEL_FEATURE_COLUMNS = [
    'price', 'review_count', 'comment_count',
    'avg_comment_rating', 'neutral_ratio', 'sentiment_score'
]
```

## Cấu trúc thư mục
```
TikiBookIntelligence-v2/
├── .claude/
│   └── CLAUDE.md
├── data/
│   ├── tiki_books_reviews.csv
│   ├── tiki_books_labeled.csv
│   ├── tiki_books_sentiment.csv
│   ├── tiki_books_sentiment_phobert.csv
│   ├── model_comparison.csv
│   ├── classification_report.csv
│   ├── gridsearch_rf.csv
│   ├── gridsearch_xgb.csv
│   └── crawl_state.json
├── images/
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── models/
│   ├── label_encoder.pkl
│   └── product_performance_model.pkl
├── app.py
├── crawl_tiki_books.py
├── label_products.py
├── sentiment_simple.py
├── sentiment_phobert.py
├── train_product_classifier.py
├── update_sold_count.py
├── requirements.txt
├── README.md
└── CLAUDE.md
```

## Quy tắc code
- Comment bằng **tiếng Việt**
- Dùng **f-string**, không dùng `.format()`
- Luôn `try/except` khi gọi API Tiki
- Luôn `fillna(0)` trước khi xử lý số
- **Không để print bị lặp**
- **Không bịa số liệu** — mọi con số phải từ code chạy thật
- Print emoji: ✅ thành công | ❌ lỗi | ⚠️ cảnh báo | 🚀 bắt đầu | 📊 thống kê

## Debug thường gặp
| Lỗi | Cách fix |
|-----|----------|
| Feature names mismatch | Chạy lại train_product_classifier.py; kiểm tra MODEL_FEATURE_COLUMNS trong app.py |
| Nhãn hiển thị sai | Chạy lại label_products.py → sentiment_simple.py → train |
| `monotonic_cst` error | Chạy lại train_product_classifier.py |
| Streamlit không load model | Chạy train_product_classifier.py trước |
| HDBSCAN ra ít cụm | Điều chỉnh `min_cluster_size` trong label_products.py |

---
*Cập nhật lần cuối: 2026-05-18 — Dự án hoàn chỉnh: pipeline, dashboard, deploy, báo cáo, slide*