# Tiki Book Intelligence v2

**Hệ thống dự đoán và phân tích hiệu quả sản phẩm sách trên sàn thương mại điện tử Tiki**  
kết hợp Data Mining, Machine Learning và Deep Learning NLP (PhoBERT).

🔗 **Dashboard trực tuyến**: https://tikibookintelligence.streamlit.app/  
📁 **GitHub**: https://github.com/ndhieu2ce/TikiBookIntelligence-v2

---

## Giới thiệu

Tiki Book Intelligence v2 sử dụng **học có giám sát** để dự đoán và phân tích hiệu quả các sản phẩm sách trên Tiki. Hệ thống gồm 3 lớp chính:

- **Data Mining**: Crawl dữ liệu sản phẩm + reviews từ API public của Tiki
- **Machine Learning**: Gán nhãn bằng HDBSCAN clustering + train Random Forest / XGBoost / Decision Tree
- **Deep Learning NLP**: Phân tích cảm xúc tiếng Việt bằng PhoBERT (`wonrax/phobert-base-vietnamese-sentiment`)

---

## Dữ liệu thực tế

| Chỉ số | Giá trị |
|--------|---------|
| Tổng dòng reviews | 60,778 |
| Sản phẩm duy nhất | 15,060 |
| Bình luận có nội dung | 26,830 |
| Bình luận rỗng | 33,948 |
| Reviews tối đa / sản phẩm | 50 |

---

## 5 Nhãn hiệu quả sản phẩm (HDBSCAN — 24 cụm tự động)

| Nhãn | Số lượng | Đặc trưng |
|------|----------|-----------|
| 🌟 Best Seller | 945 | sold_count cao (~1,417), rating ~4.70 |
| 📈 High Potential | 358 | Có tín hiệu bán nhưng chưa đủ đánh giá |
| 💎 Premium / Niche Quality | 2,672 | Giá cao, rating tốt |
| 📊 Normal | 8,253 | Sản phẩm trung bình, ít review |
| ⚠️ Needs Improvement | 2,832 | Rating thấp hoặc negative_ratio cao |

> Outlier (1,898 sản phẩm — 12.6%) được tự động gán vào cụm gần nhất theo Euclidean distance.

---

## Kết quả mô hình

### Hành trình cải thiện

| Giai đoạn | Phương pháp gán nhãn | Sentiment | Accuracy | F1 Weighted | Ghi chú |
|-----------|----------------------|-----------|----------|-------------|---------|
| 1 — Ban đầu | Quantile thủ công | Không có | 99.1% | 0.9910 | ❌ Label leakage |
| 2 — HDBSCAN + Fix leakage | HDBSCAN (24 cụm) | Sentiment Simple | 80.61% | 0.7716 | ✅ Kết quả thực |
| 3 — PhoBERT + GridSearchCV | HDBSCAN (24 cụm) | PhoBERT (wonrax) | 80.11% | 0.7722 | ✅ Tối ưu siêu tham số |

### So sánh 3 mô hình (test set thực tế — sau GridSearchCV)

| Mô hình | Accuracy | Precision | Recall | F1 Weighted | CV F1 |
|---------|----------|-----------|--------|-------------|-------|
| Decision Tree | 79.32% | 0.7852 | 0.7932 | 0.7552 | — |
| **Random Forest ✅** | **80.11%** | **0.7840** | **0.8011** | **0.7722** | **0.7717** |
| XGBoost | 80.35% | 0.8058 | 0.8035 | 0.7620 | 0.7645 |

→ **Random Forest được chọn** vì F1 Weighted cao nhất (0.7722)

### Feature Importance (Random Forest)

```
avg_comment_rating : 26%  ← quan trọng nhất
price              : 25%
review_count       : 22%
sentiment_score    : 16%  ← từ PhoBERT
comment_count      :  8%
neutral_ratio      :  2%
```

### Siêu tham số tối ưu (GridSearchCV)

**Random Forest** — 18 cấu hình × 5-fold CV:
```python
RandomForestClassifier(max_depth=20, max_features='sqrt', n_estimators=50, random_state=42)
# Best CV F1: 0.7717
```

**XGBoost** — 27 cấu hình × 5-fold CV:
```python
XGBClassifier(learning_rate=0.2, max_depth=8, n_estimators=150, tree_method='hist', random_state=42)
# Best CV F1: 0.7645
```

---

## Cấu trúc thư mục

```
TikiBookIntelligence-v2/
├── data/
│   ├── tiki_books_reviews.csv              # Dữ liệu crawl thô (dòng là review)
│   ├── tiki_books_labeled.csv              # Dữ liệu đã gán nhãn HDBSCAN (dòng là sản phẩm)
│   ├── tiki_books_sentiment.csv            # Điểm sentiment từ Sentiment Simple
│   ├── tiki_books_sentiment_phobert.csv    # Điểm sentiment từ PhoBERT
│   ├── model_comparison.csv                # So sánh 3 mô hình
│   ├── classification_report.csv           # Báo cáo phân loại chi tiết
│   ├── gridsearch_rf.csv                   # Kết quả GridSearch Random Forest
│   ├── gridsearch_xgb.csv                  # Kết quả GridSearch XGBoost
│   └── crawl_state.json                    # Trạng thái crawl (trang cuối mỗi keyword)
├── images/
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── models/
│   ├── product_performance_model.pkl       # Random Forest đã train
│   └── label_encoder.pkl
├── crawl_tiki_books.py                     # Crawler đa keyword từ Tiki
├── label_products.py                       # Gán nhãn bằng HDBSCAN
├── sentiment_simple.py                     # Phân tích cảm xúc rule-based
├── sentiment_phobert.py                    # Phân tích cảm xúc PhoBERT (Colab)
├── train_product_classifier.py             # Train + GridSearchCV + lưu model
├── update_sold_count.py                    # Cập nhật sold_count từ API chi tiết
├── app.py                                  # Dashboard Streamlit (8 trang)
├── requirements.txt
└── README.md
```

---

## Cài đặt

```bash
git clone https://github.com/ndhieu2ce/TikiBookIntelligence-v2
cd TikiBookIntelligence-v2
python -m pip install -r requirements.txt
```

---

## Cách sử dụng

### Bước 1: Crawl dữ liệu

```bash
# Crawl lần đầu (28 keyword, mỗi keyword 10 trang, ~3,000 sản phẩm)
python crawl_tiki_books.py

# Crawl tiếp (tự động tiếp tục từ trang cuối)
python crawl_tiki_books.py

# Tùy chọn số trang
python crawl_tiki_books.py --pages 20

# Reset tiến độ và crawl lại từ đầu
python crawl_tiki_books.py --reset-state
```

### Bước 2: Gán nhãn sản phẩm

```bash
python label_products.py
```

Output: `data/tiki_books_labeled.csv` — mỗi dòng là 1 sản phẩm, gồm nhãn HDBSCAN và các đặc trưng tổng hợp.

### Bước 3: Phân tích cảm xúc

```bash
# Sentiment Simple (chạy local, nhanh)
python sentiment_simple.py

# Sentiment PhoBERT (khuyến nghị chạy trên Google Colab GPU T4)
python sentiment_phobert.py
```

### Bước 4: Huấn luyện mô hình

```bash
python train_product_classifier.py
```

Output: `models/product_performance_model.pkl`, `data/model_comparison.csv`, `images/confusion_matrix.png`, `images/feature_importance.png`

### Bước 5: Chạy dashboard

```bash
python -m streamlit run app.py
```

Truy cập `http://localhost:8501`

---

## Dashboard — 8 trang

| Trang | Nội dung |
|-------|----------|
| 🏠 Tổng quan | Thống kê tổng hợp, biểu đồ phân bố nhãn |
| 🗄️ Dữ liệu sản phẩm | Bảng dữ liệu, tìm kiếm và lọc theo nhãn |
| 🏷️ Gán nhãn sản phẩm | Giải thích 5 nhãn, biểu đồ donut |
| 🧠 Mô hình dự đoán | So sánh 3 mô hình, cách hệ thống hoạt động |
| 📊 Đánh giá mô hình | Confusion matrix, feature importance, classification report |
| 🔎 Dự đoán sản phẩm | Dán link Tiki → tự lấy dữ liệu → dự đoán nhãn |
| 💡 Gợi ý cải thiện | Danh sách sản phẩm Needs Improvement + gợi ý cụ thể |
| 🚩 Kết luận | Hành trình cải thiện mô hình, kết quả đạt được |

---

## Ghi chú quan trọng

**Doanh thu ước tính** = giá bán × số lượng đã bán — là ước tính tham khảo, không phải doanh thu thực tế.

**Nhãn sản phẩm** được tạo bằng HDBSCAN clustering (không dùng ngưỡng thủ công), đảm bảo không có label leakage khi train model.

**PhoBERT** nên chạy trên Google Colab GPU T4 (~8 phút cho 15,060 sản phẩm). Nếu chưa có file PhoBERT, hệ thống tự fallback sang Sentiment Simple.

---

## Hướng phát triển tương lai

- Crawl định kỳ tự động và cập nhật dashboard theo thời gian thực
- Mở rộng NLP: phân tích chủ đề (topic modeling) từ nội dung review
- Xây dựng module dự báo xu hướng bán hàng (time series)
- Mở rộng sang các danh mục sản phẩm khác ngoài sách

---

**Cập nhật lần cuối**: 2026-05-18