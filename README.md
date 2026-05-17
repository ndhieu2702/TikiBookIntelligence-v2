# 📚 Tiki Book Intelligence

> Hệ thống phân tích hiệu quả sản phẩm sách trên sàn TMĐT Tiki sử dụng **HDBSCAN Clustering**, **Random Forest** và **PhoBERT Sentiment Analysis**.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![PhoBERT](https://img.shields.io/badge/PhoBERT-VinAI-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Giới thiệu

**Tiki Book Intelligence** là hệ thống thông minh kết hợp 3 lĩnh vực:

- **Data Mining** — Crawl và xử lý 60,778 dòng dữ liệu từ Tiki
- **Machine Learning** — HDBSCAN Clustering + Random Forest/XGBoost phân loại sản phẩm
- **Deep Learning NLP** — PhoBERT (`wonrax/phobert-base-vietnamese-sentiment`) phân tích cảm xúc bình luận tiếng Việt

---

## 📊 Kết quả đạt được

| Giai đoạn | Accuracy | F1 Weighted | Ghi chú |
|-----------|----------|-------------|---------|
| Ban đầu (label leakage) | 99.1% | 0.991 | ❌ Giả |
| Sau HDBSCAN | 80.3% | 0.768 | ✅ Nhãn khách quan |
| Sau Sentiment Simple | 80.61% | 0.7716 | ✅ Thêm từ điển tiếng Việt |
| Sau PhoBERT | 80.81% | 0.7736 | ✅ Deep Learning NLP |
| **Sau PhoBERT + GridSearch** | **80.11%** | **0.7722** | ✅ Siêu tham số tối ưu |

### 5 Nhãn sản phẩm (HDBSCAN — 24 cụm tự động)

| Nhãn | Số lượng | Mô tả |
|------|----------|-------|
| 🔵 Normal | 8,253 | Sản phẩm trung bình, ít tương tác |
| 🔴 Needs Improvement | 2,832 | Cần cải thiện chất lượng hoặc phản hồi |
| 🟣 Premium / Niche Quality | 2,672 | Phân khúc cao cấp hoặc ngách |
| 🟡 High Potential | 358 | Tiềm năng tăng trưởng tốt |
| 🟢 Best Seller | 945 | Bán chạy, doanh thu cao |

---

## 🏗️ Kiến trúc hệ thống

```
crawl_tiki_books.py           → data/tiki_books_reviews.csv (60,778 dòng)
    → label_products.py       → data/tiki_books_labeled.csv (15,060 sản phẩm)
        → sentiment_simple.py → data/tiki_books_sentiment.csv
            → sentiment_phobert.py → data/tiki_books_sentiment_phobert.csv
                → train_product_classifier.py → models/*.pkl
                    → app.py  → Streamlit Dashboard
```

---

## 📁 Cấu trúc thư mục

```
TikiBookIntelligence/
├── .claude/
│   └── CLAUDE.md                          # Bộ não dự án cho Claude Code
├── data/
│   ├── tiki_books_reviews.csv             # Dữ liệu crawl thô
│   ├── tiki_books_labeled.csv             # Dữ liệu đã gán nhãn
│   ├── tiki_books_sentiment.csv           # Sentiment từ điển
│   ├── tiki_books_sentiment_phobert.csv   # Sentiment PhoBERT
│   ├── model_comparison.csv               # So sánh mô hình
│   └── classification_report.csv          # Báo cáo phân loại
├── images/
│   ├── confusion_matrix.png               # Ma trận nhầm lẫn
│   └── feature_importance.png             # Tầm quan trọng đặc trưng
├── models/
│   ├── label_encoder.pkl                  # Bộ mã hóa nhãn
│   └── product_performance_model.pkl      # Mô hình tốt nhất
├── app.py                                 # Streamlit dashboard
├── crawl_tiki_books.py                    # Crawler Tiki
├── label_products.py                      # Gán nhãn HDBSCAN
├── sentiment_simple.py                    # Sentiment từ điển tiếng Việt
├── sentiment_phobert.py                   # Sentiment PhoBERT
├── train_product_classifier.py            # Huấn luyện mô hình
├── update_sold_count.py                   # Cập nhật lượt bán
├── requirements.txt                       # Thư viện cần thiết
└── README.md
```

---

## 🚀 Cài đặt và chạy

### 1. Clone repo
```bash
git clone https://github.com/ndhieu2ce/TikiBookIntelligence-v2.git
cd TikiBookIntelligence-v2
```

### 2. Cài thư viện
```bash
pip install -r requirements.txt
```

### 3. Chạy pipeline (theo thứ tự)
```bash
# Crawl dữ liệu
python crawl_tiki_books.py

# Gán nhãn bằng HDBSCAN
python label_products.py

# Tính sentiment
python sentiment_simple.py
python sentiment_phobert.py  # Cần GPU, khuyến nghị chạy trên Google Colab

# Huấn luyện mô hình
python train_product_classifier.py

# Chạy dashboard
python -m streamlit run app.py
```

---

## 🧠 Công nghệ sử dụng

### Machine Learning
- **HDBSCAN** — Clustering tự động, không cần đặt số cụm k
- **Random Forest** (F1 Weighted = 0.7722, sau GridSearchCV)
  - Best params: n_estimators=50, max_depth=20, max_features='sqrt'
- **XGBoost** (Accuracy = 80.35%, sau GridSearchCV)
  - Best params: n_estimators=150, max_depth=8, learning_rate=0.2
- **Decision Tree** — Model baseline để so sánh

### Deep Learning NLP
- **PhoBERT** (`wonrax/phobert-base-vietnamese-sentiment`)
- Model Transformer 12 lớp, 135M tham số
- Fine-tuned cho sentiment tiếng Việt (POS/NEG/NEU)
- Chạy trên Google Colab GPU T4

### Features train model
```python
FEATURES = [
    'price',             # Giá bán
    'review_count',      # Số đánh giá
    'comment_count',     # Số bình luận
    'avg_comment_rating',# Điểm bình luận trung bình
    'neutral_ratio',     # Tỷ lệ trung tính
    'sentiment_score'    # Điểm cảm xúc từ PhoBERT
]
```

---

## 📈 Dashboard

Dashboard Streamlit gồm **8 trang**:

| Trang | Nội dung |
|-------|---------|
| 🏠 Tổng quan | Thống kê tổng hợp, phân bố nhãn |
| 🗄️ Dữ liệu sản phẩm | Bảng dữ liệu, tìm kiếm, lọc theo nhãn |
| 🏷️ Gán nhãn | Biểu đồ donut, thống kê nhãn |
| 🧠 Mô hình dự đoán | So sánh 3 mô hình ML |
| 📊 Đánh giá mô hình | Confusion matrix, Feature importance, So sánh PhoBERT |
| 🔎 Dự đoán sản phẩm | Nhập link Tiki → dự đoán nhãn realtime |
| 💡 Gợi ý cải thiện | Danh sách sản phẩm cần cải thiện |
| 🚩 Kết luận | Tổng kết, hành trình cải thiện, hướng phát triển |

---

## 📝 Ghi chú

- **Doanh thu ước tính** = giá bán × lượt bán (tham khảo, không phải thực tế)
- **PhoBERT** chạy trên CPU mất 2-4 tiếng, khuyến nghị dùng Google Colab GPU
- **Accuracy 80.81%** là con số thực tế sau khi fix label leakage

---

## 👨‍💻 Tác giả

- **Sinh viên**: Năm 3, ngành Khoa học Máy tính - AI
- **Mục tiêu**: Tốt nghiệp loại Xuất Sắc

---

*Cập nhật lần cuối: 2026-05-17*
