"""
Phân tích cảm xúc đơn giản bằng từ điển tiếng Việt.
- Đọc comment_text từ tiki_books_reviews.csv
- Tính sentiment_score (-1 đến 1) cho từng sản phẩm
- Lưu ra data/tiki_books_sentiment.csv (merge với tiki_books_labeled.csv)
"""

import os
import re
import pandas as pd
import numpy as np
from datetime import datetime

REVIEWS_PATH = "data/tiki_books_reviews.csv"
LABELED_PATH = "data/tiki_books_labeled.csv"
OUTPUT_PATH = "data/tiki_books_sentiment.csv"

# ==================== TỪ ĐIỂN CẢM XÚC TIẾNG VIỆT ====================

# Từ phủ định (đảo ngược cảm xúc trong phạm vi 3 từ tiếp theo)
NEGATION_WORDS = {
    'không', 'chẳng', 'chưa', 'đừng', 'chớ', 'ko', 'k', 'hông', 'hem',
    'chả', 'nào', 'đâu có', 'không hề', 'không bao giờ',
}

# Từ tăng cường (tăng trọng số lên 1.5x)
INTENSIFIERS = {
    'rất', 'quá', 'cực', 'siêu', 'vô cùng', 'cực kỳ', 'cực kì',
    'hết sức', 'lắm', 'thật sự', 'thực sự', 'hoàn toàn', 'đặc biệt',
    'tuyệt đối', 'hẳn',
}

# Từ đơn tích cực
POSITIVE_WORDS = {
    # Đánh giá chung
    'tốt', 'hay', 'đẹp', 'tuyệt', 'xuất sắc', 'hài lòng', 'thích',
    'yêu', 'hoàn hảo', 'chất lượng', 'ổn', 'được', 'ok', 'oke',
    'đúng', 'chuẩn', 'xịn', 'ngon', 'mượt', 'mịn', 'ưng', 'thỏa',
    'tuyệt vời', 'tốt lắm', 'hay lắm', 'đỉnh', 'chất', 'ngầu',
    # Giao hàng / đóng gói
    'nhanh', 'kịp thời', 'an toàn', 'cẩn thận', 'gọn', 'sạch', 'mới',
    'nguyên vẹn', 'đầy đủ', 'kín đáo', 'bảo quản', 'chắc chắn',
    # Nội dung sách
    'bổ ích', 'thú vị', 'hấp dẫn', 'cuốn hút', 'dễ hiểu', 'rõ ràng',
    'chi tiết', 'sâu sắc', 'phong phú', 'đa dạng', 'sinh động',
    'bổ sung', 'học được', 'hiểu', 'kiến thức', 'sáng tạo', 'mới mẻ',
    'giá trị', 'ý nghĩa', 'hay ho', 'tuyệt hay', 'đáng đọc',
    # Giá cả / giá trị
    'rẻ', 'hợp lý', 'tiết kiệm', 'đáng', 'xứng đáng', 'đáng tiền',
    'đáng mua', 'khuyến mãi', 'ưu đãi', 'giảm giá', 'tốt giá',
    # Cảm xúc tích cực
    'vui', 'hạnh phúc', 'thỏa mãn', 'ưng ý', 'hài', 'hài lòng',
    'recommend', 'giới thiệu', 'ủng hộ', 'mua lại', 'mua thêm',
    # Chất lượng in ấn
    'sắc nét', 'nét', 'rõ', 'sắc', 'sáng', 'đẹp mắt', 'bắt mắt',
    'tỉ mỉ', 'kỹ lưỡng', 'chuyên nghiệp', 'uy tín', 'tin tưởng',
    # Từ lóng / internet
    'oke', 'ok', 'good', 'great', 'nice', 'perfect', 'excellent',
    'amazing', 'wow', 'tuyệt quá', 'quá tốt', 'quá hay', 'quá đẹp',
    # Seller / shop
    'nhiệt tình', 'thân thiện', 'hỗ trợ', 'chu đáo', 'tận tâm',
    'phục vụ', 'trân trọng', 'cảm ơn', 'tốt bụng',
    # Bổ sung
    'thật', 'thực', 'chính hãng', 'hàng thật', 'hàng chính hãng',
    'đúng mô tả', 'như mô tả', 'như hình', 'đúng như',
    'dày', 'dày dặn', 'to', 'lớn', 'nhiều trang', 'đủ trang',
}

# Từ đơn tiêu cực
NEGATIVE_WORDS = {
    # Đánh giá chung
    'tệ', 'kém', 'xấu', 'tồi', 'thất vọng', 'buồn', 'chán', 'dở',
    'không ổn', 'không được', 'tệ hại', 'quá tệ', 'quá kém',
    # Lỗi sản phẩm
    'lỗi', 'hỏng', 'vỡ', 'rách', 'nát', 'bẩn', 'bụi', 'cũ', 'cong',
    'méo', 'ẩm', 'ướt', 'ố vàng', 'nhòa', 'nhòe', 'mờ', 'phai',
    'nhạt', 'lem', 'in xấu', 'in kém', 'nhăn', 'gấp', 'quăn',
    'móp', 'méo mó', 'hư', 'hư hỏng', 'khuyết tật', 'giả',
    # Giao hàng kém
    'chậm', 'muộn', 'trễ', 'lâu', 'thiếu', 'sai', 'nhầm', 'mất',
    'thất lạc', 'không giao', 'chưa nhận', 'chờ lâu', 'chờ mãi',
    # Nội dung sách
    'khó hiểu', 'phức tạp', 'khô khan', 'nhàm chán', 'lặp',
    'sai sót', 'lỗi dịch', 'dịch kém', 'dịch tệ', 'dịch sai',
    'không hay', 'không bổ ích', 'vô nghĩa', 'không đáng',
    # Giá cả / giá trị
    'đắt', 'mắc', 'lãng phí', 'không đáng tiền', 'không xứng',
    'quá đắt', 'mắc quá', 'đắt quá', 'chặt chém', 'phí tiền', 'phí',
    # Cảm xúc tiêu cực
    'tiếc', 'hối hận', 'không nên mua', 'không mua lại', 'cảnh báo',
    # Lừa đảo / giả mạo
    'nhái', 'fake', 'lừa', 'lừa đảo', 'gian lận', 'không chính hãng',
    'hàng giả', 'hàng nhái', 'hàng kém chất lượng',
    # Từ lóng tiêu cực
    'terrible', 'bad', 'awful', 'horrible', 'poor',
    # In ấn kém
    'khó đọc', 'in mờ', 'chữ mờ', 'chữ nhỏ', 'chữ nhòe',
    'không sắc nét', 'không rõ ràng', 'không rõ',
}

# Cụm từ tích cực (2 từ)
POSITIVE_BIGRAMS = {
    'giao nhanh', 'ship nhanh', 'đóng gói tốt', 'đóng gói đẹp',
    'seller tốt', 'shop tốt', 'phục vụ tốt', 'đóng gói cẩn thận',
    'đóng gói kỹ', 'hàng đẹp', 'sách đẹp', 'chất lượng tốt',
    'giá rẻ', 'giá hợp lý', 'đáng tiền', 'đúng hẹn', 'giao đúng',
    'hàng thật', 'sách hay', 'rất hay', 'rất tốt', 'rất đẹp',
    'nhanh chóng', 'an toàn', 'cẩn thận', 'nguyên vẹn', 'đầy đủ',
    'bổ ích', 'thú vị', 'hấp dẫn', 'đáng đọc', 'hay lắm',
    'tốt lắm', 'đẹp lắm', 'mua lại', 'mua thêm', 'giới thiệu',
}

# Cụm từ tiêu cực (2 từ)
NEGATIVE_BIGRAMS = {
    'in mờ', 'in xấu', 'in nhòe', 'giấy mỏng', 'giấy xấu',
    'không hay', 'không tốt', 'không đẹp', 'không hài lòng',
    'không đáng', 'lãng phí', 'hàng lỗi', 'hàng hỏng', 'hàng giả',
    'hàng nhái', 'giao chậm', 'ship chậm', 'chờ lâu', 'chờ mãi',
    'bị rách', 'bị nhòe', 'bị hỏng', 'thất vọng', 'không recommend',
    'không mua lại', 'không mua', 'đắt quá', 'mắc quá', 'phí tiền',
    'thiếu hàng', 'sai hàng', 'nhầm hàng', 'lỗi chính tả',
    'dịch sai', 'dịch kém', 'dịch tệ', 'khó hiểu', 'khô khan',
}


# ==================== FUNCTIONS ====================

def tokenize(text):
    """Tách từ đơn giản — chuyển thường, xóa ký tự đặc biệt."""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()


def score_comment(text):
    """
    Tính điểm cảm xúc cho một bình luận.
    Trả về float trong [-1, 1], hoặc None nếu không có nội dung.
    """
    if not text or not str(text).strip():
        return None

    tokens = tokenize(text)
    n = len(tokens)
    if n == 0:
        return None

    pos_score = 0.0
    neg_score = 0.0

    for i, token in enumerate(tokens):
        # Kiểm tra từ phủ định trong 3 từ trước
        window = tokens[max(0, i - 3):i]
        negated = any(t in NEGATION_WORDS for t in window)

        # Kiểm tra từ tăng cường trong 2 từ trước
        intensified = any(t in INTENSIFIERS for t in tokens[max(0, i - 2):i])
        weight = 1.5 if intensified else 1.0

        if token in POSITIVE_WORDS:
            if negated:
                neg_score += weight        # "không tốt" → tiêu cực
            else:
                pos_score += weight
        elif token in NEGATIVE_WORDS:
            if negated:
                pos_score += weight * 0.5  # "không tệ" → nhẹ tích cực
            else:
                neg_score += weight

    # Kiểm tra bigrams
    for i in range(n - 1):
        bigram = f"{tokens[i]} {tokens[i + 1]}"
        if bigram in POSITIVE_BIGRAMS:
            pos_score += 1.0
        elif bigram in NEGATIVE_BIGRAMS:
            neg_score += 1.0

    total = pos_score + neg_score
    if total == 0:
        return 0.0  # Không có từ cảm xúc → trung tính

    return (pos_score - neg_score) / total  # [-1, 1]


def compute_product_sentiment(df_reviews, df_labeled):
    """Tính sentiment_score cho từng sản phẩm, merge vào df_labeled."""
    print("\n📊 Tính sentiment score cho từng bình luận...")

    df_reviews = df_reviews.copy()
    df_reviews['comment_text'] = df_reviews['comment_text'].fillna('')

    # Tính điểm từng bình luận — có progress nếu có tqdm
    try:
        from tqdm import tqdm
        tqdm.pandas(desc="   Scoring")
        df_reviews['comment_sentiment'] = df_reviews['comment_text'].progress_apply(score_comment)
    except ImportError:
        total = len(df_reviews)
        scores = []
        for idx, text in enumerate(df_reviews['comment_text']):
            scores.append(score_comment(text))
            if (idx + 1) % 10000 == 0 or (idx + 1) == total:
                print(f"   → {idx + 1:,}/{total:,} bình luận đã xử lý")
        df_reviews['comment_sentiment'] = scores

    # Thống kê comment có/không có điểm
    has_score = df_reviews['comment_sentiment'].notna().sum()
    no_text = df_reviews['comment_sentiment'].isna().sum()
    print(f"   ✅ {has_score:,} bình luận có điểm | {no_text:,} bình luận rỗng")

    # Aggregate theo product_id — trung bình các điểm hợp lệ
    print("\n📊 Tổng hợp sentiment theo sản phẩm...")
    product_sentiment = (
        df_reviews.groupby('product_id')['comment_sentiment']
        .apply(lambda x: x.dropna().mean() if x.dropna().size > 0 else np.nan)
        .reset_index()
        .rename(columns={'comment_sentiment': 'sentiment_score'})
    )

    # Merge vào labeled data
    df_out = df_labeled.merge(product_sentiment, on='product_id', how='left')

    # Fallback cho sản phẩm không có bình luận
    no_sentiment_mask = df_out['sentiment_score'].isna()
    n_no_sentiment = no_sentiment_mask.sum()
    if n_no_sentiment > 0:
        # Chỉ dùng rating khi > 0: (rating 1-5) → (-1, 1)
        # avg_comment_rating = 0 nghĩa là không có đánh giá → trung tính = 0.0
        has_rating_mask = no_sentiment_mask & (df_out['avg_comment_rating'] > 0)
        no_rating_mask  = no_sentiment_mask & (df_out['avg_comment_rating'] == 0)

        if has_rating_mask.sum() > 0:
            fallback = ((df_out.loc[has_rating_mask, 'avg_comment_rating'] - 3) / 2).clip(-1, 1)
            df_out.loc[has_rating_mask, 'sentiment_score'] = fallback

        df_out.loc[no_rating_mask, 'sentiment_score'] = 0.0

        print(f"   ✅ Fallback rating: {has_rating_mask.sum():,} sản phẩm "
              f"| Trung tính (rating=0): {no_rating_mask.sum():,} sản phẩm")

    return df_out


def print_statistics(df_out):
    """In thống kê kết quả sentiment."""
    print("\n" + "=" * 60)
    print("📊 THỐNG KÊ SENTIMENT SCORE")
    print("=" * 60)
    s = df_out['sentiment_score']
    print(f"Tổng sản phẩm : {len(df_out):,}")
    print(f"Min            : {s.min():.4f}")
    print(f"Max            : {s.max():.4f}")
    print(f"Mean           : {s.mean():.4f}")
    print(f"Std            : {s.std():.4f}")
    print(f"Median         : {s.median():.4f}")

    pos = (s > 0.1).sum()
    neg = (s < -0.1).sum()
    neu = len(df_out) - pos - neg
    print(f"\nPhân loại:")
    print(f"  Tích cực (> 0.1) : {pos:,} sản phẩm ({100 * pos / len(df_out):.1f}%)")
    print(f"  Trung tính       : {neu:,} sản phẩm ({100 * neu / len(df_out):.1f}%)")
    print(f"  Tiêu cực (< -0.1): {neg:,} sản phẩm ({100 * neg / len(df_out):.1f}%)")

    print(f"\nSentiment trung bình theo nhãn:")
    if 'product_label' in df_out.columns:
        label_sentiment = df_out.groupby('product_label')['sentiment_score'].mean().sort_values(ascending=False)
        for label, score in label_sentiment.items():
            print(f"  {label:<30} : {score:.4f}")

    print("=" * 60)


def main():
    print(f"\n🚀 Tiki Book Intelligence - Sentiment đơn giản (từ điển tiếng Việt)")
    print(f"   Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Từ tích cực: {len(POSITIVE_WORDS)} | Cụm tích cực: {len(POSITIVE_BIGRAMS)}")
    print(f"   Từ tiêu cực: {len(NEGATIVE_WORDS)} | Cụm tiêu cực: {len(NEGATIVE_BIGRAMS)}")

    # Kiểm tra file đầu vào
    if not os.path.exists(REVIEWS_PATH):
        print(f"❌ File {REVIEWS_PATH} không tồn tại")
        print(f"   Chạy: python crawl_tiki_books.py")
        exit(1)

    if not os.path.exists(LABELED_PATH):
        print(f"❌ File {LABELED_PATH} không tồn tại")
        print(f"   Chạy: python label_products.py")
        exit(1)

    # Đọc dữ liệu
    print(f"\n📂 Đọc dữ liệu...")
    df_reviews = pd.read_csv(REVIEWS_PATH)
    df_labeled = pd.read_csv(LABELED_PATH)
    print(f"   ✅ Reviews: {len(df_reviews):,} dòng ({df_reviews['product_id'].nunique():,} sản phẩm)")
    print(f"   ✅ Labeled: {len(df_labeled):,} sản phẩm")

    # Tính sentiment và merge
    df_out = compute_product_sentiment(df_reviews, df_labeled)

    # In thống kê
    print_statistics(df_out)

    # Lưu file
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Lưu: {OUTPUT_PATH}")
    print(f"   {len(df_out):,} sản phẩm | {len(df_out.columns)} cột")
    print(f"   Cột mới: sentiment_score")
    print(f"\n⏩ Bước tiếp theo: python train_product_classifier.py")
    print(f"   (sẽ đọc từ {OUTPUT_PATH} và thêm sentiment_score vào features)")


if __name__ == "__main__":
    main()
