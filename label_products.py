"""
Gán nhãn sản phẩm dựa trên dữ liệu crawl.
- Tạo các đặc trưng cấp sản phẩm.
- Gán nhãn 5 lớp: Best Seller, High Potential, Premium, Normal, Needs Improvement.
"""

import os
import re
import pandas as pd
from datetime import datetime

DATA_PATH = "data/tiki_books_reviews.csv"
OUTPUT_PATH = "data/tiki_books_labeled.csv"

# ==================== UTILITY FUNCTIONS ====================
def load_data():
    """Đọc dữ liệu từ file."""
    if not os.path.exists(DATA_PATH):
        print(f"❌ File {DATA_PATH} không tồn tại")
        print(f"   Hãy chạy: python crawl_tiki_books.py")
        exit(1)
    
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Đã đọc {len(df)} dòng từ {DATA_PATH}")
    return df

def parse_numeric_value(value):
    """Chuyển các giá trị price/sold_count sang số nguyên hợp lệ."""
    if pd.isna(value):
        return 0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    text = str(value).strip()
    if not text:
        return 0

    # Nếu là chuỗi dict hoặc JSON-like, cố gắng lấy số đầu tiên
    match = re.search(r"([0-9]+(?:[\.,][0-9]+)?)([kKmM]?)", text)
    if not match:
        return 0

    number_str, suffix = match.groups()
    number_str = number_str.replace(',', '.')
    try:
        number = float(number_str)
    except ValueError:
        return 0

    if suffix.lower() == 'k':
        number *= 1_000
    elif suffix.lower() == 'm':
        number *= 1_000_000

    return int(round(number))


def clean_data(df):
    """Làm sạch dữ liệu."""
    print("\n🧹 Làm sạch dữ liệu...")
    
    # Xử lý giá và số lượng bán trước khi numeric
    df['price'] = df['price'].apply(parse_numeric_value)
    df['sold_count'] = df['sold_count'].apply(parse_numeric_value)

    # Chuyển các cột số về numeric
    numeric_cols = ['price', 'rating', 'review_count', 'sold_count', 'comment_rating']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['sold_count'] = df['sold_count'].astype(float).round().astype(int)
    df['price'] = df['price'].astype(float).fillna(0)

    # Xử lý cột text
    df['comment_text'] = df['comment_text'].fillna('')
    df['product_url'] = df['product_url'].fillna('')
    
    print(f"   ✓ Chuyển đổi cột số")
    print(f"   ✓ Làm sạch cột text")
    
    return df

def create_product_features(df):
    """Tạo đặc trưng cấp sản phẩm."""
    print("\n📊 Tạo đặc trưng cấp sản phẩm...")
    
    # Gom theo product_id và product_name
    grouped = df.groupby(['product_id', 'product_name'], as_index=False).agg({
        'price': 'first',
        'rating': 'first',
        'review_count': 'max',
        'sold_count': 'max',
        'comment_text': lambda x: sum(1 for t in x if t.strip()),  # comment_count
        'comment_rating': 'mean',  # avg_comment_rating
        'product_url': 'first'
    })
    
    # Đổi tên cột
    grouped.rename(columns={'comment_text': 'comment_count', 'comment_rating': 'avg_comment_rating'}, inplace=True)
    
    # Tính toán tỷ lệ
    def calculate_ratios(df_group):
        """Tính tỷ lệ review theo rating."""
        comment_ratings = df_group['comment_rating'].values
        
        if len(comment_ratings) == 0:
            return 0, 0, 0
        
        total = len(comment_ratings)
        positive = sum(1 for r in comment_ratings if r >= 4)
        neutral = sum(1 for r in comment_ratings if r == 3)
        negative = sum(1 for r in comment_ratings if 0 < r < 3)
        
        return (positive / total if total > 0 else 0,
                neutral / total if total > 0 else 0,
                negative / total if total > 0 else 0)
    
    # Tính tỷ lệ cho mỗi sản phẩm
    ratios = []
    for product_id in grouped['product_id']:
        product_reviews = df[df['product_id'] == product_id]
        pos_ratio, neu_ratio, neg_ratio = calculate_ratios(product_reviews)
        ratios.append((pos_ratio, neu_ratio, neg_ratio))
    
    grouped[['positive_ratio', 'neutral_ratio', 'negative_ratio']] = pd.DataFrame(ratios, index=grouped.index)
    
    # Tính doanh thu ước tính
    grouped['estimated_revenue'] = grouped['price'] * grouped['sold_count']
    
    # Xử lý avg_comment_rating
    grouped['avg_comment_rating'] = grouped['avg_comment_rating'].fillna(0)

    print(f"   ✓ Gom theo sản phẩm: {len(grouped)} sản phẩm duy nhất")
    print(f"   - price: min={grouped['price'].min():,.0f}, max={grouped['price'].max():,.0f}, mean={grouped['price'].mean():,.0f}")
    print(f"   - sold_count: min={grouped['sold_count'].min():,.0f}, max={grouped['sold_count'].max():,.0f}, mean={grouped['sold_count'].mean():,.0f}")
    print(f"   - estimated_revenue: min={grouped['estimated_revenue'].min():,.0f}, max={grouped['estimated_revenue'].max():,.0f}, mean={grouped['estimated_revenue'].mean():,.0f}, sum={grouped['estimated_revenue'].sum():,.0f}")
    if grouped['estimated_revenue'].sum() == 0:
        print("   ⚠️ estimated_revenue toàn bộ bằng 0, kiểm tra lại price và sold_count")
    
    return grouped

def assign_labels(df):
    """Gán nhãn sản phẩm bằng HDBSCAN clustering."""
    try:
        import hdbscan as hdbscan_lib
    except ImportError:
        print("❌ Chưa cài hdbscan. Chạy: pip install hdbscan")
        exit(1)

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import euclidean_distances

    # Features dùng để phân cụm (nhóm gán nhãn — tách biệt với nhóm train model)
    CLUSTER_FEATURES = ['sold_count', 'estimated_revenue', 'rating', 'positive_ratio', 'negative_ratio']
    LABEL_ORDER = ["Best Seller", "Premium / Niche Quality", "High Potential", "Normal", "Needs Improvement"]

    print(f"\n🔬 Phân cụm HDBSCAN với {len(df)} sản phẩm...")
    print(f"   Features: {CLUSTER_FEATURES}")

    df = df.copy()

    # Chuẩn bị ma trận X — lưu lại để tính stats trên dữ liệu gốc
    X = df[CLUSTER_FEATURES].fillna(0).values

    # Chuẩn hóa bằng StandardScaler trước khi đưa vào HDBSCAN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # HDBSCAN clustering
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=50,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom',
    )
    raw_labels = clusterer.fit_predict(X_scaled)

    unique_clusters = sorted(set(raw_labels) - {-1})
    n_noise = int((raw_labels == -1).sum())

    print(f"   ✓ Số cụm tìm được: {len(unique_clusters)}")
    print(f"   ✓ Outlier (noise): {n_noise} ({100 * n_noise / len(df):.1f}%)")

    if not unique_clusters:
        print("❌ HDBSCAN không tìm được cụm nào. Kiểm tra lại dữ liệu.")
        exit(1)

    # Gán outlier về cụm gần nhất theo khoảng cách Euclidean đến centroid
    cluster_assignments = raw_labels.copy()
    if n_noise > 0:
        centroids = np.array([X_scaled[raw_labels == c].mean(axis=0) for c in unique_clusters])
        noise_idx = np.where(raw_labels == -1)[0]
        dist = euclidean_distances(X_scaled[noise_idx], centroids)
        cluster_assignments[noise_idx] = [unique_clusters[int(i)] for i in np.argmin(dist, axis=1)]
        print(f"   ✓ Đã gán {n_noise} outlier vào cụm gần nhất")

    # Tính đặc trưng trung bình mỗi cụm trên dữ liệu gốc (trước khi scale)
    feat_idx = {feat: i for i, feat in enumerate(CLUSTER_FEATURES)}
    cluster_stats = {}
    for c in unique_clusters:
        mask = cluster_assignments == c
        sub = X[mask]
        cluster_stats[c] = {
            'n':      int(mask.sum()),
            'sold':   float(sub[:, feat_idx['sold_count']].mean()),
            'rev':    float(sub[:, feat_idx['estimated_revenue']].mean()),
            'rating': float(sub[:, feat_idx['rating']].mean()),
            'pos':    float(sub[:, feat_idx['positive_ratio']].mean()),
            'neg':    float(sub[:, feat_idx['negative_ratio']].mean()),
        }

    # Tính điểm tổng hợp cho mỗi cụm bằng minmax chuẩn hóa giữa các cụm
    def _norm(val, vals):
        mn, mx = min(vals), max(vals)
        return (val - mn) / (mx - mn) if mx > mn else 0.5

    sold_v   = [s['sold']   for s in cluster_stats.values()]
    rev_v    = [s['rev']    for s in cluster_stats.values()]
    rating_v = [s['rating'] for s in cluster_stats.values()]
    pos_v    = [s['pos']    for s in cluster_stats.values()]
    neg_v    = [s['neg']    for s in cluster_stats.values()]

    for c, s in cluster_stats.items():
        cluster_stats[c]['score'] = (
            0.30 * _norm(s['sold'],   sold_v) +
            0.30 * _norm(s['rev'],    rev_v) +
            0.20 * _norm(s['rating'], rating_v) +
            0.10 * _norm(s['pos'],    pos_v) -
            0.10 * _norm(s['neg'],    neg_v)
        )

    # Sắp xếp cụm theo điểm giảm dần (cụm 0 = tốt nhất)
    sorted_clusters = sorted(unique_clusters, key=lambda c: cluster_stats[c]['score'], reverse=True)
    n = len(sorted_clusters)

    # Map cụm → nhãn dựa trên đặc trưng và thứ hạng
    cluster_to_label = {}
    for rank, c in enumerate(sorted_clusters):
        s = cluster_stats[c]
        pct = rank / max(n - 1, 1)  # 0.0 = tốt nhất, 1.0 = kém nhất

        # Cụm có negative cao hoặc rating thấp → Needs Improvement
        if s['neg'] >= 0.35 or (0 < s['rating'] < 3.5):
            cluster_to_label[c] = "Needs Improvement"
        # Cụm tốt nhất (rank đầu) → Best Seller
        elif rank == 0:
            cluster_to_label[c] = "Best Seller"
        # Cụm chất lượng: rating cao + positive cao → Premium / Niche Quality
        elif s['rating'] >= 4.0 and s['pos'] >= 0.5:
            cluster_to_label[c] = "Premium / Niche Quality"
        # Cụm trung bình khá → High Potential
        elif pct < 0.5:
            cluster_to_label[c] = "High Potential"
        # Còn lại → Normal
        else:
            cluster_to_label[c] = "Normal"

    # In bảng mapping cụm → nhãn
    print("\n   📊 Bảng mapping cụm → nhãn:")
    print(f"   {'Cụm':>4} {'n':>6} {'score':>6}  {'sold':>8} {'rev':>13} {'rating':>7} {'pos':>5} {'neg':>5}  Nhãn")
    print(f"   {'-'*80}")
    for c in sorted_clusters:
        s = cluster_stats[c]
        print(
            f"   {c:4d} {s['n']:6d} {s['score']:6.3f}  {s['sold']:8.0f} "
            f"{s['rev']:13.0f} {s['rating']:7.2f} {s['pos']:5.2f} {s['neg']:5.2f}"
            f"  {cluster_to_label[c]}"
        )

    # Kiểm tra nhãn bị thiếu
    used_labels = set(cluster_to_label.values())
    missing = [l for l in LABEL_ORDER if l not in used_labels]
    if missing:
        print(f"\n   ⚠️ Nhãn chưa được gán từ clustering: {missing}")

    # Gán nhãn cho từng sản phẩm
    df['product_label'] = [cluster_to_label[cluster_assignments[i]] for i in range(len(df))]

    # Fallback per-product nếu còn nhãn thiếu
    if missing:
        print(f"   → Áp dụng fallback per-product...")
        if 'High Potential' in missing:
            # Lấy top 20% Normal theo sold_count làm High Potential
            normal_sold = df.loc[df['product_label'] == 'Normal', 'sold_count']
            if len(normal_sold) > 0 and normal_sold.max() > 0:
                thr = normal_sold.quantile(0.80)
                hp_mask = (df['product_label'] == 'Normal') & (df['sold_count'] >= thr) & (df['sold_count'] > 0)
                df.loc[hp_mask, 'product_label'] = 'High Potential'
                print(f"   ✓ Fallback High Potential: {hp_mask.sum()} sản phẩm")

        if 'Needs Improvement' in missing:
            ni_mask = (df['product_label'] == 'Normal') & (
                ((df['rating'] > 0) & (df['rating'] < 3.5)) | (df['negative_ratio'] >= 0.35)
            )
            df.loc[ni_mask, 'product_label'] = 'Needs Improvement'
            print(f"   ✓ Fallback Needs Improvement: {ni_mask.sum()} sản phẩm")

        if 'Best Seller' in missing:
            bs_mask = (df['product_label'] == 'Premium / Niche Quality') & (
                df['sold_count'] >= df['sold_count'].quantile(0.90)
            )
            df.loc[bs_mask, 'product_label'] = 'Best Seller'
            print(f"   ✓ Fallback Best Seller: {bs_mask.sum()} sản phẩm")

        if 'Premium / Niche Quality' in missing:
            pm_mask = (df['product_label'] == 'High Potential') & (
                (df['rating'] >= 4.0) & (df['positive_ratio'] >= 0.6)
            )
            df.loc[pm_mask, 'product_label'] = 'Premium / Niche Quality'
            print(f"   ✓ Fallback Premium: {pm_mask.sum()} sản phẩm")

    return df

def print_statistics(df):
    """In thống kê."""
    print("\n" + "="*60)
    print("📊 THỐNG KÊ GÁN NHÃN")
    print("="*60)
    print(f"Tổng số sản phẩm: {len(df)}")
    
    label_order = [
        "Best Seller",
        "High Potential",
        "Premium / Niche Quality",
        "Normal",
        "Needs Improvement"
    ]
    label_counts = df['product_label'].value_counts()
    present_labels = 0
    for label in label_order:
        count = int(label_counts.get(label, 0))
        print(f"  {label}: {count}")
        if count > 0:
            present_labels += 1
    
    total_revenue = df['estimated_revenue'].sum()
    print(f"\nTổng doanh thu ước tính: {total_revenue:,.0f} VND")
    
    has_comment = len(df[df['comment_count'] > 0])
    no_comment = len(df[df['comment_count'] == 0])
    print(f"\nCó bình luận: {has_comment} sản phẩm")
    print(f"Không bình luận: {no_comment} sản phẩm")
    
    if present_labels < 3:
        print("\n⚠️ Cảnh báo: phân bố nhãn chưa đa dạng, cần kiểm tra lại ngưỡng.")
    
    print(f"\n✅ Lưu: {OUTPUT_PATH}")
    print("="*60)

def main():
    print(f"\n🚀 Tiki Book Intelligence - Gán nhãn sản phẩm")
    print(f"   Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Đọc và làm sạch dữ liệu
    df = load_data()
    df = clean_data(df)
    
    # Tạo đặc trưng
    df_products = create_product_features(df)
    
    # Gán nhãn
    df_products = assign_labels(df_products)
    
    # In thống kê
    print_statistics(df_products)
    
    # Lưu dữ liệu
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    df_products.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n✅ Hoàn thành!")

if __name__ == "__main__":
    main()
