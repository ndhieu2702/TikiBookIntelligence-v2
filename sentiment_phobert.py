"""
Phân tích cảm xúc PhoBERT cho sản phẩm sách Tiki.
Model  : wonrax/phobert-base-vietnamese-sentiment
Input  : data/tiki_books_reviews.csv  (review-level)
         data/tiki_books_labeled.csv  (product-level, để merge nhãn)
Output : data/tiki_books_sentiment_phobert.csv (product-level sentiment)

Chạy trên Google Colab GPU T4:
    pip install transformers torch sentencepiece tqdm
    python sentiment_phobert.py
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# ==================== CẤU HÌNH ====================

REVIEWS_PATH    = "data/tiki_books_reviews.csv"
LABELED_PATH    = "data/tiki_books_labeled.csv"
OUTPUT_PATH     = "data/tiki_books_sentiment_phobert.csv"
CHECKPOINT_PATH = "data/checkpoint_phobert.csv"

MODEL_NAME        = "wonrax/phobert-base-vietnamese-sentiment"
BATCH_SIZE        = 32      # số text mỗi lần đưa vào model
CHECKPOINT_EVERY  = 1000    # lưu checkpoint sau mỗi N sản phẩm
MAX_TEXT_LENGTH   = 256     # giới hạn token (PhoBERT max = 256)


# ==================== DEVICE DETECTION ====================

def detect_device() -> str:
    """Tự detect GPU hay CPU — ưu tiên GPU nếu có."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"   ✅ GPU: {name} ({vram:.1f} GB VRAM)")
            return "cuda"
    except ImportError:
        pass
    print("   ⚠️ Không tìm thấy GPU — dùng CPU (sẽ chậm hơn nhiều)")
    return "cpu"


# ==================== LOAD MODEL ====================

def load_pipeline(device: str):
    """Tải pipeline PhoBERT từ HuggingFace."""
    from transformers import pipeline as hf_pipeline

    print(f"\n📥 Tải model: {MODEL_NAME}")
    pipe = hf_pipeline(
        "text-classification",
        model=MODEL_NAME,
        device=0 if device == "cuda" else -1,
        truncation=True,
        max_length=MAX_TEXT_LENGTH,
    )
    print("   ✅ Model đã tải xong")
    return pipe


# ==================== SENTIMENT SCORING ====================

def calc_sentiment(label: str, score: float) -> float:
    """
    Chuyển nhãn PhoBERT + confidence thành điểm [-1, 1].
      POS → +score  (ví dụ: POS 0.92 → +0.92)
      NEG → -score  (ví dụ: NEG 0.85 → -0.85)
      NEU → 0.0     (trung tính luôn = 0)
    """
    label = label.upper()
    if label == "POS":
        return float(score)
    if label == "NEG":
        return -float(score)
    return 0.0  # NEU


def score_texts(texts: list, pipe) -> list:
    """
    Chạy PhoBERT trên một danh sách texts.
    - Bỏ qua text rỗng → None
    - Xử lý lỗi batch → None cho toàn batch lỗi
    Trả về list float (hoặc None) đúng thứ tự đầu vào.
    """
    n = len(texts)
    results = [None] * n

    # Tách vị trí text hợp lệ
    valid_idx = []
    valid_texts = []
    for i, text in enumerate(texts):
        cleaned = str(text).strip() if text else ""
        if cleaned:
            valid_idx.append(i)
            valid_texts.append(cleaned[:512])  # cắt cứng phòng text cực dài

    if not valid_texts:
        return results

    # Chạy model theo sub-batch BATCH_SIZE
    outputs = []
    for start in range(0, len(valid_texts), BATCH_SIZE):
        sub = valid_texts[start : start + BATCH_SIZE]
        try:
            outs = pipe(sub, truncation=True, max_length=MAX_TEXT_LENGTH)
            outputs.extend(outs)
        except Exception as err:
            print(f"   ⚠️ Lỗi sub-batch [{start}:{start+len(sub)}]: {err}")
            outputs.extend([None] * len(sub))

    # Gán kết quả về đúng vị trí ban đầu
    for idx, out in zip(valid_idx, outputs):
        if out is not None:
            results[idx] = calc_sentiment(out["label"], out["score"])

    return results


# ==================== AGGREGATE THEO SẢN PHẨM ====================

def compute_phobert_sentiment(df_reviews: pd.DataFrame, pipe) -> pd.DataFrame:
    """
    Tính sentiment_score cho từng sản phẩm.
    - Lưu checkpoint mỗi CHECKPOINT_EVERY sản phẩm.
    - Nếu checkpoint tồn tại → tiếp tục từ điểm đã dừng.
    Trả về DataFrame cột: product_id, sentiment_score
    """
    from tqdm import tqdm

    # Nạp checkpoint nếu đã có
    if Path(CHECKPOINT_PATH).exists():
        done_df = pd.read_csv(CHECKPOINT_PATH)
        done_ids = set(done_df["product_id"].tolist())
        results = done_df.to_dict("records")
        print(f"   📂 Tiếp tục từ checkpoint: {len(done_ids):,} sản phẩm đã xong")
    else:
        done_ids = set()
        results = []

    # Danh sách sản phẩm chưa xử lý
    all_products = df_reviews["product_id"].unique().tolist()
    remaining = [p for p in all_products if p not in done_ids]
    print(f"\n📊 Cần xử lý: {len(remaining):,} sản phẩm "
          f"(tổng {len(all_products):,}, đã có {len(done_ids):,})")

    if not remaining:
        print("   ✅ Tất cả sản phẩm đã xử lý xong!")
        return pd.DataFrame(results)

    # Xử lý theo từng nhóm CHECKPOINT_EVERY sản phẩm
    for group_start in tqdm(range(0, len(remaining), CHECKPOINT_EVERY),
                            desc="Nhóm sản phẩm", unit=f"x{CHECKPOINT_EVERY}"):

        group_ids = remaining[group_start : group_start + CHECKPOINT_EVERY]
        group_reviews = df_reviews[df_reviews["product_id"].isin(group_ids)].copy()

        # Lấy tất cả comment_text của nhóm này → chạy qua PhoBERT một lần
        texts = group_reviews["comment_text"].fillna("").tolist()
        print(f"\n   🔄 Nhóm {group_start // CHECKPOINT_EVERY + 1}: "
              f"{len(group_ids):,} sản phẩm | {len(texts):,} reviews")

        comment_scores = score_texts(texts, pipe)
        group_reviews["comment_sentiment"] = comment_scores

        # Aggregate từng sản phẩm trong nhóm
        for pid in group_ids:
            pid_scores = group_reviews.loc[
                group_reviews["product_id"] == pid, "comment_sentiment"
            ].tolist()
            valid = [s for s in pid_scores if s is not None]
            sentiment = float(np.mean(valid)) if valid else None
            results.append({"product_id": pid, "sentiment_score": sentiment})

        # Lưu checkpoint
        Path(CHECKPOINT_PATH).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
        print(f"   💾 Checkpoint lưu: {len(results):,} sản phẩm")

    return pd.DataFrame(results)


# ==================== FALLBACK & MERGE ====================

def apply_fallback(df_out: pd.DataFrame) -> pd.DataFrame:
    """
    Sản phẩm không có bình luận hợp lệ → fallback:
    - Có avg_comment_rating > 0 → đổi sang thang [-1, 1]
    - Không có rating           → 0.0 (trung tính)
    """
    mask_none = df_out["sentiment_score"].isna()
    n_none = mask_none.sum()

    if n_none == 0:
        return df_out

    # Fallback bằng avg_comment_rating nếu có
    if "avg_comment_rating" in df_out.columns:
        has_rating = mask_none & (df_out["avg_comment_rating"].fillna(0) > 0)
        no_rating  = mask_none & (~has_rating)

        if has_rating.sum() > 0:
            fallback = ((df_out.loc[has_rating, "avg_comment_rating"] - 3) / 2).clip(-1, 1)
            df_out.loc[has_rating, "sentiment_score"] = fallback

        df_out.loc[no_rating, "sentiment_score"] = 0.0
        print(f"   ✅ Fallback rating: {has_rating.sum():,} | Trung tính (0.0): {no_rating.sum():,}")
    else:
        df_out.loc[mask_none, "sentiment_score"] = 0.0
        print(f"   ✅ Fallback 0.0: {n_none:,} sản phẩm không có bình luận")

    return df_out


# ==================== THỐNG KÊ ====================

def print_statistics(df_out: pd.DataFrame) -> None:
    """In thống kê kết quả sentiment PhoBERT."""
    print("\n" + "=" * 60)
    print("📊 THỐNG KÊ SENTIMENT PHOBERT")
    print("=" * 60)
    s = df_out["sentiment_score"].fillna(0)
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
    print(f"  Tích cực (> +0.1) : {pos:,} ({100 * pos / len(df_out):.1f}%)")
    print(f"  Trung tính        : {neu:,} ({100 * neu / len(df_out):.1f}%)")
    print(f"  Tiêu cực (< -0.1) : {neg:,} ({100 * neg / len(df_out):.1f}%)")

    if "product_label" in df_out.columns:
        print(f"\nSentiment trung bình theo nhãn:")
        label_sent = (
            df_out.groupby("product_label")["sentiment_score"]
            .mean()
            .sort_values(ascending=False)
        )
        for lbl, sc in label_sent.items():
            print(f"  {lbl:<30} : {sc:.4f}")

    print("=" * 60)


# ==================== MAIN ====================

def main():
    print(f"\n🚀 Tiki Book Intelligence — PhoBERT Sentiment")
    print(f"   Thời gian : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Model     : {MODEL_NAME}")
    print(f"   Batch size: {BATCH_SIZE} | Checkpoint: mỗi {CHECKPOINT_EVERY:,} sản phẩm")

    # Kiểm tra file đầu vào
    for path in (REVIEWS_PATH, LABELED_PATH):
        if not os.path.exists(path):
            print(f"❌ Không tìm thấy: {path}")
            print(f"   Chạy pipeline từ đầu: crawl → label → sentiment")
            return

    # Detect thiết bị
    print("\n🖥️  Kiểm tra thiết bị...")
    device = detect_device()

    # Tải model
    pipe = load_pipeline(device)

    # Đọc dữ liệu
    print("\n📂 Đọc dữ liệu...")
    df_reviews = pd.read_csv(REVIEWS_PATH)
    df_labeled = pd.read_csv(LABELED_PATH)
    df_reviews["comment_text"] = df_reviews["comment_text"].fillna("")

    n_with_text = (df_reviews["comment_text"].str.strip() != "").sum()
    print(f"   ✅ Reviews : {len(df_reviews):,} dòng "
          f"({df_reviews['product_id'].nunique():,} sản phẩm)")
    print(f"   ✅ Labeled : {len(df_labeled):,} sản phẩm")
    print(f"   ✅ Bình luận có nội dung: {n_with_text:,} "
          f"| Rỗng: {len(df_reviews) - n_with_text:,}")

    # Tính PhoBERT sentiment theo sản phẩm
    product_sent_df = compute_phobert_sentiment(df_reviews, pipe)

    # Merge với labeled data
    print("\n🔗 Merge với dữ liệu sản phẩm...")
    df_out = df_labeled.merge(product_sent_df, on="product_id", how="left")

    # Fallback cho sản phẩm không có bình luận
    df_out = apply_fallback(df_out)

    # In thống kê
    print_statistics(df_out)

    # Lưu output
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Đã lưu: {OUTPUT_PATH}")
    print(f"   {len(df_out):,} sản phẩm | {len(df_out.columns)} cột")
    print(f"   Cột mới: sentiment_score (PhoBERT)")

    # Xóa checkpoint sau khi hoàn thành
    if Path(CHECKPOINT_PATH).exists():
        os.remove(CHECKPOINT_PATH)
        print(f"   🗑️  Đã xóa checkpoint tạm: {CHECKPOINT_PATH}")

    print(f"\n⏩ Bước tiếp theo: python train_product_classifier.py")
    print(f"   (sẽ tự đọc {OUTPUT_PATH} và dùng sentiment_score PhoBERT)")


if __name__ == "__main__":
    main()
