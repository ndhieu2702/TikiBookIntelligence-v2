import os
import random
import time
import requests
import pandas as pd
from datetime import datetime

DATA_PATH = "data/tiki_books_reviews.csv"
BACKUP_PATH = "data/tiki_books_reviews_backup_before_update_sold_count.csv"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://tiki.vn/",
}
TIMEOUT = 10


def parse_sold_count(value):
    """Chuyển giá trị sold_count sang số nguyên."""
    if value is None:
        return 0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0
    if isinstance(value, dict):
        if value.get("value") is not None:
            return parse_sold_count(value.get("value"))
        if value.get("text") is not None:
            return parse_sold_count(value.get("text"))
        return 0

    text = str(value).strip().lower()
    if not text:
        return 0

    text = text.replace("đã bán", "").replace("sold", "").strip()
    text = text.replace(" ", " ").strip()

    import re
    match = re.search(r"([0-9]+(?:[\.,][0-9]+)?)([km]?)", text)
    if not match:
        return 0

    number_str, suffix = match.groups()
    number_str = number_str.replace(",", ".")
    try:
        number = float(number_str)
    except ValueError:
        return 0

    if suffix == "k":
        number *= 1_000
    elif suffix == "m":
        number *= 1_000_000

    return int(round(number))


def extract_sold_count(product_json):
    """Lấy sold_count từ JSON chi tiết sản phẩm."""
    if not isinstance(product_json, dict):
        return 0

    quantity_sold = product_json.get("quantity_sold")
    all_time_quantity_sold = product_json.get("all_time_quantity_sold")
    sold_count = product_json.get("sold_count")
    order_count = product_json.get("order_count")

    if quantity_sold is not None:
        return parse_sold_count(quantity_sold)
    if all_time_quantity_sold is not None:
        return parse_sold_count(all_time_quantity_sold)
    if sold_count is not None:
        return parse_sold_count(sold_count)
    if order_count is not None:
        return parse_sold_count(order_count)
    return 0


def fetch_product_detail(product_id):
    """Gọi API chi tiết sản phẩm Tiki."""
    url = f"https://tiki.vn/api/v2/products/{product_id}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        print(f"  ⚠️ Không lấy được detail product_id={product_id}: {exc}")
        return None
    except ValueError:
        print(f"  ⚠️ JSON không hợp lệ cho product_id={product_id}")
        return None


def normalize_product_id(product_id):
    if pd.isna(product_id):
        return ""
    try:
        return str(int(float(product_id)))
    except (ValueError, TypeError):
        return str(product_id).strip()


def main():
    print(f"\n🚀 Cập nhật sold_count từ API Tiki")
    print(f"   Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not os.path.exists(DATA_PATH):
        print(f"❌ File không tồn tại: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    df["product_id_str"] = df["product_id"].apply(normalize_product_id)
    unique_ids = df["product_id_str"].dropna().unique().tolist()

    print(f"📦 Tổng số product_id duy nhất: {len(unique_ids)}")

    product_sold_map = {}
    updated_count = 0
    skipped_count = 0

    for idx, pid in enumerate(unique_ids, start=1):
        if not pid:
            skipped_count += 1
            continue

        print(f"[{idx}/{len(unique_ids)}] Lấy sold_count cho product_id={pid}")
        product_json = fetch_product_detail(pid)
        if product_json is None:
            skipped_count += 1
            time.sleep(random.uniform(0.2, 0.5))
            continue

        sold_count = extract_sold_count(product_json)
        if sold_count > 0:
            product_sold_map[pid] = sold_count
            updated_count += 1
        else:
            skipped_count += 1

        time.sleep(random.uniform(0.2, 0.5))

    if os.path.exists(BACKUP_PATH):
        os.remove(BACKUP_PATH)
    df.to_csv(BACKUP_PATH, index=False)
    print(f"✅ Backup file cũ: {BACKUP_PATH}")

    replaced = 0
    for pid, sold_count in product_sold_map.items():
        mask = df["product_id_str"] == pid
        if mask.any():
            before_zero = (df.loc[mask, "sold_count"] == 0).sum()
            df.loc[mask, "sold_count"] = sold_count
            replaced += before_zero

    df.drop(columns=["product_id_str"], inplace=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Lưu dữ liệu đã cập nhật: {DATA_PATH}")

    sold_after = pd.to_numeric(df["sold_count"], errors="coerce").fillna(0).astype(int)
    print("\n📊 Thống kê sau cập nhật:")
    print(f"   Tổng product_id duy nhất: {len(unique_ids)}")
    print(f"   Product_id cập nhật được sold_count>0: {len(product_sold_map)}")
    print(f"   Product_id không lấy được sold_count>0: {skipped_count}")
    print(f"   sold_count min: {sold_after.min():,}")
    print(f"   sold_count max: {sold_after.max():,}")
    print(f"   sold_count mean: {sold_after.mean():,.2f}")

    sample = df[df["sold_count"] > 0].drop_duplicates(subset=["product_id"]).head(10)
    if not sample.empty:
        print("\n   Ví dụ sold_count > 0:")
        for _, row in sample.iterrows():
            print(f"     product_id={row['product_id']} sold_count={int(row['sold_count'])}")


if __name__ == "__main__":
    main()
