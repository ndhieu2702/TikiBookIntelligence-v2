"""
Multi-keyword crawler for Tiki book data.

The crawler uses public Tiki search/review APIs only, keeps crawl progress
per keyword, and merges new rows with the existing CSV without dropping
previously collected unique data.
"""

import argparse
import json
import math
import os
import random
import re
import shutil
import time
from datetime import datetime

import pandas as pd
import requests
from tqdm import tqdm


# ==================== CONFIGURATION ====================
KEYWORDS = [
    "sách",
    "sách hay",
    "sách bán chạy",
    "sách kinh tế",
    "sách kỹ năng",
    "sách văn học",
    "sách thiếu nhi",
    "sách tiếng anh",
    "sách self help",
    "sách tư duy",
    "sách tài chính",
    "sách quản trị",
    "sách tâm lý",
    "sách marketing",
    "sách lập trình",
    "sách ngoại ngữ",
    "đắc nhân tâm",
    "nhà giả kim",
    "atomic habits",
    "muôn kiếp nhân sinh",
    "tư duy nhanh và chậm",
    "dám bị ghét",
    "cây cam ngọt của tôi",
    "đời ngắn đừng ngủ dài",
    "người giàu có nhất thành babylon",
    "cha giàu cha nghèo",
    "quẳng gánh lo đi và vui sống",
    "7 thói quen hiệu quả",
]

PRODUCTS_PER_PAGE = 50
MAX_PAGES_PER_KEYWORD = 10
MAX_REVIEWS_PER_PRODUCT = 50
REVIEWS_PER_PAGE = 10
EMPTY_PAGE_LIMIT = 3

DATA_PATH = "data/tiki_books_reviews.csv"
BACKUP_PATH = "data/tiki_books_reviews_backup.csv"
STATE_PATH = "data/crawl_state.json"

# Public Tiki API endpoints.
SEARCH_URL = "https://tiki.vn/api/v2/products"
REVIEW_URL = "https://tiki.vn/api/v2/reviews"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://tiki.vn/",
}

OUTPUT_COLUMNS = [
    "product_id",
    "product_name",
    "price",
    "rating",
    "review_count",
    "sold_count",
    "product_url",
    "search_keyword",
    "review_id",
    "comment_content",
    "comment_text",
    "comment_rating",
    "review_created_at",
]

HOT_BOOK_CHECKS = ["đắc nhân tâm", "nhà giả kim", "atomic habits"]


# ==================== UTILITY FUNCTIONS ====================
def normalize_id(value):
    """Chuẩn hóa mã ID."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def parse_sold_count(value):
    """Đổi số đã bán về số nguyên."""
    if value is None:
        return 0

    if isinstance(value, dict):
        raw_value = value.get("value")
        if raw_value not in (None, ""):
            return parse_sold_count(raw_value)
        return parse_sold_count(value.get("text"))

    try:
        if pd.isna(value):
            return 0
    except (TypeError, ValueError):
        pass

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return int(value)

    text = str(value).strip().lower()
    if not text or text in {"none", "nan", "null"}:
        return 0

    match = re.search(r"(\d+(?:[.,]\d+)?)\s*([km])?", text)
    if not match:
        return 0

    number_text = match.group(1)
    suffix = match.group(2) or ""

    if suffix in {"k", "m"}:
        number = float(number_text.replace(",", "."))
        multiplier = 1000 if suffix == "k" else 1_000_000
        return int(number * multiplier)

    digits = re.sub(r"\D", "", number_text)
    return int(digits) if digits else 0


def parse_number(value, default=0):
    """Đổi giá trị về số."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value

    text = str(value).strip().replace(",", "")
    if not text:
        return default

    try:
        number = float(text)
    except ValueError:
        return default

    return int(number) if number.is_integer() else number


def extract_sold_count(product):
    """Lấy số đã bán."""
    first_zero = 0

    for key in ("quantity_sold", "all_time_quantity_sold", "sold_count", "order_count"):
        if key not in product:
            continue
        raw_value = product.get(key)
        if raw_value in (None, "", {}):
            continue
        parsed = parse_sold_count(raw_value)
        if parsed > 0:
            return parsed
        first_zero = parsed

    quantity_sold = product.get("quantity_sold")
    if isinstance(quantity_sold, dict):
        for sub_key in ("value", "text"):
            raw_value = quantity_sold.get(sub_key)
            if raw_value in (None, ""):
                continue
            parsed = parse_sold_count(raw_value)
            if parsed > 0:
                return parsed
            first_zero = parsed

    # Existing API responses often expose this key as a compact text field.
    if product.get("sold") not in (None, ""):
        parsed = parse_sold_count(product.get("sold"))
        if parsed > 0:
            return parsed
        first_zero = parsed

    return first_zero


def make_product_url(product, product_id):
    """Tạo link sản phẩm."""
    url_path = product.get("url_path") or product.get("url_key")
    if url_path:
        url_path = str(url_path).strip()
        if url_path.startswith("http"):
            return url_path
        return f"https://tiki.vn/{url_path.lstrip('/')}"
    return f"https://tiki.vn/p/{product_id}"


def load_crawl_state():
    """Đọc trạng thái crawl."""
    if not os.path.exists(STATE_PATH):
        return {}

    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            raw_state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(raw_state, dict):
        return {}

    state = {}

    legacy_last_page = raw_state.get("last_crawled_page")
    if legacy_last_page is not None:
        state["sách"] = {"last_crawled_page": int(parse_number(legacy_last_page, 0))}

    for keyword, value in raw_state.items():
        if keyword == "last_crawled_page":
            continue
        if isinstance(value, dict):
            last_page = value.get("last_crawled_page", 0)
        else:
            last_page = value
        state[keyword] = {"last_crawled_page": int(parse_number(last_page, 0))}

    return state


def save_crawl_state(state):
    """Lưu trạng thái crawl."""
    os.makedirs(os.path.dirname(STATE_PATH) or ".", exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def get_last_crawled_page(state, keyword):
    """Lấy trang đã crawl cuối."""
    value = state.get(keyword, {})
    if isinstance(value, dict):
        return int(parse_number(value.get("last_crawled_page", 0), 0))
    return int(parse_number(value, 0))


def set_last_crawled_page(state, keyword, last_page):
    """Cập nhật trang đã crawl."""
    state[keyword] = {"last_crawled_page": int(last_page)}


def make_request(url, params=None, timeout=15):
    """Gọi API Tiki."""
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print(f"  Timeout: {url} params={params}")
        return None
    except requests.exceptions.RequestException as exc:
        print(f"  Request error: {exc} url={url} params={params}")
        return None
    except json.JSONDecodeError:
        print(f"  JSON decode error: {url} params={params}")
        return None


def get_reviews(product_id, max_reviews=MAX_REVIEWS_PER_PRODUCT, exclude_review_ids=None):
    """Lấy review sản phẩm, bỏ qua review đã có theo review_id."""
    reviews = []
    if max_reviews <= 0:
        return reviews

    exclude = exclude_review_ids or set()
    max_pages = math.ceil(max_reviews / REVIEWS_PER_PAGE)

    for review_page in range(1, max_pages + 1):
        remaining = max_reviews - len(reviews)
        params = {
            "limit": min(REVIEWS_PER_PAGE, remaining),
            "include": "comments,contribute_info,attribute_vote_summary",
            "sort": "score|desc,id|desc,stars|all",
            "page": review_page,
            "product_id": product_id,
        }
        data = make_request(REVIEW_URL, params=params)
        if not data:
            break

        review_list = data.get("data", [])
        if not review_list:
            break

        for review in review_list:
            rid = normalize_id(review.get("id") or review.get("review_id"))
            if rid and rid in exclude:
                continue  # bỏ qua review đã crawl
            content = (review.get("content") or "").strip()
            reviews.append(
                {
                    "review_id": rid,
                    "comment_content": content,
                    "comment_text": content,
                    "comment_rating": parse_number(review.get("rating", 0), 0),
                    "review_created_at": review.get("created_at") or "",
                }
            )
            if len(reviews) >= max_reviews:
                break

        time.sleep(random.uniform(0.25, 0.6))

    return reviews


def build_product_rows(product, keyword, max_reviews, existing_reviews_by_product=None):
    """Tạo dòng dữ liệu sản phẩm."""
    product_id = normalize_id(product.get("id"))
    if not product_id:
        return []

    product_name = (product.get("name") or "").strip()
    price = parse_number(product.get("price", 0), 0)
    rating = parse_number(product.get("rating_average", 0), 0)
    review_count = int(parse_number(product.get("review_count", 0), 0))
    sold_count = extract_sold_count(product)
    product_url = make_product_url(product, product_id)

    reviews = []
    if review_count > 0:
        # Lấy set review_id đã có của sản phẩm này để lọc trùng
        exclude = set()
        if existing_reviews_by_product:
            exclude = existing_reviews_by_product.get(product_id, set())
        reviews = get_reviews(product_id, max_reviews=max_reviews, exclude_review_ids=exclude)

    if not reviews:
        reviews = [
            {
                "review_id": "",
                "comment_content": "",
                "comment_text": "",
                "comment_rating": 0,
                "review_created_at": "",
            }
        ]

    rows = []
    for review in reviews:
        rows.append(
            {
                "product_id": product_id,
                "product_name": product_name,
                "price": price,
                "rating": rating,
                "review_count": review_count,
                "sold_count": sold_count,
                "product_url": product_url,
                "search_keyword": keyword,
                "review_id": review["review_id"],
                "comment_content": review["comment_content"],
                "comment_text": review["comment_text"],
                "comment_rating": review["comment_rating"],
                "review_created_at": review["review_created_at"],
            }
        )

    return rows


def crawl_keyword(keyword, start_page, end_page, products_per_page, max_reviews, existing_reviews_by_product=None):
    """Crawl theo từ khóa."""
    all_rows = []
    empty_count = 0
    last_successful_page = start_page - 1

    print(f"\nKeyword: {keyword}")
    print(f"  Pages: {start_page} -> {end_page}")
    print(f"  Products/page: {products_per_page}")

    for page in tqdm(range(start_page, end_page + 1), desc=f"Crawling {keyword}"):
        params = {
            "limit": products_per_page,
            "page": page,
            "q": keyword,
            "include": "advertisement",
            "aggregations": 2,
            "version": "home-personalized",
        }
        data = make_request(SEARCH_URL, params=params)

        if not data:
            empty_count += 1
            print(f"  Page {page}: no data returned")
            if empty_count >= EMPTY_PAGE_LIMIT:
                print(f"  Stopped after {EMPTY_PAGE_LIMIT} empty/error pages")
                break
            time.sleep(random.uniform(1.0, 2.0))
            continue

        products = data.get("data", [])
        if not products:
            empty_count += 1
            print(f"  Page {page}: no products")
            if empty_count >= EMPTY_PAGE_LIMIT:
                print(f"  Stopped after {EMPTY_PAGE_LIMIT} empty pages")
                break
            time.sleep(random.uniform(1.0, 2.0))
            continue

        empty_count = 0
        last_successful_page = page
        print(f"  Page {page}: {len(products)} products")

        for idx, product in enumerate(products, start=1):
            product_name = (product.get("name") or "").strip()
            print(f"    Product {idx}/{len(products)}: {product_name[:70]}")
            all_rows.extend(build_product_rows(product, keyword, max_reviews, existing_reviews_by_product))
            time.sleep(random.uniform(0.35, 0.8))

        time.sleep(random.uniform(0.8, 1.6))

    return all_rows, last_successful_page


def read_existing_data():
    """Đọc dữ liệu CSV cũ."""
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
        try:
            return pd.read_csv(DATA_PATH, encoding=encoding)
        except UnicodeDecodeError:
            continue

    return pd.read_csv(DATA_PATH)


def ensure_schema(df):
    """Chuẩn hóa cột dữ liệu."""
    df = df.copy()

    if "comment_content" not in df.columns and "comment_text" in df.columns:
        df["comment_content"] = df["comment_text"]
    if "comment_text" not in df.columns and "comment_content" in df.columns:
        df["comment_text"] = df["comment_content"]

    defaults = {
        "product_id": "",
        "product_name": "",
        "price": 0,
        "rating": 0,
        "review_count": 0,
        "sold_count": 0,
        "product_url": "",
        "search_keyword": "",
        "review_id": "",
        "comment_content": "",
        "comment_text": "",
        "comment_rating": 0,
        "review_created_at": "",
    }

    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    df["product_id"] = df["product_id"].apply(normalize_id)
    df["review_id"] = df["review_id"].apply(normalize_id)

    for text_column in ("product_name", "product_url", "search_keyword", "comment_content", "comment_text"):
        df[text_column] = df[text_column].fillna("").astype(str).str.strip()

    df["comment_content"] = df["comment_content"].where(
        df["comment_content"].str.len() > 0,
        df["comment_text"],
    )
    df["comment_text"] = df["comment_text"].where(
        df["comment_text"].str.len() > 0,
        df["comment_content"],
    )

    for number_column in ("price", "rating", "review_count", "comment_rating"):
        df[number_column] = pd.to_numeric(df[number_column], errors="coerce").fillna(0)

    df["review_count"] = df["review_count"].astype(int)
    df["comment_rating"] = df["comment_rating"].astype(int)
    df["sold_count"] = df["sold_count"].apply(parse_sold_count).astype(int)

    ordered_columns = OUTPUT_COLUMNS + [col for col in df.columns if col not in OUTPUT_COLUMNS]
    return df[ordered_columns]


def combine_and_deduplicate(old_df, new_df):
    """Gộp và xóa trùng."""
    old_df = ensure_schema(old_df)
    new_df = ensure_schema(new_df)

    old_product_ids = set(old_df["product_id"].dropna().astype(str))
    new_product_ids = set(new_df["product_id"].dropna().astype(str))
    new_products_added = len(new_product_ids - old_product_ids)

    old_df["_source_order"] = range(len(old_df))
    old_df["_source_rank"] = 0
    new_df["_source_order"] = range(len(old_df), len(old_df) + len(new_df))
    new_df["_source_rank"] = 1

    combined_df = pd.concat([old_df, new_df], ignore_index=True)
    combined_df = ensure_schema(combined_df)
    combined_df["_source_order"] = pd.to_numeric(combined_df["_source_order"], errors="coerce").fillna(0)
    combined_df["_source_rank"] = pd.to_numeric(combined_df["_source_rank"], errors="coerce").fillna(1)
    combined_df["_has_review_id"] = combined_df["review_id"].str.len() > 0
    combined_df["_has_comment"] = combined_df["comment_content"].str.len() > 0
    combined_df["_has_search_keyword"] = combined_df["search_keyword"].str.len() > 0

    with_review_id = combined_df[combined_df["_has_review_id"]].copy()
    with_comment_no_review_id = combined_df[
        ~combined_df["_has_review_id"] & combined_df["_has_comment"]
    ].copy()
    without_comment = combined_df[
        ~combined_df["_has_review_id"] & ~combined_df["_has_comment"]
    ].copy()

    with_review_id = with_review_id.sort_values(
        ["_has_search_keyword", "_source_rank", "_source_order"],
        ascending=[False, True, True],
    ).drop_duplicates(subset=["product_id", "review_id"], keep="first")

    with_comment_no_review_id = with_comment_no_review_id.sort_values(
        ["_has_search_keyword", "_source_rank", "_source_order"],
        ascending=[False, True, True],
    ).drop_duplicates(subset=["product_id", "comment_content"], keep="first")

    without_comment = without_comment.sort_values(
        ["sold_count", "_has_search_keyword", "_source_rank", "_source_order"],
        ascending=[False, False, True, True],
    ).drop_duplicates(subset=["product_id", "search_keyword"], keep="first")

    result = pd.concat([with_review_id, with_comment_no_review_id, without_comment], ignore_index=True)
    result = result.sort_values("_source_order")

    helper_columns = [
        "_source_order",
        "_source_rank",
        "_has_review_id",
        "_has_comment",
        "_has_search_keyword",
    ]
    result = result.drop(columns=[col for col in helper_columns if col in result.columns])
    result = ensure_schema(result)

    return result, new_products_added


def backup_existing_data():
    """Sao lưu CSV cũ."""
    if not os.path.exists(DATA_PATH):
        return

    os.makedirs(os.path.dirname(BACKUP_PATH) or ".", exist_ok=True)
    shutil.copy2(DATA_PATH, BACKUP_PATH)
    print(f"Backup saved: {BACKUP_PATH}")


def save_data(df):
    """Lưu dữ liệu mới."""
    os.makedirs(os.path.dirname(DATA_PATH) or ".", exist_ok=True)
    backup_existing_data()
    df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved: {DATA_PATH}")


def print_top_sold_products(df):
    """In sách bán chạy."""
    product_view = (
        df.sort_values("sold_count", ascending=False)
        .drop_duplicates(subset=["product_id"], keep="first")
        .head(20)
    )
    columns = ["product_id", "product_name", "sold_count", "rating", "review_count", "search_keyword"]
    print("\nTop 20 products by sold_count:")
    if product_view.empty:
        print("  No products")
        return
    print(product_view[columns].to_string(index=False))


def print_top_keywords(df):
    """In từ khóa hiệu quả."""
    keyword_counts = (
        df[df["search_keyword"].str.len() > 0]
        .drop_duplicates(subset=["search_keyword", "product_id"])
        .groupby("search_keyword")["product_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(20)
    )

    print("\nTop 20 keywords by unique products:")
    if keyword_counts.empty:
        print("  No keyword data")
        return
    print(keyword_counts.to_string())


def print_hot_book_checks(df):
    """Kiểm tra sách nổi bật."""
    product_view = df.drop_duplicates(subset=["product_id"], keep="first").copy()
    product_view["_name_lower"] = product_view["product_name"].fillna("").astype(str).str.lower()

    print("\nHot book checks:")
    for keyword in HOT_BOOK_CHECKS:
        matches = product_view[product_view["_name_lower"].str.contains(keyword, regex=False)]
        if matches.empty:
            print(f"  {keyword}: not found")
            continue

        best_match = matches.sort_values("sold_count", ascending=False).iloc[0]
        print(
            f"  {keyword}: found {len(matches)} product(s); "
            f"top='{best_match['product_name']}', sold_count={best_match['sold_count']}"
        )


def print_statistics(old_len, new_len, merged_df, new_products_added):
    """In thống kê crawl."""
    print("\n" + "=" * 72)
    print("CRAWL STATISTICS")
    print("=" * 72)
    print(f"Tổng dòng dữ liệu cũ: {old_len}")
    print(f"Tổng dòng dữ liệu mới: {new_len}")
    print(f"Tổng dòng sau gộp: {len(merged_df)}")
    print(f"Số product_id duy nhất: {merged_df['product_id'].nunique()}")
    print(f"Số sản phẩm mới thêm: {new_products_added}")
    print_top_sold_products(merged_df)
    print_top_keywords(merged_df)
    print_hot_book_checks(merged_df)
    print("=" * 72)


def parse_keyword_arg(value):
    """Tách danh sách từ khóa."""
    if not value:
        return KEYWORDS
    return [keyword.strip() for keyword in value.split(",") if keyword.strip()]


def main():
    """Chạy chương trình crawl."""
    parser = argparse.ArgumentParser(description="Crawl books from Tiki public search API")
    parser.add_argument(
        "--pages-per-keyword",
        "--pages",
        dest="pages_per_keyword",
        type=int,
        default=MAX_PAGES_PER_KEYWORD,
        help="Number of pages to crawl per keyword",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=PRODUCTS_PER_PAGE,
        help="Products per search page",
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=MAX_REVIEWS_PER_PRODUCT,
        help="Maximum reviews to crawl per product",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default="",
        help="Optional comma-separated keyword subset",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Reset crawl state for the selected keywords",
    )
    args = parser.parse_args()

    selected_keywords = parse_keyword_arg(args.keywords)
    state = load_crawl_state()

    if args.reset_state:
        for keyword in selected_keywords:
            set_last_crawled_page(state, keyword, 0)
        save_crawl_state(state)
        print("Reset crawl_state.json for selected keywords")

    print("\nTiki Book Intelligence - Multi-keyword Crawler")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Keywords: {len(selected_keywords)}")
    print(f"Products/page: {args.limit}")
    print(f"Pages/keyword: {args.pages_per_keyword}")
    print(f"Max reviews/product: {args.max_reviews}")

    # Đọc dữ liệu cũ trước khi crawl để lọc review trùng theo review_id
    old_df = read_existing_data()
    old_df = ensure_schema(old_df)
    old_len = len(old_df)

    # Tạo index: product_id → set(review_id đã có)
    existing_reviews_by_product = (
        old_df[old_df["review_id"].str.len() > 0]
        .groupby("product_id")["review_id"]
        .apply(set)
        .to_dict()
    )

    all_new_rows = []

    for keyword in selected_keywords:
        last_page = get_last_crawled_page(state, keyword)
        start_page = last_page + 1
        end_page = last_page + args.pages_per_keyword

        rows, final_page = crawl_keyword(
            keyword=keyword,
            start_page=start_page,
            end_page=end_page,
            products_per_page=args.limit,
            max_reviews=args.max_reviews,
            existing_reviews_by_product=existing_reviews_by_product,
        )
        all_new_rows.extend(rows)

        if final_page >= start_page:
            set_last_crawled_page(state, keyword, final_page)
            save_crawl_state(state)
            print(f"Updated state: {keyword} -> last_crawled_page={final_page}")
        else:
            print(f"State unchanged for keyword '{keyword}' because no page succeeded")

    new_df = pd.DataFrame(all_new_rows)
    new_df = ensure_schema(new_df)
    new_len = len(new_df)

    if new_df.empty:
        print("\nNo new rows were crawled. Existing data was not rewritten.")
        print_statistics(old_len, new_len, old_df, 0)
        return

    merged_df, new_products_added = combine_and_deduplicate(old_df, new_df)
    save_data(merged_df)
    print_statistics(old_len, new_len, merged_df, new_products_added)

    print("\nDone. Next run will continue from the next page for each keyword.")


if __name__ == "__main__":
    main()
