from __future__ import annotations

import gzip
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .core import stable_id

SOURCE = "Amazon Reviews'23"


def records(path, counters):
    opener = gzip.open if Path(path).suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        for line in stream:
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError
                yield row
            except ValueError:
                counters["invalid_rows"] += 1


def identifier(value):
    return isinstance(value, str) and value.isascii() and value.isalnum() and len(value) <= 40


def strings(value):
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def read_details(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def image_rows(value):
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        keys = ("hi_res", "large", "thumb", "variant")
        lists = {key: value[key] for key in keys if isinstance(value.get(key), list)}
        return [{key: data[i] for key, data in lists.items() if i < len(data)}
                for i in range(max(map(len, lists.values()), default=0))]
    return []


def snapshot_price(value):
    try:
        amount = Decimal(str(value))
        return int((amount * 100).quantize(Decimal("1"))) if amount.is_finite() and 0 <= amount <= 1_000_000 else None
    except (ValueError, InvalidOperation):
        return None


def category_for(title, categories):
    import re
    text = " ".join([title, *categories]).lower()
    for category, pattern in (
        ("dock", r"\b(docking station|usb[ -]?c hub|thunderbolt dock|dock for)\b"),
        ("monitor", r"\b(monitors|display monitor|gaming monitor)\b"),
        ("laptop", r"\b(laptops|laptop computer|notebook computer|chromebook)\b"),
        ("filter", r"\b(air purifier filter|replacement air filter)\b"),
        ("pet_food", r"\b(cat food|kitten food)\b"),
    ):
        if re.search(pattern, text):
            return category
    return "other"


def metadata_document(row):
    asin, title = row.get("parent_asin"), row.get("title")
    if not identifier(asin) or not isinstance(title, str) or not title.strip():
        return None
    key = "reviews23-" + asin
    details = read_details(row.get("details"))
    categories = strings(row.get("categories"))
    photos = []
    for image in image_rows(row.get("images")):
        url = next((image[k] for k in ("hi_res", "large", "thumb")
                    if isinstance(image.get(k), str) and image[k].startswith("https://")), None)
        if url and url not in photos:
            photos.append(url)
    rating = row.get("average_rating")
    count = row.get("rating_number")
    return {
        "id": key, "product_id": key, "sku_id": None, "external_id": asin, "source": SOURCE,
        "source_version": "2023", "title": title.strip()[:500],
        "brand": str(details.get("Brand") or details.get("Manufacturer") or "")[:150],
        "category": category_for(title, categories), "category_path": categories[:20],
        "details": details, "description": " ".join(strings(row.get("description")) + strings(row.get("features")))[:10000],
        "snapshot_price": snapshot_price(row.get("price")), "snapshot_currency": "USD",
        "rating": rating if isinstance(rating, (int, float)) and 0 <= rating <= 5 else None,
        "rating_count": count if type(count) is int and count >= 0 else None,
        "images": photos[:12], "specs": {}, "active": True, "market": "US",
    }


def review_document(row, product_id):
    asin = row.get("asin")
    asin = asin if identifier(asin) else None
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    title = row.get("title")
    title = title if isinstance(title, str) else ""
    when = row.get("timestamp", row.get("sort_timestamp"))
    helpful = row.get("helpful_vote", row.get("helpful_votes"))
    key = stable_id(SOURCE, product_id, asin or "", str(when), str(row.get("user_id")), title, text)
    rating = row.get("rating")
    return {
        "id": key, "product_id": product_id, "sku_id": None, "external_asin": asin,
        "kind": "review", "text": (title + "\n" + text).strip()[:8000], "source": SOURCE,
        "rating": rating if isinstance(rating, (int, float)) and 1 <= rating <= 5 else None,
        "timestamp": when if type(when) is int else None,
        "helpful_votes": helpful if type(helpful) is int and helpful >= 0 else None,
        "verified_purchase": row.get("verified_purchase") is True,
    }


def batches(rows, size=200):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def import_reviews23(store, metadata_path, reviews_path, *, product_limit=1000, reviews_per_product=20):
    counters = Counter()
    retained = set()
    for batch in batches(records(metadata_path, counters)):
        with store.transaction(True) as tx:
            for row in batch:
                doc = metadata_document(row)
                if not doc:
                    counters["invalid_metadata"] += 1
                    continue
                key = doc["id"]
                if key not in retained and len(retained) >= product_limit:
                    break
                tx.put("catalog", key, doc)
                tx.put("products", key, doc)
                retained.add(key)
        if len(retained) >= product_limit:
            break
    counters["products"] = len(retained)
    if not reviews_path:
        return dict(counters)
    counts = Counter()
    for batch in batches(records(reviews_path, counters)):
        with store.transaction(True) as tx:
            for row in batch:
                parent = row.get("parent_asin")
                if not identifier(parent):
                    counters["invalid_reviews"] += 1
                    continue
                product_id = "reviews23-" + parent
                if product_id not in retained:
                    counters["reviews_outside_subset"] += 1
                    continue
                if counts[product_id] >= reviews_per_product:
                    continue
                doc = review_document(row, product_id)
                if not doc:
                    counters["invalid_reviews"] += 1
                    continue
                if tx.put("evidence", doc["id"], doc, owner=product_id, once=True):
                    counters["reviews_added"] += 1
                counts[product_id] += 1
        if retained and len(counts) == len(retained) and min(counts.values()) >= reviews_per_product:
            break
    return dict(counters)
