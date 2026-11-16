from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pydantic import Field

from .attributes import SCHEMA, Scalar, category_attributes, metadata_attributes, normalize_value
from .core import Input, stable_id, timestamp

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
    text = " ".join([title, *categories]).lower()
    for category, spec in SCHEMA["categories"].items():
        if (pattern := spec.get("catalog", spec["query"])) and re.search(pattern, text):
            return category
    return "other"


def metadata_document(row):
    asin, title = row.get("parent_asin"), row.get("title")
    if not identifier(asin) or not isinstance(title, str) or not title.strip():
        return None
    key = "reviews23-" + asin
    details = read_details(row.get("details"))
    categories = strings(row.get("categories"))
    category = category_for(title, categories)
    features, description = strings(row.get("features")), strings(row.get("description"))
    photos = []
    for image in sorted(image_rows(row.get("images")), key=lambda image: image.get("variant") != "MAIN"):
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
        "category": category, "category_path": categories[:20], "main_category": row.get("main_category"),
        "store": row.get("store"), "details": details, "attributes": metadata_attributes(category, details),
        "features": features, "description": " ".join(description),
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
        "kind": "review", "text": (title + "\n" + text).strip(), "source": SOURCE,
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


def import_reviews23(store, metadata_path, reviews_path, *, product_limit=1000, reviews_per_product=20,
                     parent_asins=None):
    counters = Counter()
    retained = set()
    requested = set(parent_asins) if parent_asins is not None else None
    limit = len(requested) if requested is not None else product_limit

    def products():
        for row in records(metadata_path, counters):
            if requested is not None and row.get("parent_asin") not in requested:
                continue
            doc = metadata_document(row)
            if doc is None:
                counters["invalid_metadata"] += 1
                continue
            retained.add(doc["id"])
            yield doc
            if len(retained) >= limit:
                break

    for batch in batches(products(), 150):
        unique = list({doc["id"]: doc for doc in batch}.values())
        with store.transaction(True) as tx:
            tx.put_many("catalog", unique, owner_field="product_id")
            tx.put_many("products", unique)
    counters["products"] = len(retained)
    if requested is not None:
        counters["missing_products"] = len(requested) - len(retained)
    if not reviews_path or not retained:
        return dict(counters)
    counts = Counter()
    review_ids = set()
    saturated = 0

    def selected_reviews():
        nonlocal saturated
        for row in records(reviews_path, counters):
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
            if doc["id"] in review_ids:
                continue
            review_ids.add(doc["id"])
            counts[product_id] += 1
            yield doc
            if counts[product_id] == reviews_per_product:
                saturated += 1
                if saturated == len(retained):
                    break

    for batch in batches(selected_reviews(), 150):
        with store.transaction(True) as tx:
            added = tx.put_many("evidence", batch, owner_field="product_id", once=True)
            if added:
                counters["reviews_added"] += added
    return dict(counters)


class DemoOffer(Input):
    parent_asin: str = Field(pattern=r"^[A-Za-z0-9]{1,40}$")
    sku_id: str = Field(pattern=r"^demo-[A-Za-z0-9-]{1,90}$")
    variant: str = Field(min_length=1, max_length=100)
    attributes: dict[str, Scalar | list[Scalar]]
    price_cents: int = Field(strict=True, ge=1, le=100_000_000)
    stock: int = Field(strict=True, ge=0)
    shipping_cents: int = Field(default=0, strict=True, ge=0)
    lead_hours: int = Field(default=24, strict=True, ge=0)


def seed_demo_offers(store, offers_path):
    counters = Counter()
    rows = [DemoOffer.model_validate(row) for row in records(offers_path, counters)]
    if counters["invalid_rows"] or not rows or len({row.sku_id for row in rows}) != len(rows):
        raise ValueError("Expected a nonempty demo offer file with unique SKU IDs.")
    now = timestamp()
    added = 0
    with store.transaction(True) as tx:
        for row in rows:
            parent = tx.get("products", "reviews23-" + row.parent_asin)
            if not parent or parent.get("source") != SOURCE:
                raise ValueError(f"Import parent {row.parent_asin} before seeding its demo offer.")
            specs = {}
            definitions = category_attributes(parent["category"])
            for key, value in row.attributes.items():
                spec = definitions.get(key)
                if not spec or (spec["type"] == "set") != isinstance(value, list):
                    raise ValueError(f"Invalid {key} for {row.sku_id}.")
                values = value if isinstance(value, list) else [value]
                typed = [normalize_value(spec, item) for item in values]
                if not typed or any(item is None for item in typed):
                    raise ValueError(f"Invalid {key} value for {row.sku_id}.")
                specs[key] = typed if isinstance(value, list) else typed[0]
            catalog = {**parent, "id": row.sku_id, "sku_id": row.sku_id, "variant": row.variant,
                       "specs": specs, "source": "Amazon Reviews'23 + simulated variant",
                       "variant_source": "Demo fixture"}
            tx.put("catalog", row.sku_id, catalog, owner=parent["id"], once=True)
            offer_id = f"{row.sku_id}-offer"
            offer = {"id": offer_id, "sku_id": row.sku_id, "title": parent["title"], "variant": row.variant,
                     "seller": "Demo Fashion Store", "unit_price": row.price_cents, "stock": row.stock,
                     "currency": "USD", "shipping": row.shipping_cents, "lead_hours": row.lead_hours,
                     "updated_at": now, "source": "Demo fixture"}
            if tx.put("offers", offer_id, offer, owner=row.sku_id, once=True):
                key = f"{offer_id}-initial"
                tx.put("price_history", key, {**offer, "id": key, "at": now}, owner=offer_id, once=True)
                added += 1
    return added
