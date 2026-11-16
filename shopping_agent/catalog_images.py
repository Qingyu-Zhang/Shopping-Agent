from __future__ import annotations

from collections import Counter
from urllib.parse import urlsplit

import httpx

HOSTS = {"m.media-amazon.com", "images-na.ssl-images-amazon.com", "images.amazon.com"}
MAX_BYTES = 8_000_000


def prepare_images(catalog, directory, limit, parent_asins=None):
    from io import BytesIO
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    requested = set(parent_asins) if parent_asins else None
    stats = Counter()
    with httpx.Client(timeout=10, follow_redirects=False) as http:
        for page in catalog._scan("catalog"):
            for doc in page:
                if (doc.get("source") != "Amazon Reviews'23" or doc.get("sku_id")
                    or requested is not None and doc.get("external_id") not in requested):
                    continue
                if catalog.image_path(directory, doc["id"]):
                    stats["already_present"] += 1
                    continue
                urls = [value for value in doc.get("images", []) if isinstance(value, str)
                        and urlsplit(value).scheme == "https" and urlsplit(value).netloc in HOSTS]
                if not urls:
                    continue
                if stats["attempted"] >= limit:
                    return dict(stats)
                stats["attempted"] += 1
                try:
                    with http.stream("GET", urls[0]) as response:
                        response.raise_for_status()
                        data = bytearray()
                        for block in response.iter_bytes():
                            data.extend(block)
                            if len(data) > MAX_BYTES:
                                raise ValueError("Image exceeds the size limit.")
                    with Image.open(BytesIO(data)) as source:
                        if source.width * source.height > 16_000_000:
                            raise ValueError("Image dimensions exceed the size limit.")
                        source.thumbnail((768, 768))
                        image = BytesIO()
                        source.convert("RGB").save(image, format="JPEG", quality=86)
                    target = directory / f'{doc["id"]}.jpg'
                    temporary = target.with_suffix(".jpg.part")
                    temporary.write_bytes(image.getvalue())
                    temporary.replace(target)
                    stats["ready"] += 1
                except (httpx.HTTPError, OSError, ValueError):
                    stats["unavailable"] += 1
    return dict(stats)
