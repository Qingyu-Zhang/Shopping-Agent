from __future__ import annotations

import json
import heapq
import logging
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import batched, groupby
from pathlib import Path
from time import perf_counter

import httpx

from .core import Fault, elapsed_ms, new_id, stable_id
from .models import tokens
from .passages import document_passages, document_text

LOG = logging.getLogger(__name__)
REVIEW_QUERY = re.compile(
    r"\b(?:reviews?|reviewers?|comfort(?:able)?|durab(?:le|ility)|fit|fits|run[s]?\s+(?:small|large|true)|"
    r"wide feet|narrow feet|(?:long|all) day|(?:stand|standing) (?:for|all)|walk(?:ing)? for|"
    r"break[ -]in|breathab(?:le|ility)|quality|blisters?|wash(?:es|ing)? well)\b", re.I,
)


def rankings(lists, k=60, *, bonuses=None):
    score = Counter(bonuses or {})
    for entries in lists:
        for rank, key in enumerate(dict.fromkeys(entries), 1):
            score[key] += 1 / (k + rank)
    return sorted(score, key=lambda key: (-score[key], key))


def matches(doc, filters):
    return all(doc.get(key) in value if isinstance(value, list) else doc.get(key) == value
               for key, value in filters.items())


class Catalog:
    def __init__(self, store, models, settings, *, source=None):
        self.store, self.models, self.settings = store, models, settings
        self.sources = (source, store) if source is not None else (store,)
        self.http = httpx.Client(
            base_url=settings.elastic_url or "http://localhost:9200", timeout=12,
            headers={"Authorization": f"ApiKey {settings.elastic_key}"} if settings.elastic_key else {},
        )

    @staticmethod
    def _source_rows(source, bucket, owners):
        after = None
        while page := source.list(bucket, owner=owners, after=after, limit=200):
            yield from page
            after = page[-1]["id"]

    def _scan(self, bucket, owners=None):
        merged = heapq.merge(*(self._source_rows(source, bucket, owners) for source in self.sources),
                             key=lambda doc: doc["id"])
        return batched((next(rows) for _, rows in groupby(merged, key=lambda doc: doc["id"])), 200)

    def _documents(self, bucket, filters):
        owners = filters.get("product_id") if bucket == "evidence" else None
        return [passage for page in self._scan(bucket, owners) for doc in page if matches(doc, filters)
                for passage in document_passages(doc)]

    def get_many(self, bucket, keys):
        missing = set(keys)
        found = {}
        for source in self.sources:
            if not missing:
                break
            values = source.get_many(bucket, missing)
            found.update(values)
            missing.difference_update(values)
        return found

    def get(self, bucket, key):
        return self.get_many(bucket, [key]).get(key)

    def _catalog_for_products(self, product_ids, filters):
        if not product_ids:
            return []
        found = {}
        for source in self.sources:
            for doc in source.catalog_for_products(product_ids):
                found.setdefault(doc["id"], doc)
        return [doc for doc in found.values() if matches(doc, filters)]

    def _catalog_vectors(self, docs):
        identity = stable_id(self.settings.embedding_model or "demo-hash-128-v1",
                             self.settings.embedding_url or self.settings.model_url)
        texts = [document_text(doc) for doc in docs]
        keys = [stable_id(identity, text) for text in texts]
        found = {}
        pending = {}
        cached = {key: doc for batch in batched(dict.fromkeys(keys), 400)
                  for key, doc in self.store.get_many("vectors", batch).items()}
        for key, text in zip(keys, texts, strict=True):
            if key in cached:
                found[key] = cached[key]["vector"]
            else:
                pending[key] = text
        if pending:
            for batch in batched(pending, 64):
                vectors = self.models.embeddings([pending[key] for key in batch])
                with self.store.transaction(True) as tx:
                    for key, vector in zip(batch, vectors, strict=True):
                        tx.put("vectors", key, {"vector": vector}, once=True)
                        found[key] = vector
        return [found[key] for key in keys]

    @staticmethod
    def _lexical(docs, query, size):
        words = set(tokens(query))
        term_lists = [Counter(tokens(document_text(doc))) for doc in docs]
        df = Counter(word for terms in term_lists for word in terms if word in words)
        average = sum(sum(x.values()) for x in term_lists) / max(1, len(docs)) or 1
        scored = []
        for doc, terms in zip(docs, term_lists, strict=True):
            size_factor = 1.2 * (0.25 + 0.75 * sum(terms.values()) / average)
            score = sum(math.log1p((len(docs)-df[word]+0.5)/(df[word]+0.5)) * 2.2 * terms[word]
                        / (terms[word] + size_factor) for word in words if terms[word])
            score += 0.8 * len(words.intersection(tokens(doc.get("title", ""))))
            if score > 0:
                scored.append((score, doc["id"]))
        return [key for _, key in sorted(scored, key=lambda row: (-row[0], row[1]))[:size]]

    def _request(self, method, path, **kwargs):
        response = self.http.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    def _elastic_search(self, bucket, filters, query, number, vector=None, field="embedding"):
        clauses = [{"terms" if isinstance(v, list) else "term": {k: v}} for k, v in filters.items()]
        body = {"size": number, "_source": ["record_id", "product_id"]}
        if vector is None:
            phrase = list(dict.fromkeys(tokens(query)))[:32]
            text_query = {"bool": {"filter": clauses, "should": [{
                "multi_match": {"query": query, "fields": ["title^4", "brand^2", "variant^2", "text"],
                                "type": "best_fields"}
            }], "minimum_should_match": 1}}
            if phrase:
                text_query["bool"]["should"].append({"terms": {"search_terms": phrase, "boost": 3}})
            body["query"] = text_query
        else:
            if not any(vector):
                return []
            body["knn"] = {"field": field, "query_vector": vector, "k": number,
                           "num_candidates": min(1000, number * 4), "filter": clauses}
        try:
            hits = self._request("POST", f"/{self.settings.index}-{bucket}/_search", json=body)["hits"]["hits"]
        except httpx.HTTPStatusError as error:
            if field == "image_embedding" and error.response.status_code == 400:
                LOG.warning("No compatible image search mapping. Keeping text ranks.")
                return []
            raise
        return [{"id": row["_id"], "record_id": row.get("_source", {}).get("record_id", row["_id"]),
                 "product_id": row.get("_source", {}).get("product_id")} for row in hits]

    @staticmethod
    def _rank_hits(groups, key="record_id"):
        by_key = {}
        for group in groups:
            for hit in group:
                if hit.get(key):
                    by_key.setdefault(hit[key], hit)
        return [by_key[value] for value in rankings([[hit[key] for hit in group if hit.get(key)] for group in groups])]

    def _ranks(self, bucket, queries, filters, depth, visual=None, *, by="record_id"):
        if self.settings.elastic_url:
            with ThreadPoolExecutor(max_workers=8) as workers:
                lex = [workers.submit(self._elastic_search, bucket, filters, query, depth) for query in queries]
                semantic = workers.submit(self.models.embeddings, queries)
                image = (workers.submit(self._elastic_search, bucket, filters, "", depth, visual, "image_embedding")
                         if visual else None)
                try:
                    vec = [workers.submit(self._elastic_search, bucket, filters, query, depth, vector)
                           for query, vector in zip(queries, semantic.result(), strict=True)]
                    semantic_hits = self._rank_hits([result.result() for result in vec], by)
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    LOG.warning("Semantic search unavailable. Keeping keyword ranks.")
                    semantic_hits = []
                channels = {"keyword": self._rank_hits([result.result() for result in lex], by),
                            "semantic": semantic_hits}
                if image:
                    try:
                        channels["image"] = self._rank_hits([image.result()], by)
                    except httpx.HTTPError:
                        LOG.warning("Image retrieval unavailable. Keeping text ranks.")
                return channels
        docs = self._documents(bucket, filters)
        if not docs:
            return {"keyword": [], "semantic": []}
        by_key = {doc["id"]: doc for doc in docs}
        lex = [[by_key[key] for key in self._lexical(docs, query, depth)] for query in queries]
        vectors = self._catalog_vectors(docs)
        vec = []
        for query_vector in self.models.embeddings(queries):
            scores = [(sum(a*b for a, b in zip(query_vector, vector, strict=True)), doc["id"])
                      for doc, vector in zip(docs, vectors, strict=True)]
            vec.append([by_key[key] for score, key in sorted(scores, key=lambda row: (-row[0], row[1]))
                        if score > 0][:depth])
        return {"keyword": self._rank_hits(lex, by), "semantic": self._rank_hits(vec, by)}

    def _hydrate_hits(self, hits, records, filters):
        by_record = {}
        for hit in hits:
            record = records.get(hit["record_id"])
            if not record or not matches(record, filters) or record["id"] in by_record:
                continue
            for passage in document_passages(record):
                if passage["id"] == hit["id"]:
                    by_record[record["id"]] = passage
                    break
        return by_record

    def _discover(self, query, filters, size):
        channels = self._ranks("evidence", [query], {}, size * 3, by="product_id")
        products = rankings([[doc["product_id"] for doc in group] for group in channels.values()])[:size]
        hits = [hit for group in channels.values() for hit in group if hit["product_id"] in products]
        records = self.get_many("evidence", [hit["record_id"] for hit in hits])
        passages = self._hydrate_hits(hits, records, {"product_id": products})
        evidence = {}
        for passage in passages.values():
            excerpts = evidence.setdefault(passage["product_id"], [])
            if len(excerpts) < 2:
                excerpts.append(passage)
        candidates = sorted(self._catalog_for_products(products, filters),
                            key=lambda doc: (products.index(doc["product_id"]), doc["id"]))
        return candidates, evidence, products

    def hybrid(self, bucket, query, filters, *, alternates=(), limit=12, pool=40, visual=None, reject=None,
               review_query=None, prefer=None):
        started = perf_counter()
        queries = list(dict.fromkeys([query, *alternates]))[:3]
        size = max(limit, pool)
        related, evidence, review_products = [], {}, []
        try:
            with ThreadPoolExecutor(max_workers=2) as workers:
                review_future = (workers.submit(self._discover, review_query, filters, size)
                                 if bucket == "catalog" and review_query else None)
                channels = self._ranks(bucket, queries, filters, size * 3, visual)
                if review_future:
                    try:
                        related, evidence, review_products = review_future.result()
                    except (httpx.HTTPError, ValueError, KeyError, TypeError):
                        LOG.warning("Comment-based discovery unavailable. Keeping catalog ranks.")
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
            raise Fault("search_unavailable", "The search index is unavailable. Try again shortly.", 503) from error
        lists = {name: [hit["record_id"] for hit in hits[:size]] for name, hits in channels.items()}
        review_rank = {key: rank for rank, key in enumerate(review_products, 1)}
        review_bonus = {doc["id"]: 1.1 / (60 + review_rank[doc["product_id"]]) for doc in related}
        keys = rankings(list(lists.values()), bonuses=review_bonus)
        lists["review"] = list(review_bonus)
        records = self.get_many(bucket, keys)
        passages = self._hydrate_hits([hit for group in channels.values() for hit in group], records, filters)
        docs = [records[key] for key in keys if key in records and matches(records[key], filters)]
        timings = {"retrieval": elapsed_ms(started)}
        traces = {**{name: group[:size] for name, group in lists.items()}, "fused": keys[:size],
                  "review_products": review_products, "timings_ms": timings}
        if reject is not None:
            started = perf_counter()
            eligible, rejected = [], []
            for doc in docs:
                reason = reject(doc)
                if reason:
                    rejected.append({"id": doc["id"], "reason": reason})
                else:
                    eligible.append(doc)
            docs = eligible
            timings["eligibility"] = elapsed_ms(started)
            traces["rejected"] = rejected
        started = perf_counter()
        docs = docs[:size]
        texts = []
        for doc in docs:
            text = document_text(passages.get(doc["id"], doc))[:2000]
            notes = evidence.get(doc["product_id"], [])
            if notes:
                text += "\nCustomer experiences:\n" + "\n".join(document_text(note) for note in notes)
            texts.append(text)
        rank_order = self.models.rerank(query, texts) if docs else []
        ordered = [docs[i] for i in rank_order]
        if prefer:
            ordered.sort(key=prefer, reverse=True)
        selected = ordered[:limit]
        if bucket == "evidence":
            selected = [{**doc, "text": document_text(passages[doc["id"]]), "record_id": doc["id"],
                         "id": passages[doc["id"]]["id"]} for doc in selected if doc["id"] in passages]
        timings["rerank"] = elapsed_ms(started)
        traces["shortlist"] = [doc["id"] for doc in selected]
        return selected, traces

    def search(self, query, category=None, *, alternates=(), limit=12, pool=40, visual=None, reject=None,
               review_query=None, prefer=None):
        filters = {"active": True, "market": "US"}
        if category:
            filters["category"] = category
        if not self.settings.review_recall:
            review_query = None
        elif review_query is None and REVIEW_QUERY.search(query):
            review_query = query
        return self.hybrid("catalog", query, filters, alternates=alternates, limit=limit, pool=pool, visual=visual,
                           reject=reject, review_query=review_query, prefer=prefer)

    def evidence(self, product_ids, query, limit=8, *, sku_id=None, external_asin=None):
        if not product_ids:
            return []
        filters = {"product_id": product_ids}
        if sku_id or external_asin:
            exact = {**filters, **({"external_asin": external_asin} if external_asin else {"sku_id": sku_id})}
            docs, _ = self.hybrid("evidence", query, exact, limit=limit, pool=limit*2)
            if docs:
                return docs
        docs, _ = self.hybrid("evidence", query, filters, limit=limit, pool=limit*2)
        return docs

    @staticmethod
    def image_path(directory, record_id):
        if not directory or not re.fullmatch(r"[\w-]+", record_id):
            return None
        return next((directory / (record_id + ext) for ext in (".jpg", ".jpeg", ".png", ".webp")
                     if (directory / (record_id + ext)).is_file()), None)

    def visual_rerank(self, query, candidates, reference=None):
        if not (self.settings.vision_chat_model and self.settings.model_url and self.settings.catalog_images):
            return candidates, False
        try:
            directory, images, details = Path(self.settings.catalog_images), {}, {}
            for doc in candidates[:6]:
                path = self.image_path(directory, doc["id"])
                if path and path.stat().st_size <= 8_000_000:
                    images[doc["id"]] = path.read_bytes()
                    details[doc["id"]] = {key: doc.get(key) for key in ("title", "variant")}
            if len(images) < 2:
                return candidates, False
            order = iter(self.models.vision_rank(query, images, reference, details=details))
            by_id = {doc["id"]: doc for doc in candidates}
            return [by_id[next(order)] if doc["id"] in images else doc for doc in candidates], True
        except (httpx.HTTPError, ValueError, KeyError, TypeError, OSError, ImportError):
            LOG.warning("Visual reranking unavailable. Keeping retrieval ranks.")
            return candidates, False

    def rebuild(self, images: Path | None = None):
        if images is None and self.settings.catalog_images and self.settings.clip_model:
            images = Path(self.settings.catalog_images)
        if not self.settings.elastic_url:
            raise ValueError("Configure ELASTIC_URL to rebuild the search indices.")
        if images and not self.settings.clip_model:
            raise ValueError("Set CLIP_MODEL before supplying local catalog images.")
        dim = len(self.models.embeddings(["index probe"])[0])
        vector_type = {"type": "dense_vector", "dims": dim, "index": True, "similarity": "cosine",
                       "index_options": {"type": "hnsw"}}
        properties = {
            key: {"type": "keyword"} for key in (
                "id", "category", "market", "product_id", "record_id", "sku_id", "external_asin",
            )
        }
        properties.update({key: {"type": "text", "analyzer": "english"} for key in ("title", "text", "brand", "variant")})
        properties.update(active={"type": "boolean"}, embedding=vector_type, search_terms={"type": "keyword"})
        mapping = {"dynamic": "strict", "properties": properties}
        suffix = new_id()[:12]
        actions = []
        counts = {}
        for bucket in ("catalog", "evidence"):
            alias = f"{self.settings.index}-{bucket}"
            name = f"{alias}-{suffix}"
            bucket_mapping = {"dynamic": "strict", "properties": dict(mapping["properties"])}
            if images and bucket == "catalog":
                image_dim = len(self.models.visual_query(text="probe"))
                bucket_mapping["properties"]["image_embedding"] = {**vector_type, "dims": image_dim}
            self._request("PUT", "/" + name, json={"mappings": bucket_mapping})
            counts[bucket] = 0
            chunks = (part for page in self._scan(bucket) for doc in page for part in document_passages(doc))
            for page in batched(chunks, 100):
                vectors = self._catalog_vectors(page)
                operations = []
                for doc, vector in zip(page, vectors, strict=True):
                    payload = {key: doc[key] for key in (
                        "id", "category", "market", "product_id", "title", "brand", "variant", "active",
                        "record_id", "sku_id", "external_asin",
                    ) if doc.get(key) is not None}
                    text = document_text(doc)
                    payload.update(text=text, search_terms=list(dict.fromkeys(tokens(text)))[:600])
                    if any(vector):
                        payload["embedding"] = vector
                    if images and bucket == "catalog" and doc["id"] == doc["record_id"]:
                        path = self.image_path(images, doc["id"]) or self.image_path(images, doc["product_id"])
                        if path and path.stat().st_size <= 8_000_000:
                            try:
                                vector = self.models.visual_query(content=path.read_bytes())
                                if any(vector):
                                    payload["image_embedding"] = vector
                            except (OSError, ValueError):
                                LOG.warning("Skipping unusable catalog image for %s.", doc["id"])
                    operations.extend([{"index": {"_id": doc["id"]}}, payload])
                out = self._request("POST", f"/{name}/_bulk", content="\n".join(
                    json.dumps(line, ensure_ascii=True) for line in operations
                ) + "\n", headers={"Content-Type": "application/x-ndjson"})
                if out.get("errors"):
                    raise RuntimeError("Elasticsearch rejected an index batch. Existing aliases are intact.")
                counts[bucket] += len(page)
            self._request("POST", f"/{name}/_refresh")
            existing = self.http.get(f"/_alias/{alias}")
            if existing.status_code != 404:
                existing.raise_for_status()
                actions.extend({"remove": {"index": index, "alias": alias}} for index in existing.json())
            actions.append({"add": {"index": name, "alias": alias}})
        self._request("POST", "/_aliases", json={"actions": actions})
        return counts
