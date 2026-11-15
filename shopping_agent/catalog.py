from __future__ import annotations

import json
import logging
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from .core import Fault, new_id, stable_id
from .models import tokens

LOG = logging.getLogger(__name__)


def document_text(doc):
    if doc.get("kind") == "review":
        return doc["text"]
    fields = [doc.get("title"), doc.get("brand"), doc.get("category"), doc.get("variant"),
              doc.get("external_id"), doc.get("description")]
    fields.extend(doc.get("category_path", []))
    fields.extend(json.dumps(doc.get(field, {}), ensure_ascii=False)[:1800] for field in ("specs", "details"))
    return " ".join(str(value) for value in fields if value)[:10000]


def rankings(lists, k=60):
    score = Counter()
    for entries in lists:
        for rank, key in enumerate(dict.fromkeys(entries), 1):
            score[key] += 1 / (k + rank)
    return sorted(score, key=lambda key: (-score[key], key))


def matches(doc, filters):
    return all(doc.get(key) in value if isinstance(value, list) else doc.get(key) == value
               for key, value in filters.items())


class Catalog:
    def __init__(self, store, models, settings):
        self.store, self.models, self.settings = store, models, settings
        self.http = httpx.Client(
            base_url=settings.elastic_url or "http://localhost:9200", timeout=12,
            headers={"Authorization": f"ApiKey {settings.elastic_key}"} if settings.elastic_key else {},
        )

    def _scan(self, bucket):
        after = None
        while page := self.store.list(bucket, after=after, limit=200):
            yield page
            after = page[-1]["id"]

    def _documents(self, bucket, filters):
        return [doc for page in self._scan(bucket) for doc in page if matches(doc, filters)]

    def _catalog_vectors(self, docs):
        identity = stable_id(self.settings.embedding_model or "demo-hash-128-v1",
                             self.settings.embedding_url or self.settings.model_url)
        texts = [document_text(doc) for doc in docs]
        keys = [stable_id(identity, text) for text in texts]
        found = {}
        pending = {}
        with self.store.transaction() as tx:
            for key, text in zip(keys, texts, strict=True):
                cached = tx.get("vectors", key)
                if cached:
                    found[key] = cached["vector"]
                else:
                    pending[key] = text
        if pending:
            vectors = self.models.embeddings(list(pending.values()))
            with self.store.transaction(True) as tx:
                for key, vector in zip(pending, vectors, strict=True):
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
        body = {"size": number, "_source": ["id"]}
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
        return [row["_id"] for row in hits]

    def hybrid(self, bucket, query, filters, *, alternates=(), limit=12, pool=40, visual=None):
        queries = list(dict.fromkeys([query, *alternates]))[:3]
        size = max(limit, pool)
        if self.settings.elastic_url:
            try:
                with ThreadPoolExecutor(max_workers=8) as workers:
                    lex = [workers.submit(self._elastic_search, bucket, filters, value, size) for value in queries]
                    semantic = workers.submit(self.models.embeddings, queries)
                    image = (workers.submit(self._elastic_search, bucket, filters, "", size, visual, "image_embedding")
                             if visual else None)
                    try:
                        vectors = semantic.result()
                        vec = [workers.submit(self._elastic_search, bucket, filters, value, size, vector)
                               for value, vector in zip(queries, vectors, strict=True)]
                        semantic_hits = rankings([x.result() for x in vec])
                    except (httpx.HTTPError, ValueError, KeyError, TypeError):
                        LOG.warning("Semantic search unavailable. Keeping keyword ranks.")
                        semantic_hits = []
                    keyword = rankings([x.result() for x in lex])
                    lists = [keyword, semantic_hits]
                    if image:
                        lists.append(image.result())
                keys = rankings(lists)[:size]
                with self.store.transaction() as tx:
                    docs = [doc for key in keys if (doc := tx.get(bucket, key)) and matches(doc, filters)]
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
                raise Fault("search_unavailable", "The search index is unavailable. Try again shortly.", 503) from error
        else:
            docs = self._documents(bucket, filters)
            lex = [self._lexical(docs, value, size) for value in queries]
            vectors = self._catalog_vectors(docs)
            vec = []
            for query_vector in self.models.embeddings(queries):
                scores = [(sum(a * b for a, b in zip(query_vector, vector, strict=True)), doc["id"])
                          for doc, vector in zip(docs, vectors, strict=True)]
                vec.append([key for score, key in sorted(scores, key=lambda row: (-row[0], row[1])) if score > 0][:size])
            lists = [rankings(lex), rankings(vec)]
            keys = rankings(lists)[:size]
            by_key = {doc["id"]: doc for doc in docs}
            docs = [by_key[key] for key in keys]
        rank_order = self.models.rerank(query, [document_text(doc) for doc in docs])
        traces = {"keyword": lists[0][:size], "semantic": lists[1][:size], "fused": keys}
        if len(lists) == 3:
            traces["image"] = lists[2][:size]
        return [docs[i] for i in rank_order[:limit]], traces

    def search(self, query, category=None, *, alternates=(), limit=12, pool=40, visual=None):
        filters = {"active": True, "market": "US"}
        if category:
            filters["category"] = category
        return self.hybrid("catalog", query, filters, alternates=alternates, limit=limit, pool=pool, visual=visual)

    def evidence(self, product_ids, query, limit=8):
        if not product_ids:
            return []
        docs, _ = self.hybrid("evidence", query, {"product_id": product_ids}, limit=limit, pool=limit*2)
        return docs

    def rebuild(self, images: Path | None = None):
        if not self.settings.elastic_url:
            raise ValueError("Configure ELASTIC_URL to rebuild the search indices.")
        if images and not self.settings.clip_model:
            raise ValueError("Set CLIP_MODEL before supplying local catalog images.")
        dim = len(self.models.embeddings(["index probe"])[0])
        vector_type = {"type": "dense_vector", "dims": dim, "index": True, "similarity": "cosine",
                       "index_options": {"type": "hnsw"}}
        properties = {
            key: {"type": "keyword"} for key in ("id", "category", "market", "product_id")
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
            for page in self._scan(bucket):
                vectors = self._catalog_vectors(page)
                operations = []
                for doc, vector in zip(page, vectors, strict=True):
                    payload = {key: doc[key] for key in (
                        "id", "category", "market", "product_id", "title", "brand", "variant", "active",
                    ) if doc.get(key) is not None}
                    text = document_text(doc)
                    payload.update(text=text, search_terms=list(dict.fromkeys(tokens(text)))[:600])
                    if any(vector):
                        payload["embedding"] = vector
                    if images and bucket == "catalog":
                        path = next((images / (doc["id"] + ext) for ext in (".jpg", ".jpeg", ".png", ".webp")
                                     if (images / (doc["id"] + ext)).is_file()), None)
                        if path and path.stat().st_size <= 8_000_000:
                            payload["image_embedding"] = self.models.visual_query(content=path.read_bytes())
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
