from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from functools import cached_property

import httpx

from .core import Settings, stable_id

LOG = logging.getLogger(__name__)
TOKEN = re.compile(r"[a-z0-9]+(?:[-_.@][a-z0-9]+)*", re.I)
ALIASES = {
    "notebook": "laptop", "laptops": "laptop", "ultralight": "lightweight", "light": "lightweight",
    "portable": "lightweight", "hub": "dock", "docking": "dock", "adapter": "dock",
    "travel": "commute", "commuter": "commute", "60hz": "60", "displayport": "dp",
}


def tokens(text: str) -> list[str]:
    return [ALIASES.get(x, x) for x in TOKEN.findall(text.lower().replace("–", "-"))]


class Models:
    def __init__(self, settings: Settings, cache=None):
        self.settings = settings
        self.http = httpx.Client(timeout=12)
        self.cache = cache

    @property
    def enabled(self):
        return bool(self.settings.model_url and self.settings.chat_model)

    def _call(self, base, key, path, body):
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        response = self.http.post(base.rstrip("/") + path, json=body, headers=headers)
        response.raise_for_status()
        return response.json()

    def chat(self, messages, *, tools=None, structured=False):
        body = {"model": self.settings.chat_model, "messages": messages, "temperature": 0}
        if self.settings.chat_model == "gpt-6-luna":
            # Luna requires reasoning to be disabled for Chat Completions tool calls.
            body.update(reasoning_effort="none", max_completion_tokens=650)
        else:
            body["max_tokens"] = 650
        if tools:
            body.update(tools=tools, tool_choice="auto")
        if structured:
            body["response_format"] = {"type": "json_object"}
        return self._call(self.settings.model_url, self.settings.model_key, "/chat/completions", body)["choices"][0]["message"]

    def embeddings(self, texts):
        if not texts:
            return []
        if not self.settings.embedding_model:
            return [self._local_vector(text) for text in texts]
        if self.cache is None:
            return self._embed(texts)
        from redis.exceptions import RedisError
        pairs = [(f"shopping:embedding:{stable_id(self.settings.embedding_model, self.settings.embedding_url or self.settings.model_url, text)}",
                  text) for text in texts]
        try:
            raw = self.cache.mget([key for key, _ in pairs])
            vectors = [json.loads(value) if value else None for value in raw]
        except (RedisError, ValueError, TypeError):
            return self._embed(texts)
        missing = [index for index, value in enumerate(vectors) if value is None]
        if missing:
            fresh = self._embed([pairs[index][1] for index in missing])
            for index, value in zip(missing, fresh, strict=True):
                vectors[index] = value
            try:
                with self.cache.pipeline(transaction=False) as pipeline:
                    for index in missing:
                        pipeline.set(pairs[index][0], json.dumps(vectors[index]), ex=600)
                    pipeline.execute()
            except RedisError:
                pass
        return vectors

    def _embed(self, texts):
        base = self.settings.embedding_url or self.settings.model_url
        if not base:
            raise ValueError("Configure EMBEDDING_URL or MODEL_URL for EMBEDDING_MODEL.")
        key = self.settings.embedding_key if self.settings.embedding_url else self.settings.model_key
        out = self._call(base, key, "/embeddings", {"model": self.settings.embedding_model, "input": texts})
        rows = sorted(out["data"], key=lambda row: row["index"])
        vectors = [r["embedding"] for r in rows]
        if (
            [r["index"] for r in rows] != list(range(len(texts))) or not vectors[0]
            or any(len(v) != len(vectors[0]) or not all(math.isfinite(x) for x in v) for v in vectors)
        ):
            raise ValueError("Invalid embedding response.")
        return vectors

    @staticmethod
    def _local_vector(text):
        vector = [0.0] * 128
        for word in tokens(text):
            digest = hashlib.blake2s(word.encode(), digest_size=2).digest()
            location = int.from_bytes(digest, "little") % len(vector)
            vector[location] += 1
        norm = math.sqrt(sum(x*x for x in vector))
        return [x/norm for x in vector] if norm else vector

    def rerank(self, query, docs):
        if self.settings.rerank_url and docs:
            try:
                value = self._call(self.settings.rerank_url, self.settings.rerank_key, "/rerank", {
                    "model": self.settings.rerank_model, "query": query, "documents": docs, "top_n": len(docs),
                })
                rows = value["results"]
                indices = [r["index"] for r in rows]
                if (sorted(indices) == list(range(len(docs)))
                    and all(math.isfinite(r["relevance_score"]) for r in rows)):
                    return [r["index"] for r in sorted(rows, key=lambda row: -row["relevance_score"])]
            except (httpx.HTTPError, KeyError, TypeError, ValueError):
                LOG.warning("Reranker unavailable. Keeping fused ranks.")
        return list(range(len(docs)))

    @cached_property
    def clip(self):
        from transformers import AutoProcessor, CLIPModel
        name = self.settings.clip_model
        return AutoProcessor.from_pretrained(name), CLIPModel.from_pretrained(name).eval()

    def _clip_features(self, method, inputs):
        import torch
        _, model = self.clip
        with torch.inference_mode():
            output = getattr(model, method)(**inputs)
            value = output if isinstance(output, torch.Tensor) else output.pooler_output
            return torch.nn.functional.normalize(value, dim=-1)[0].tolist()

    def visual_query(self, text=None, content=None):
        from io import BytesIO
        from PIL import Image
        processor, _ = self.clip
        if content:
            with Image.open(BytesIO(content)) as source:
                if source.width * source.height > 16_000_000:
                    raise ValueError("The image is too large.")
                inputs = processor(images=source.convert("RGB"), return_tensors="pt")
            return self._clip_features("get_image_features", dict(inputs))
        inputs = processor(text=[text], truncation=True, padding=True, return_tensors="pt")
        return self._clip_features("get_text_features", dict(inputs))
