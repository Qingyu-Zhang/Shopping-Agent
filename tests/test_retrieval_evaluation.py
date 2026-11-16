import json
from math import log2
from types import SimpleNamespace

import pytest

from shopping_agent.__main__ import evaluate, relevance_grades


def test_ndcg_with_graded_relevance_and_repeated_product(tmp_path):
    class Catalog:
        def search(self, query, *_args, **_kwargs):
            order = {
                "graded": [
                    {"id": "b-1", "product_id": "b"}, {"id": "b-2", "product_id": "b"},
                    {"id": "a-1", "product_id": "a"},
                ],
                "binary": [{"id": "x", "product_id": "x"}, {"id": "hit", "product_id": "product"}],
                "unjudged": [{"id": "x", "product_id": "x"}],
            }
            return order[query], {"keyword": [], "semantic": []}

    path = tmp_path / "queries.jsonl"
    labels = [
        {"query": "graded", "relevance": {"a": 3, "b": 2, "other": 0}},
        {"query": "binary", "relevant": ["hit"]},
        {"query": "unjudged", "relevance": {"x": 0}},
    ]
    path.write_text("\n".join(json.dumps(label) for label in labels), encoding="utf-8")
    report = evaluate(SimpleNamespace(catalog=Catalog()), path, 3)
    graded = (3 + 7 / log2(4)) / (7 + 3 / log2(3))
    binary = 1 / log2(3)
    assert report["cases"][0]["ndcg"] == pytest.approx(graded)
    assert report["cases"][1]["ndcg"] == pytest.approx(binary)
    assert report["cases"][2]["ndcg"] is None
    assert report["mean_ndcg"] == pytest.approx((graded + binary) / 2)
    assert report["mean_recall"] == 1.0


def test_relevance_grade_validation():
    with pytest.raises(ValueError, match="relevance grades"):
        relevance_grades({"relevance": {"sku": True}})
