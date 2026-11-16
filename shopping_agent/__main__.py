from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from math import ceil, log2
from pathlib import Path
from time import perf_counter

from .core import Settings, elapsed_ms
from .dataset import identifier, import_reviews23, records, seed_demo_offers
from .runtime import Runtime, Worker
from .storage import Store


def relevance_grades(case):
    if "relevance" in case:
        grades = case["relevance"]
        if not isinstance(grades, dict) or any(
            not isinstance(key, str) or not key.strip() or type(grade) is not int or not 0 <= grade <= 3
            for key, grade in grades.items()
        ):
            raise ValueError("Expected relevance grades from 0 to 3 keyed by product or document ID.")
        return {key: grade for key, grade in grades.items() if grade > 0}
    labels = case["relevant"]
    if not isinstance(labels, list) or any(not isinstance(key, str) or not key.strip() for key in labels):
        raise ValueError("Expected relevant to be a list of product or document IDs.")
    return dict.fromkeys(labels, 1)


def ndcg_at_k(matches, grades, k):
    if not grades:
        return None
    seen = set()
    dcg = 0.0
    for rank, matched in enumerate(matches[:k], 1):
        new = matched - seen
        if new:
            dcg += (2 ** max(grades[key] for key in new) - 1) / log2(rank + 1)
            seen.update(new)
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    return dcg / idcg


def evaluate(runtime, path, k, mode="review"):
    rows = []
    invalid = Counter()
    for case in records(path, invalid):
        started = perf_counter()
        query = case["query"]
        grades = relevance_grades(case)
        use_clip = (mode in {"clip", "vlm"} and case.get("visual") and runtime.settings.clip_model
                    and runtime.settings.elastic_url)
        visual = runtime.models.visual_query(text=query) if use_clip else None
        docs, trace = runtime.catalog.search(query, case.get("category"), limit=max(k, 6) if mode == "vlm" else k,
                                             review_query=False if mode == "text" else None, visual=visual)
        vlm = False
        if mode == "vlm" and case.get("visual"):
            docs, vlm = runtime.catalog.visual_rerank(query, docs)
        docs = docs[:k]
        ranked = [doc["id"] for doc in docs]
        labels = set(grades)
        relevant = [({doc["id"], doc["product_id"]} & labels) for doc in docs]
        found = set().union(*relevant)
        reciprocal_rank = next((1/i for i, matched in enumerate(relevant, 1) if matched), 0)
        terms = case.get("evidence_terms", [])
        product_ids = list(dict.fromkeys(doc["product_id"] for doc, matched in zip(docs, relevant) if matched))
        evidence = runtime.catalog.evidence(product_ids, query, limit=8) if terms and product_ids else []
        evidence_hit = (any(term.casefold() in item["text"].casefold() for term in terms for item in evidence)
                        if terms else None)
        rows.append({"query": query, "recall": len(found)/len(labels) if labels else None,
                     "reciprocal_rank": reciprocal_rank, "ndcg": ndcg_at_k(relevant, grades, k), "ranked": ranked,
                     "keyword": trace["keyword"][:k], "semantic": trace["semantic"][:k],
                     "review": trace.get("review_products", [])[:k], "image": trace.get("image", [])[:k],
                     "vlm_applied": vlm, "evidence_hit": evidence_hit, "latency_ms": elapsed_ms(started)})
    if invalid or not rows:
        raise ValueError("Expected a nonempty search evaluation file with valid JSON lines.")
    recalls = [x["recall"] for x in rows if x["recall"] is not None]
    ndcgs = [row["ndcg"] for row in rows if row["ndcg"] is not None]
    hits = [row["evidence_hit"] for row in rows if row["evidence_hit"] is not None]
    times = sorted(row["latency_ms"] for row in rows)
    return {"mode": mode, "k": k, "mean_recall": sum(recalls)/len(recalls) if recalls else None,
            "mrr": sum(x["reciprocal_rank"] for x in rows)/len(rows),
            "mean_ndcg": sum(ndcgs)/len(ndcgs) if ndcgs else None,
            "evidence_term_hit_rate": sum(hits)/len(hits) if hits else None,
            "clip_queries_with_hits": sum(bool(row["image"]) for row in rows),
            "vlm_queries_reranked": sum(row["vlm_applied"] for row in rows),
            "p95_ms": times[ceil(0.95*len(times))-1], "cases": rows}


def read_parent_asins(path):
    if path is None:
        return None
    parents = path.read_text(encoding="utf-8-sig").split()
    if not parents or not all(identifier(value) for value in parents):
        raise ValueError("The manifest must contain valid parent ASINs.")
    return parents


def main():
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Shopping Agent V2")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, port in (("api", 8000), ("platform", 8001)):
        command = commands.add_parser(name)
        command.add_argument("--host", default="127.0.0.1")
        command.add_argument("--port", type=int, default=port)
    commands.add_parser("worker")
    index = commands.add_parser("reindex")
    index.add_argument("--images", type=Path)
    images = commands.add_parser("prepare-images")
    images.add_argument("--images", type=Path, required=True)
    images.add_argument("--limit", type=int, default=100)
    images.add_argument("--parent-asins", type=Path)
    importer = commands.add_parser("import-reviews23")
    importer.add_argument("--metadata", type=Path, required=True)
    importer.add_argument("--reviews", type=Path)
    importer.add_argument("--products", type=int, default=1000)
    importer.add_argument("--reviews-per-product", type=int, default=20)
    importer.add_argument("--parent-asins", type=Path, help="Import an explicit manifest with one parent ASIN per line.")
    importer.add_argument("--demo-offers", type=Path, help="Load explicitly simulated SKU and offer fixtures in demo mode.")
    evaluator = commands.add_parser("evaluate")
    evaluator.add_argument("--cases", type=Path, default=Path("fixtures/search_cases.jsonl"))
    evaluator.add_argument("--k", type=int, default=5)
    evaluator.add_argument("--mode", choices=("text", "review", "clip", "vlm"), action="append",
                           help="Repeat to compare retrieval and visual ranking on the same cases.")
    journeys = commands.add_parser("evaluate-journeys")
    journeys.add_argument("--cases", type=Path, default=Path("fixtures/shopping_journeys.jsonl"))
    journeys.add_argument("--k", type=int, default=3, choices=(1, 2, 3))
    journeys.add_argument("--with-models", action="store_true", help="Use configured model services in the isolated evaluation.")
    args = parser.parse_args()
    if args.command in {"api", "platform"}:
        factory = "shopping_agent.api:create_app" if args.command == "api" else "shopping_agent.platform:create_platform_app"
        uvicorn.run(factory, factory=True, host=args.host, port=args.port)
        return
    settings = Settings.from_env()
    if args.command == "import-reviews23":
        if args.products < 1 or args.reviews_per_product < 1:
            parser.error("Import limits must be positive.")
        if args.demo_offers and not settings.demo:
            parser.error("Only import simulated offers into a demo deployment.")
        parents = read_parent_asins(args.parent_asins)
        store = Store(settings.platform_database)
        try:
            store.initialize()
            result = import_reviews23(store, args.metadata, args.reviews, product_limit=args.products,
                                      reviews_per_product=args.reviews_per_product, parent_asins=parents)
            if args.demo_offers:
                result["demo_offers_added"] = seed_demo_offers(store, args.demo_offers)
            print(json.dumps(result, indent=2))
        finally:
            store.engine.dispose()
        return
    if args.command == "evaluate-journeys":
        from .evaluation import evaluate_journeys
        report = evaluate_journeys(args.cases, args.k, model_settings=settings if args.with_models else None)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["failed_turns"]:
            raise SystemExit(1)
        return
    runtime = Runtime(settings, setup_graph=args.command == "worker")
    try:
        if args.command == "worker":
            worker = Worker(runtime)
            worker.start()
            try:
                while not worker.stop.wait(1):
                    pass
            except KeyboardInterrupt:
                pass
            finally:
                worker.close()
        elif args.command == "reindex":
            print(json.dumps(runtime.catalog.rebuild(args.images), indent=2))
        elif args.command == "prepare-images":
            from .catalog_images import prepare_images
            if args.limit < 1:
                parser.error("Image limit must be positive.")
            print(json.dumps(prepare_images(runtime.catalog, args.images, args.limit,
                                            read_parent_asins(args.parent_asins)), indent=2))
        elif args.command == "evaluate":
            if not 1 <= args.k <= 50:
                parser.error("K must be between 1 and 50.")
            modes = list(dict.fromkeys(args.mode or ["review"]))
            results = [evaluate(runtime, args.cases, args.k, mode) for mode in modes]
            print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=2))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
