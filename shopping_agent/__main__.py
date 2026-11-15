from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .core import Settings
from .dataset import import_reviews23, records
from .runtime import Runtime, Worker


def evaluate(runtime, path, k):
    rows = []
    for case in records(path, {}):
        docs, trace = runtime.catalog.search(case["query"], case.get("category"), limit=k)
        ranked = [doc["id"] for doc in docs]
        labels = set(case["relevant"])
        found = labels.intersection(ranked)
        reciprocal_rank = next((1/i for i, doc_id in enumerate(ranked, 1) if doc_id in labels), 0)
        rows.append({"query": case["query"], "recall": len(found)/len(labels) if labels else None,
                     "reciprocal_rank": reciprocal_rank, "ranked": ranked,
                     "keyword": trace["keyword"][:k], "semantic": trace["semantic"][:k]})
    recalls = [x["recall"] for x in rows if x["recall"] is not None]
    return {"k": k, "mean_recall": sum(recalls)/len(recalls) if recalls else None,
            "mrr": sum(x["reciprocal_rank"] for x in rows)/len(rows) if rows else None, "cases": rows}


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
    importer = commands.add_parser("import-reviews23")
    importer.add_argument("--metadata", type=Path, required=True)
    importer.add_argument("--reviews", type=Path)
    importer.add_argument("--products", type=int, default=1000)
    importer.add_argument("--reviews-per-product", type=int, default=20)
    evaluator = commands.add_parser("evaluate")
    evaluator.add_argument("--cases", type=Path, default=Path("fixtures/search_cases.jsonl"))
    evaluator.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    if args.command in {"api", "platform"}:
        factory = "shopping_agent.api:create_app" if args.command == "api" else "shopping_agent.platform:create_platform_app"
        uvicorn.run(factory, factory=True, host=args.host, port=args.port)
        return
    settings = Settings.from_env()
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
        elif args.command == "import-reviews23":
            if args.products < 1 or args.reviews_per_product < 1:
                parser.error("Import limits must be positive.")
            print(json.dumps(import_reviews23(runtime.store, args.metadata, args.reviews,
                                              product_limit=args.products,
                                              reviews_per_product=args.reviews_per_product), indent=2))
        elif args.command == "evaluate":
            if not 1 <= args.k <= 50:
                parser.error("K must be between 1 and 50.")
            print(json.dumps(evaluate(runtime, args.cases, args.k), ensure_ascii=False, indent=2))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
