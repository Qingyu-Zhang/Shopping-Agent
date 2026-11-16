# Shopping Agent V2

An English-language shopping backend. FastAPI exposes versioned shopping tasks, catalog research, cart, checkout, orders, returns and price automation. LangGraph orchestrates hybrid retrieval, SKU filtering and recommendations with cited evidence. The platform sandbox exposes commerce tools over HTTP and MCP Streamable HTTP. The browser demo streams progress and recommendations.

## Run on Windows

Install Python 3.12 or later. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m shopping_agent api
```

Open [the demo](http://127.0.0.1:8000/) or [the API docs](http://127.0.0.1:8000/docs). Demo API tokens are `demo-alex` and `demo-sam`. This mode uses persistent SQLite files in `.data`, including the LangGraph checkpoints, an embedded worker, and a seeded commerce sandbox.

Try “Find a lightweight laptop and a dock that can drive a 4K 60 Hz monitor, under $1,000 total.” Follow with “Option 2, keep the laptop and replace the dock with a lighter dock.” You can also request a dock by itself and answer the host-protocol clarification. Supported browsers offer English speech input into the same editable message field.

## Run the service stack

Start Docker Desktop and run these commands from the project directory. Use `--build` after a source update to rebuild the application images.

```powershell
docker compose up --build -d
docker compose ps -a
```

Open [the demo](http://127.0.0.1:8000/) when the API and Worker are running. The `indexer` runs once and exits with code 0 after a successful build. The sandbox exposes Swagger at port 8001 and MCP at `/mcp`. PostgreSQL holds separate Agent and platform databases. The indexer and Agent read the platform catalog, and the Agent calls the sandbox over MCP for commerce operations. PostgreSQL, Redis and Elasticsearch use the private Compose network.

```powershell
docker compose logs api worker platform
docker compose stop
```

`docker compose stop` preserves the PostgreSQL and Elasticsearch data volumes. The supplied Compose stack runs in demo mode with test credentials. A non-demo deployment configures `DEMO=0`, JWTs from a trusted issuer, a platform service, separate secrets and secure infrastructure. Automatic buying is enabled explicitly with `AUTOBUY_ENABLED=1`. `CHECKOUT_ENABLED=0` pauses new checkout and return confirmations while previously accepted checkout attempts continue reconciliation.

## Configure models

Copy `.env.example` to `.env` as needed. Set `MODEL_URL` to a compatible API base URL including `/v1`, then set `CHAT_MODEL` and `MODEL_KEY`. Shopping turns use structured intent extraction and bounded read-only tools for complex or sparse searches. The model streams explanations for up to three candidates, and the service verifies candidate IDs and verbatim citations before publishing each recommendation as an SSE event. The demo can also display recommendations assembled directly from available evidence. Compatibility checks use catalog specifications, while prices and delivery estimates come from platform quotes.

Set `EMBEDDING_MODEL` with an embeddings endpoint to use semantic text vectors. Set `RERANK_URL` and `RERANK_MODEL` for a `POST /rerank` service returning indexed relevance scores. Local defaults use English parsing, BM25 and a deterministic demo vector. The fallback supports the built-in scenarios and regression tests. Real models are configurable.

Elasticsearch serves separate catalog and review passage indices. Keyword and dense-vector searches run in parallel. For experience-focused requests, review passages can introduce products into the RRF ranking. Short reviews stay intact, and passages from longer reviews retain a reference to the original record. The planner checks required attributes on the same SKU using [the catalog schema](shopping_agent/attributes.json). Equality, ranges and set matching share the same filtering mechanism. Rebuild the indices after catalog or embedding changes.

For CLIP retrieval, install `.[vision]` locally or rebuild Compose with `INSTALL_VISION=1`. Set `CLIP_MODEL` and `CATALOG_IMAGES`, then reindex with Elasticsearch enabled. Image files use catalog document IDs and JPEG, PNG or WebP extensions. `prepare-images --images PATH` can cache a bounded set of primary Amazon images. Appearance requests and reference images can add CLIP retrieval. Set `VISION_CHAT_MODEL` to a vision-capable model on the same provider to enable optional reranking of a small set of candidates with images. In Compose, use `CATALOG_IMAGES=/catalog-images`.

## Import Amazon Reviews’23

The importer accepts JSONL or gzip-compressed JSONL and batch-writes into `PLATFORM_DATABASE`. `CATALOG_DATABASE` can specify a separate connection for read-only catalog access. Both indexing and search-result hydration read the platform catalog. Demo records and earlier Agent imports remain readable for compatibility.

```powershell
.\.venv\Scripts\python.exe -m shopping_agent import-reviews23 --metadata .\fixtures\reviews23_meta.jsonl --reviews .\fixtures\reviews23_reviews.jsonl
.\.venv\Scripts\python.exe -m shopping_agent evaluate
```

The examples above use synthetic files. With the official files in `dataset/`, the following Compose commands import a fixed 23-product Amazon Fashion sample into PostgreSQL and rebuild the search indices. The optional demo offer file adds four simulated SKU variants with prices and inventory. To import a larger file-order sample, omit the manifest and demo offer options and set `--products` and `--reviews-per-product`.

```powershell
$datasetPath = (Resolve-Path -LiteralPath '.\dataset').Path
docker compose build indexer
docker compose run --rm --volume "${datasetPath}:/import:ro" indexer import-reviews23 --metadata /import/meta_Amazon_Fashion.jsonl --reviews /import/Amazon_Fashion.jsonl --parent-asins fixtures/fashion_parent_asins.txt --reviews-per-product 16 --demo-offers fixtures/fashion_demo_offers.jsonl
docker compose run --rm indexer reindex
docker compose run --rm indexer evaluate --cases fixtures/fashion_search_cases.jsonl --k 3 --mode text --mode review
```

Metadata and reviews join on `parent_asin`, and review citations retain the child ASIN. Imported listings show historical USD prices when present. The checkout planner requires verified SKU attributes and a live platform offer. Evaluation reports Recall@K, MRR, NDCG@K, evidence term matches and p95 latency. With visual models and images configured, append `--mode clip --mode vlm` to compare visual retrieval and reranking on the same queries. [Dataset fields and examples](https://amazon-reviews-2023.github.io/#data-fields)

After importing the sample and its simulated offers, try “Find black shoes in US size 8 under $60.” For a review question, try “How do Bandolino BERTA platform pumps fit? Do reviewers say they run small?”

## Start with the API

| Operation | Entry point |
|---|---|
| Research and refinement | `POST /v1/tasks`, then `POST /v1/tasks/{task_id}/turns`. Include `expected_version` and the displayed `plan_version` for edits. |
| Progress and clarification | `GET /v1/runs/{run_id}` returns the run. `GET /v1/runs/{run_id}/events` streams progress and recommendations and supports `Last-Event-ID`. Reply to a pending clarification at `POST /v1/runs/{run_id}/clarification`. |
| Catalog and review evidence | `GET /v1/catalog/search` and `GET /v1/products/{product_id}/evidence`. The Agent uses these catalog capabilities through its own read-only research tools. |
| Task trace | `GET /v1/tasks/{task_id}/trace` lists account-scoped events from research through a cart change, quote and checkout linked to a selected plan. The run result includes stage timings and candidate rejection reasons. |
| Compare, cart and checkout | The run or task includes structured plans. Submit the selected plan and its version to `/v1/cart/changes` or `/v1/quotes`. Confirm a quote by posting its exact `terms` value to `/v1/quotes/{id}/confirm`, then poll the checkout attempt. |
| Order and returns | `GET /v1/orders` shows per-seller parcels. Preview a return on the delivered line, then confirm the preview and reason. |
| Price tracking and replenishment | `POST /v1/automations` binds exact offers or a saved search to a price, stock or schedule trigger. Search-based cart and purchase rules require explicitly allowed SKUs and sellers. Read the rules and execution history, revoke rules, and see in-app notices. |

For a separate platform process over HTTP, set `PLATFORM_URL=http://127.0.0.1:8001`. For MCP, use `PLATFORM_URL=http://127.0.0.1:8001/mcp` and `PLATFORM_TRANSPORT=mcp`. The platform registers 12 commerce tools covering accounts, offers, price history, quotes, cart, checkout, orders and returns. For imported products, point `CATALOG_DATABASE` at the platform catalog when running the Agent separately. A platform adapter must validate the account and quote, support stable operation IDs and allow querying the original checkout attempt after a timeout.

### Failure drills

In local mode the Agent exposes demo controls under `/demo`. In Compose the sandbox exposes the controls at port 8001 with bearer token `compose-platform-token`. For example set the next payment outcome via `PUT /demo/payment?mode=lost_response`. After confirmation the Agent queries the original attempt and returns the existing order. To test repricing, update `unit_price` in cents on an offer with `PATCH /demo/offers/{offer_id}` after opening checkout. The platform returns updated terms for a new confirmation.

## Regression tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m shopping_agent evaluate-journeys
.\.venv\Scripts\python.exe -m shopping_agent evaluate --k 3
```

The tests cover HTTP and MCP, bundle refinements, checkpoint and payment recovery, quote confirmation, cart and return idempotency, and authorized automation. Catalog tests cover Reviews’23 ingestion, SKU filtering, review passages, verified streaming recommendations and ranking metrics. The Compose stack provides the integration path for PostgreSQL, Elasticsearch and Redis.

`evaluate-journeys` replays 109 labeled English shopping journeys across 131 turns in temporary databases. Cases cover typed SKU constraints, connector compatibility, stock and shipping changes, delivery deadlines, review evidence and multi-turn refinements. The report includes intent and constraint accuracy, eligible bundle hit rate, returned-plan constraint satisfaction, multi-turn retention and p50/p95 timings. Case labels can also check the first-ranked SKU bundle, exact offers, quoted totals and expected review IDs. The default run uses seeded electronics and simulated fashion fixtures. Use `--cases` to select a case file and `--with-models` to exercise configured model services. A failed check produces a nonzero exit status.

`evaluate` accepts binary relevance through `relevant` or graded relevance through a `relevance` map. Use grades 0–3 from irrelevant to highly relevant, and label either document IDs or product IDs consistently within each query. The report includes `ndcg` per query and `mean_ndcg` at the requested `--k`. Repeated hits for the same judged product earn credit once. Queries with no positive judgments are excluded from the mean.
