# Shopping Agent V2

An English-language shopping backend. FastAPI exposes versioned shopping tasks, catalog research, cart, checkout, orders, returns and price automation. LangGraph orchestrates the research turn. The platform sandbox exposes the same commerce contract over HTTP and MCP Streamable HTTP. The browser demo exercises the API.

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

```powershell
docker compose up --build -d
docker compose ps
```

The API listens on port 8000. The sandbox exposes Swagger at port 8001 and MCP at `/mcp`. Compose runs PostgreSQL, Redis, Elasticsearch, an index build, the FastAPI app and a separate Worker. The Agent and sandbox have separate databases. The API connects to the sandbox via MCP and receives offer events over the authenticated internal endpoint. PostgreSQL, Redis and Elasticsearch use the private Compose network.

```powershell
docker compose logs api worker platform
docker compose stop
```

Compose and the demo use test credentials. A non-demo deployment uses `DEMO=0`, JWTs from a trusted issuer, a platform sandbox or service URL, separate secrets, and secure infrastructure. Automatic buying is enabled explicitly with `AUTOBUY_ENABLED=1`. `CHECKOUT_ENABLED=0` pauses new checkout and return confirmations while previously accepted checkout attempts continue reconciliation.

## Configure models

Copy `.env.example` to `.env` as needed. Set `MODEL_URL` to a compatible API base URL including `/v1`, then set `CHAT_MODEL` and `MODEL_KEY` as required. Open-ended turns use a structured intent prompt, bounded read-only function calls, and a grounded review-excerpt prompt. The service validates tool arguments and verifies that cited excerpts exist in the cited evidence. It renders prices, delivery and verified compatibility from structured service results.

Set `EMBEDDING_MODEL` with an embeddings endpoint to use semantic text vectors. Set `RERANK_URL` and `RERANK_MODEL` for a `POST /rerank` service returning indexed relevance scores. Local defaults use English parsing, BM25 and a deterministic demo vector. The fallback supports the built-in scenarios and regression tests. Real models are configurable.

Elasticsearch uses separate aliases for SKU search and review evidence. Keyword and dense-vector searches run in parallel and RRF fuses their ranks. The bundle planner checks known SKU requirements before optional reranking and shortlisting, including RAM, weight, dock output and host capability. It checks the selected products together, then validates platform quotes against the budget and delivery deadline. Final ordering uses quoted totals and arrival times with the user's price, weight or delivery preference. Index mappings specify English text analysis and HNSW. Run the index command after catalog changes or to include SKU identifiers in an existing index. The command bulk-writes both new indices before switching their aliases.

For visual search, install `.[vision]`, set `CLIP_MODEL`, and reindex with `--images PATH`. Supply local images named by catalog document ID with a JPEG, PNG or WebP extension. Specific appearance requests or an `image_base64` turn field can trigger the optional image route. VLM reranking and IVF comparisons are future evaluation work.

## Import Amazon Reviews’23

The importer accepts local JSONL and gzip-compressed JSONL. It streams records and keeps a bounded, repeatable subset in file order. It recognizes the raw and exported metadata layouts, including structured and serialized details, both image layouts, nullable USD snapshot prices, and review timestamp and helpful-vote aliases.

```powershell
.\.venv\Scripts\python.exe -m shopping_agent import-reviews23 --metadata .\fixtures\reviews23_meta.jsonl --reviews .\fixtures\reviews23_reviews.jsonl
.\.venv\Scripts\python.exe -m shopping_agent evaluate
```

The fixture files are synthetic examples of the schemas. To use real data, pass the downloaded metadata and review files, optionally with `--products 1000 --reviews-per-product 20`. In Compose, mount the data directory into the indexer:

```powershell
$datasetPath = (Resolve-Path -LiteralPath '.\fixtures').Path
docker compose run --rm --volume "${datasetPath}:/import:ro" indexer import-reviews23 --metadata /import/reviews23_meta.jsonl --reviews /import/reviews23_reviews.jsonl
docker compose run --rm indexer reindex
docker compose run --rm indexer evaluate
```

Metadata and reviews join on `parent_asin`. A child review ASIN remains a source reference. Imported dataset products can appear in research results and reviews, with USD snapshot prices labeled separately. Exact SKU compatibility and purchasable offers use the platform catalog and quote contract. Supply English relevance labels in the JSONL search-case format to evaluate a different catalog subset. [Dataset fields and examples](https://amazon-reviews-2023.github.io/#data-fields)

## Start with the API

| Operation | Entry point |
|---|---|
| Research and refinement | `POST /v1/tasks`, then `POST /v1/tasks/{task_id}/turns`. Include `expected_version` and the displayed `plan_version` for edits. |
| Progress and clarification | `GET /v1/runs/{run_id}` or `GET /v1/runs/{run_id}/events` with optional `Last-Event-ID`. Reply to a pending interruption at `POST /v1/runs/{run_id}/clarification`. |
| Task trace | `GET /v1/tasks/{task_id}/trace` lists account-scoped events from research through a cart change, quote and checkout linked to a selected plan. The run result includes stage timings and candidate rejection reasons. |
| Compare, cart and checkout | The run or task includes structured plans. Submit the selected plan and its version to `/v1/cart/changes` or `/v1/quotes`. Confirm a quote by posting its exact `terms` value to `/v1/quotes/{id}/confirm`, then poll the checkout attempt. |
| Order and returns | `GET /v1/orders` shows per-seller parcels. Preview a return on the delivered line, then confirm the preview and reason. |
| Price tracking and replenishment | `POST /v1/automations` binds exact offers or a saved search to a price, stock or schedule trigger. Search-based cart and purchase rules require explicitly allowed SKUs and sellers. Read the rules and execution history, revoke rules, and see in-app notices. |

For a separate platform process over HTTP set `PLATFORM_URL=http://127.0.0.1:8001`. For MCP use `PLATFORM_URL=http://127.0.0.1:8001/mcp` and `PLATFORM_TRANSPORT=mcp`. Both use the authenticated tool names exposed by the sandbox. A real platform adapter must validate the account and quote, support stable operation IDs and allow querying the original checkout attempt after a timeout.

### Failure drills

In local mode the Agent exposes demo controls under `/demo`. In Compose the sandbox exposes the controls at port 8001 with bearer token `compose-platform-token`. For example set the next payment outcome via `PUT /demo/payment?mode=lost_response`. After confirmation the Agent queries the original attempt and returns the existing order. To test repricing, update `unit_price` in cents on an offer with `PATCH /demo/offers/{offer_id}` after opening checkout. The platform returns updated terms for a new confirmation.

## Regression tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m shopping_agent evaluate-journeys
```

The tests exercise the HTTP API, English bundle edits, SQLite checkpoint recovery, provider-style tool calling, strict quote confirmation, lost payment recovery, account isolation, cart and return idempotency, background authorization and deduplication, Reviews’23 ingestion, and the real MCP transport over loopback. The separate Compose stack is the PostgreSQL, Elasticsearch and Redis integration path.

`evaluate-journeys` replays 20 labeled English shopping journeys in temporary seeded databases. It reports intent and constraint extraction, eligible bundle hit rate, returned-plan constraint satisfaction, preservation of earlier constraints and selected offers, and p50/p95 timings. It exits with a nonzero status when a labeled check fails. Use `--cases` to supply compatible labels and `--with-models` to exercise configured model services. The default run uses deterministic demo data and model fallbacks.
