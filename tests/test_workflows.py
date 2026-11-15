from __future__ import annotations

import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from shopping_agent.api import create_app
from shopping_agent.core import Settings
from shopping_agent.dataset import import_reviews23
from shopping_agent.platform import RemotePlatform, create_platform_app
from shopping_agent.runtime import Runtime, Worker


AUTH = {"Authorization": "Bearer demo-alex"}
SECOND = {"Authorization": "Bearer demo-sam"}
BUNDLE = "Find a lightweight laptop and a dock that can drive a 4K 60 Hz monitor, under $1,000 total."


class Clock:
    def __init__(self):
        self.now = 1_800_000_000

    def __call__(self):
        return self.now


@pytest.fixture
def config(tmp_path):
    return replace(Settings(), database=f"sqlite:///{tmp_path.as_posix()}/agent.db",
                   platform_database=f"sqlite:///{tmp_path.as_posix()}/platform.db",
                   embedded_worker=False, autobuy_enabled=True)


@pytest.fixture
def api(config):
    clock = Clock()
    app = create_app(config, clock)
    with TestClient(app, headers=AUTH) as client:
        yield client, app.state.runtime, Worker(app.state.runtime), clock


def get_json(response, status=200):
    assert response.status_code == status, response.text
    return response.json()


def turn(client, worker, text, task=None, **fields):
    if not task:
        task = get_json(client.post("/v1/tasks"), 201)["id"]
    row = get_json(client.post(f"/v1/tasks/{task}/turns", json={"text": text, **fields}), 202)
    assert worker.tick("runs")
    return task, get_json(client.get(f'/v1/runs/{row["id"]}'))


def quote(client, *lines):
    return get_json(client.post("/v1/quotes", json={"lines": [
        {"offer_id": key, "quantity": 1} for key in lines
    ]}), 201)


def confirm(client, receipt):
    return get_json(client.post(f'/v1/quotes/{receipt["id"]}/confirm', json={"terms": receipt["terms"]}), 202)


def test_english_bundle_followup_and_versioned_actions(api):
    client, runtime, worker, _ = api
    task, row = turn(client, worker, BUNDLE, request_id="baseline")
    assert row["state"] == "completed", row
    result = row["result"]
    assert result["goal"]["budget"] == 100_000
    assert result["goal"]["categories"] == ["dock", "laptop"]
    assert len(result["plans"]) >= 2
    assert all(p["quote"]["total"] <= 100_000 and p["evidence"] for p in result["plans"])
    assert "event: completed" in client.get(f'/v1/runs/{row["id"]}/events').text
    original = next(x for x in result["plans"][1]["items"] if x["category"] == "laptop")
    second = result["plans"][1]
    _, edit = turn(client, worker, "Option 2, keep the laptop and replace the dock with a lighter dock.",
                   task, expected_version=1, plan_version=1)
    assert edit["state"] == "completed", edit
    assert edit["result"]["goal"]["budget"] == 100_000
    assert edit["result"]["plans"]
    for whole in edit["result"]["plans"]:
        assert original["offer_id"] in [item["offer_id"] for item in whole["items"]]
    receipt = get_json(client.post("/v1/quotes", json={
        "selection": {"task_id": task, "plan_id": second["id"], "plan_version": 1}
    }), 201)
    assert receipt["quote"]["total"] == second["quote"]["total"]
    assert client.get(f"/v1/tasks/{task}", headers=SECOND).status_code == 404
    assert client.get(f'/v1/runs/{row["id"]}', headers=SECOND).status_code == 404


def test_clarification_survives_restart(config):
    clock = Clock()
    first = create_app(config, clock)
    with TestClient(first, headers=AUTH) as client:
        task, row = turn(client, Worker(first.state.runtime), "Find a dock that can drive a 4K 60 Hz monitor.")
        assert row["state"] == "needs_input", row
        key, interrupt = row["id"], row["interrupt"]["id"]
    next_app = create_app(config, clock)
    with TestClient(next_app, headers=AUTH) as client:
        get_json(client.post(f"/v1/runs/{key}/clarification", json={
            "interrupt_id": interrupt, "text": "My laptop supports DisplayPort 1.4 Alt Mode."
        }), 202)
        assert Worker(next_app.state.runtime).tick("runs")
        resumed = get_json(client.get(f"/v1/runs/{key}"))
        assert resumed["state"] == "completed", resumed
        assert resumed["result"]["plans"]
        assert resumed["result"]["goal"]["categories"] == ["dock"]


def test_tool_calls_are_executed_with_read_only_scope(api):
    client, runtime, worker, _ = api
    # Replace only the chat boundary. The actual graph and retrieval still execute.
    sequence = iter([
        {"content": "{}"},
        {"tool_calls": [
            {"id": "allowed", "type": "function", "function": {
                "name": "search_products", "arguments": '{"category":"dock","query":"compact 4K HDMI dock"}'}},
            {"id": "injected", "type": "function", "function": {
                "name": "commerce_submit", "arguments": "{}"}},
        ]},
        {"content": "done"},
        {"content": '{"highlights":[{"evidence_id":"review-cedar-16","excerpt":"The screen is bright."},'
                    '{"evidence_id":"review-cedar-16","excerpt":"Verified 12 hours of battery."}]}'},
    ])
    runtime.models.settings = replace(runtime.settings, model_url="https://mock.test/v1", chat_model="test-tool-caller")
    def respond(request):
        json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": next(sequence)}]})
    runtime.models.http.close()
    runtime.models.http = httpx.Client(transport=httpx.MockTransport(respond))
    _, row = turn(client, worker, BUNDLE)
    assert row["state"] == "completed", row
    assert row["result"]["trace"]["tools"] == [
        {"tool": "search_products", "ok": True}, {"tool": "unavailable", "ok": False}
    ]
    assert all("Verified 12 hours" not in item["excerpt"] for item in row["result"]["review_highlights"])
    assert get_json(client.get("/v1/orders")) == []


def test_repricing_binds_the_exact_quote(api):
    client, runtime, worker, _ = api
    original = quote(client, "air-16-official")
    assert client.post(f'/v1/quotes/{original["id"]}/confirm', json={
        "terms": original["terms"]}, headers=SECOND).status_code == 404
    runtime.platform.change_offer("air-16-official", {"unit_price": 74_900})
    result = confirm(client, original)
    assert worker.tick("checkouts")
    final = get_json(client.get(f'/v1/checkouts/{result["id"]}'))
    assert final["state"] == "requote"
    assert final["replacement"]["quote"]["total"] == 74_900
    assert confirm(client, original)["id"] == result["id"]
    assert get_json(client.get("/v1/orders")) == []
    updated = confirm(client, final["replacement"])
    assert worker.tick("checkouts")
    assert get_json(client.get(f'/v1/checkouts/{updated["id"]}'))["state"] == "succeeded"


def test_unknown_payment_recovers_once_and_orders_are_isolated(config):
    clock = Clock()
    first = create_app(config, clock)
    with TestClient(first, headers=AUTH) as client:
        receipt = quote(client, "air-16-official", "link-mini-official")
        first.state.runtime.platform.payment_fault("lost_response")
        with ThreadPoolExecutor(max_workers=4) as pool:
            attempts = list(pool.map(lambda _: confirm(client, receipt), range(4)))
        assert len({row["id"] for row in attempts}) == 1
        key = attempts[0]["id"]
        assert Worker(first.state.runtime).tick("checkouts")
        assert get_json(client.get(f"/v1/checkouts/{key}"))["state"] == "processing"
        assert len(first.state.runtime.platform.store.list("payments")) == 1
    clock.now += 20
    second = create_app(config, clock)
    with TestClient(second, headers=AUTH) as client:
        assert Worker(second.state.runtime).tick("checkouts")
        result = get_json(client.get(f"/v1/checkouts/{key}"))
        assert result["state"] == "succeeded"
        assert len(get_json(client.get("/v1/orders"))) == 1
        assert len(second.state.runtime.platform.store.list("payments")) == 1
        assert client.get(f'/v1/orders/{result["order_id"]}', headers=SECOND).status_code == 404


def test_cart_revision_idempotency_and_split_return(api):
    client, runtime, worker, _ = api
    change = {"expected_version": 0, "operation_id": "cart-1", "lines": [{"offer_id": "link-mini-official"}]}
    assert get_json(client.post("/v1/cart/changes", json=change))["version"] == 1
    assert get_json(client.post("/v1/cart/changes", json=change))["version"] == 1
    stale = client.post("/v1/cart/changes", json={**change, "operation_id": "cart-2"})
    assert stale.status_code == 409
    result = confirm(client, quote(client, "air-16-official", "link-mini-official"))
    assert worker.tick("checkouts")
    status = get_json(client.get(f'/v1/checkouts/{result["id"]}'))
    order = get_json(client.get(f'/v1/orders/{status["order_id"]}'))
    assert len(order["parcels"]) == 2
    delivered, other = order["parcels"]
    runtime.platform.update_parcel("alex", order["id"], delivered["id"], "delivered")
    endpoint = f'/v1/orders/{order["id"]}/return-quotes'
    assert client.post(endpoint, json={"offer_id": other["offer_ids"][0]}).status_code == 409
    returned = get_json(client.post(endpoint, json={"offer_id": delivered["offer_ids"][0]}), 201)
    submit = f'/v1/return-quotes/{returned["id"]}/confirm'
    body = {"terms": returned["terms"], "reason": "Not needed"}
    first = get_json(client.post(submit, json=body))
    assert get_json(client.post(submit, json=body))["id"] == first["id"]
    assert client.post(endpoint, json={"offer_id": delivered["offer_ids"][0]}).status_code == 409


def test_events_and_authorized_automation(api):
    client, runtime, worker, _ = api
    rules = {"lines": [{"offer_id": "link-mini-official"}], "threshold": 4000}
    watch = get_json(client.post("/v1/automations", json={**rules, "mode": "notify"}), 201)
    assert worker.tick("automations")
    assert get_json(client.get("/v1/notices")) == []
    runtime.platform.change_offer("link-mini-official", {"unit_price": 3900})
    worker.tick("offer_events")
    assert worker.tick("automations")
    assert not worker.tick("automations")
    assert len(get_json(client.get("/v1/notices"))) == 1
    auto = {"mode": "buy", "payment_method": "saved-card", "total_budget": 4000, **rules}
    revoked = get_json(client.post("/v1/automations", json=auto), 201)
    assert get_json(client.delete(f'/v1/automations/{revoked["id"]}'))["state"] == "revoked"
    entry = get_json(client.post("/v1/automations", json=auto), 201)
    assert worker.tick("automations")
    assert worker.tick("checkouts")
    assert len(get_json(client.get("/v1/orders"))) == 1
    executions = get_json(client.get(f'/v1/automations/{entry["id"]}/runs'))
    assert len(executions) == 1 and executions[0]["state"] == "succeeded"
    assert get_json(client.get(f'/v1/automations/{revoked["id"]}/runs')) == []
    assert get_json(client.get(f'/v1/automations/{watch["id"]}/runs'))[0]["mode"] == "notify"


def test_reviews23_import_is_repeatable_and_orderable_data_stays_separate(api):
    client, runtime, _, _ = api
    metadata, reviews = (Path("fixtures") / path for path in ("reviews23_meta.jsonl", "reviews23_reviews.jsonl"))
    assert import_reviews23(runtime.store, metadata, reviews)["reviews_added"] == 2
    assert "reviews_added" not in import_reviews23(runtime.store, metadata, reviews)
    products = get_json(client.get("/v1/catalog/search", params={"q": "Granite Travel USB-C Hub"}))
    item = next(x for x in products if x["product_id"] == "reviews23-DEMO23HUB1")
    assert item["sku_id"] is None and item["snapshot_price"] == 3999
    evidence = get_json(client.get("/v1/products/reviews23-DEMO23HUB1/evidence", params={"q": "compact cable"}))
    assert evidence and evidence[0]["source"] == "Amazon Reviews'23"
    assert evidence[0]["sku_id"] is None


def test_search_scope_automation_and_scheduled_cart_are_bounded(api):
    client, runtime, worker, clock = api
    task, row = turn(client, worker, BUNDLE)
    scope = {"task_id": task, "plan_version": row["result"]["task_version"]}
    assert client.post("/v1/automations", json={
        "mode": "cart", "trigger": "schedule", "every_days": 1, "search_scope": scope,
    }).status_code == 422
    created = get_json(client.post("/v1/automations", json={
        "mode": "cart", "trigger": "schedule", "every_days": 1,
        "search_scope": {**scope, "allowed_skus": ["air-16", "link-mini"], "allowed_sellers": ["Aurora", "Link Works"]},
    }), 201)
    assert not worker.tick("automations")
    clock.now += 86400
    assert worker.tick("automations")
    assert worker.tick("automation_runs")
    lines = get_json(client.get("/v1/cart"))["lines"]
    assert {line["offer_id"] for line in lines} == {"air-16-official", "link-mini-official"}
    assert get_json(client.get(f'/v1/automations/{created["id"]}/runs'))[0]["state"] == "succeeded"


def test_real_streamable_http_mcp_and_http_contract(config):
    app = create_platform_app(config)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    base = f"http://127.0.0.1:{sock.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started
        http = RemotePlatform(replace(config, platform_url=base, platform_transport="http"))
        mcp = RemotePlatform(replace(config, platform_url=base+"/mcp", platform_transport="mcp"))
        try:
            args = {"sku_ids": ["air-16", "link-mini"]}
            assert http.call("commerce_offers", args) == mcp.call("commerce_offers", args)
            profile = mcp.call("commerce_account", {"account_id": "alex"})
            assert profile["id"] == "alex"
            with httpx.Client() as untrusted:
                assert untrusted.post(base+"/tools/commerce_offers", json=args).status_code == 401
        finally:
            http.http.close()
            mcp.http.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
