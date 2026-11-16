from __future__ import annotations

import json
from dataclasses import replace
from threading import Event, Thread

import httpx
import pytest

from shopping_agent.attributes import Condition
from shopping_agent.core import Settings
from shopping_agent.dataset import import_reviews23
from shopping_agent.models import Models
from shopping_agent.planner import candidate_rejection
from shopping_agent.runtime import Runtime, Worker


@pytest.fixture
def runtime(tmp_path):
    folder = tmp_path.as_posix()
    settings = Settings(database=f"sqlite:///{folder}/agent.db",
                        platform_database=f"sqlite:///{folder}/platform.db", embedded_worker=False)
    service = Runtime(settings)
    try:
        yield service
    finally:
        service.close()


def test_configured_constraints_apply_to_one_variant():
    conditions = [
        Condition(category="shoes", attribute="color", operator="eq", value="black").model_dump(),
        Condition(category="shoes", attribute="shoe_size_us", operator="eq", value=8).model_dump(),
        Condition(category="shoes", attribute="material", operator="contains_any", value=["leather"],
                  required=False).model_dump(),
    ]
    goal = {"categories": ["shoes"], "conditions": conditions}
    item = {"category": "shoes", "sku_id": "shoe"}
    assert candidate_rejection({**item, "specs": {"color": "black", "shoe_size_us": 7}}, goal)
    assert candidate_rejection({**item, "specs": {"color": "white", "shoe_size_us": 8}}, goal)
    assert candidate_rejection({**item, "specs": {"color": "black"}}, goal) == "unverified_shoe_size_us"
    assert candidate_rejection({**item, "specs": {"color": "black", "shoe_size_us": 8}}, goal) is None
    with pytest.raises(ValueError):
        Condition(category="shoes", attribute="shoe_size_us", operator="eq", value=True)


def test_platform_import_review_recall_and_long_review_citation(runtime, tmp_path):
    metadata = tmp_path / "meta.jsonl"
    review_file = tmp_path / "reviews.jsonl"
    product_id = "reviews23-EXAMPLE77"
    exact = "The zorbium lining stays comfortable while standing all day."
    metadata.write_text(json.dumps({"parent_asin": "EXAMPLE77", "title": "Harbor Work Shoes"}) + "\n", encoding="utf-8")
    review_file.write_text(json.dumps({
        "parent_asin": "EXAMPLE77", "asin": "EXAMPLE78", "text": "Generic opening. " * 80 + exact, "rating": 4,
    }) + "\n", encoding="utf-8")
    assert import_reviews23(runtime.platform.store, metadata, review_file)["reviews_added"] == 1
    assert runtime.store.get("catalog", product_id) is None
    assert runtime.catalog.get("catalog", product_id)["category"] == "shoes"
    docs, trace = runtime.catalog.search("zorbium lining while standing all day", "shoes")
    assert product_id in trace["review_products"]
    assert docs[0]["product_id"] == product_id
    evidence = runtime.catalog.evidence([product_id], "zorbium lining", external_asin="EXAMPLE78")
    assert exact in evidence[0]["text"]
    assert evidence[0]["id"] != evidence[0]["record_id"]


def test_model_consumes_streamed_partial_frames():
    models = Models(replace(Settings(), model_url="https://mock.test/v1", chat_model="test-model"))
    answers = ['{"candidate_', 'id":"a"}\n{"candidate_id":"b"}']
    stream = "".join("data: " + json.dumps({"choices": [{"delta": {"content": part}}]}) + "\n\n" for part in answers)
    stream += "data: [DONE]\n\n"

    def respond(request):
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, text=stream, headers={"Content-Type": "text/event-stream"})

    models.http.close()
    with httpx.Client(transport=httpx.MockTransport(respond)) as models.http:
        assert [json.loads(line) for line in models.stream_lines([])] == [{"candidate_id": "a"}, {"candidate_id": "b"}]


def test_visual_rerank_sends_images_and_preserves_candidates_without_images(runtime, tmp_path, monkeypatch):
    settings = replace(runtime.settings, model_url="https://mock.test/v1", vision_chat_model="test-vision",
                       catalog_images=str(tmp_path))
    runtime.catalog.settings = runtime.models.settings = settings
    docs = [{"id": key, "title": f"Shoe {key}"} for key in ("a", "b", "c")]
    for key in ("a", "c"):
        (tmp_path / f"{key}.jpg").write_bytes(b"catalog image")
    monkeypatch.setattr(runtime.models, "_image_part",
                        lambda _: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}})

    def respond(request):
        body = json.loads(request.content)
        assert body["model"] == "test-vision"
        parts = body["messages"][1]["content"]
        assert sum(part["type"] == "image_url" for part in parts) == 3
        assert "Shoe a" in str(parts) and "Shoe c" in str(parts)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ranked_ids":["c","a"]}'}}]})

    runtime.models.http.close()
    with httpx.Client(transport=httpx.MockTransport(respond)) as runtime.models.http:
        ordered, ran = runtime.catalog.visual_rerank("square toe", docs, b"reference image")
    assert ran and [doc["id"] for doc in ordered] == ["c", "b", "a"]


def test_run_publishes_first_validated_recommendation_during_generation(runtime, monkeypatch):
    runtime.models.settings = replace(runtime.settings, model_url="https://mock.test/v1", chat_model="test-model")

    def chat(messages, **_):
        if "exact review excerpts" in messages[0]["content"]:
            return {"content": '{"highlights":[]}'}
        return {"content": "{}"}

    release = Event()
    streaming = Event()

    def stream(messages):
        payload = json.loads(messages[-1]["content"])
        candidate, other = payload["candidates"][:2]
        proof, foreign = candidate["evidence"][0], other["evidence"][0]
        for citation in (foreign, proof):
            yield json.dumps({"candidate_id": candidate["candidate_id"], "reason": "Matches the catalog listing.",
                              "citations": [{"id": citation["id"], "excerpt": citation["text"][:100]}]})
        streaming.set()
        assert release.wait(4)

    monkeypatch.setattr(runtime.models, "chat", chat)
    monkeypatch.setattr(runtime.models, "stream_lines", stream)
    worker = Worker(runtime)
    task = runtime.runs.start_task("alex")
    run = runtime.runs.start_turn("alex", task["id"], {
        "text": "Find a laptop", "request_id": None, "expected_version": None, "plan_version": None,
        "address_id": None, "timezone": "America/New_York", "image_base64": None,
    })
    thread = Thread(target=worker.tick, args=("runs",))
    thread.start()
    try:
        assert streaming.wait(4)
        from time import monotonic, sleep
        deadline = monotonic() + 2
        while monotonic() < deadline:
            current = runtime.store.get("runs", run["id"])
            events = [event for event in current["events"] if event["state"] == "recommendation"]
            if events:
                break
            sleep(0.02)
        assert current["state"] == "running" and len(events) == 1
        assert events[0]["reason"] == "Matches the catalog listing."
    finally:
        release.set()
        thread.join(5)
    finished = runtime.store.get("runs", run["id"])
    assert finished["state"] == "completed"
    assert len(finished["result"]["recommendations"]) == 3
