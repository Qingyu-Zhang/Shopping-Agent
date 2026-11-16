from __future__ import annotations

import json
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import replace
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from .api import Turn
from .attributes import active_conditions
from .core import Settings, elapsed_ms
from .dataset import records
from .platform import OfferRecord
from .runtime import Runtime, Worker

# Freeze delivery calculations independently of the actual evaluation date.
EVALUATION_TIME = 1_795_824_000.0
ACCOUNT = "alex"


def _distribution(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return {"n": len(ordered), "p50": ordered[ceil(len(ordered)*0.5)-1],
            "p95": ordered[ceil(len(ordered)*0.95)-1]}


def _ratio(values):
    return sum(values)/len(values) if values else None


def _hard_pass(plan, goal, forbidden):
    quote = plan["quote"]
    if ((goal.get("budget") is not None and quote["total"] > goal["budget"])
        or (goal.get("deadline") is not None and quote["eta"] > goal["deadline"])):
        return False
    for item in plan["items"]:
        if item["sku_id"] in forbidden:
            return False
        if item["category"] == "laptop" and goal.get("min_ram"):
            if item["specs"].get("ram_gb", 0) < goal["min_ram"]:
                return False
        weight_limit = goal.get("max_weights", {}).get(item["category"])
        if weight_limit is not None and item["specs"].get("weight_g", float("inf")) > weight_limit:
            return False
        if any(condition.required and condition.matches(item["specs"].get(condition.attribute)) is not True
               for condition in active_conditions(goal, item["category"])):
            return False
    return True


def _record_stages(stages, trace):
    for stage, value in trace.get("timings_ms", {}).items():
        stages[stage].append(value)
    for slot in trace.get("search", {}).values():
        for stage, value in slot.get("timings_ms", {}).items():
            stages["catalog_" + stage].append(value)
    for attempt in trace.get("planning", {}).get("passes", []):
        for stage, value in attempt.get("timings_ms", {}).items():
            stages["planner_" + stage].append(value)


def _grade_turn(label, current, prior, k, scores):
    failures = []
    result = current.get("result", {})
    plans = result.get("plans", [])
    actual_goal = result.get("goal", {})
    old_goal = prior.get("goal", {})
    expected_goal = label.get("expect_goal", {})
    expected_state = label.get("state", "completed")
    if current["state"] != expected_state:
        failures.append(f'expected state {expected_state}; got {current["state"]}')
    if "action" in label:
        passed = result.get("trace", {}).get("action") == label["action"]
        scores["intent"].append(passed)
        if not passed:
            failures.append("unexpected action")
    for key, expected in expected_goal.items():
        passed = actual_goal.get(key) == expected
        scores["constraints"].append(passed)
        if not passed:
            failures.append(f"unexpected constraint {key}")
    for key in label.get("retain", []):
        passed = key in old_goal and actual_goal.get(key) == old_goal[key]
        scores["retention"].append(passed)
        if not passed:
            failures.append(f"constraint {key} was not retained")
    if label.get("preserve"):
        option, category = label["preserve"]["option"], label["preserve"]["category"]
        previous_plans = prior.get("plans", [])
        fixed = {item["offer_id"] for item in previous_plans[option-1]["items"] if item["category"] == category
                 } if len(previous_plans) >= option >= 1 else set()
        preserved = bool(fixed and plans and all(fixed <= {x["offer_id"] for x in plan["items"]} for plan in plans))
        scores["retention"].append(preserved)
        if not preserved:
            failures.append(f"selected {category} offer was not retained")
    if "eligible_bundles" in label:
        observed = {frozenset(x["sku_id"] for x in plan["items"]) for plan in plans[:k]}
        expected = {frozenset(ids) for ids in label["eligible_bundles"]}
        hit = bool(observed & expected)
        scores["bundle_hit"].append(hit)
        if not hit:
            failures.append(f"no labeled eligible bundle in top {k}")
    if label.get("max_plans") is not None and len(plans) > label["max_plans"]:
        failures.append("too many purchasable plans")
    first = plans[0] if plans else {}
    if "top_bundle" in label and {item["sku_id"] for item in first.get("items", [])} != set(label["top_bundle"]):
        failures.append("unexpected first-ranked bundle")
    quote = first.get("quote", {})
    if "top_offers" in label and {line["offer_id"] for line in quote.get("lines", [])} != set(label["top_offers"]):
        failures.append("unexpected first-ranked offers")
    if any(quote.get(key) != value for key, value in label.get("top_quote", {}).items()):
        failures.append("unexpected first-ranked quote")
    if "top_product" in label:
        first_product = next(iter(result.get("products", [])), {})
        if first_product.get("product_id") != label["top_product"]:
            failures.append("unexpected first-ranked product")
    observed_evidence = {item.get("record_id") or item["id"] for item in result.get("evidence", [])}
    if not set(label.get("evidence_ids", [])) <= observed_evidence:
        failures.append("missing expected review evidence")
    if "suggestion" in label and result.get("suggested_action", {}).get("type") != label["suggestion"]:
        failures.append("unexpected action suggestion")
    if "target" in label and result.get("suggested_action", {}).get("target") != label["target"]:
        failures.append("unexpected price target")
    forbidden = set(label.get("forbidden_skus", []))
    hard = {**actual_goal, **expected_goal}
    if (forbidden or any(hard.get(field) is not None for field in ("budget", "deadline", "min_ram"))
        or hard.get("max_weights") or any(condition["required"] for condition in hard.get("conditions", []))):
        for plan in plans:
            passed = _hard_pass(plan, hard, forbidden)
            scores["eligible_plans"].append(passed)
            if not passed:
                failures.append("a returned plan violated a hard constraint")
    return failures


def _load_catalog_fixture(runtime, path):
    if not path.is_file():
        return
    fixture = json.loads(path.read_text(encoding="utf-8"))
    with runtime.platform.store.transaction(True) as tx:
        for item in fixture["catalog"]:
            sku = item.get("sku_id")
            doc = {**item, "market": "US", "active": True, "source": "Benchmark fixture", "source_version": "1"}
            tx.put("catalog", doc["id"], doc, owner=doc["product_id"], once=True)
            if sku:
                offer = OfferRecord.model_validate({
                    "id": f"{sku}-offer", "sku_id": sku, "seller": "Benchmark Shop",
                    "title": doc["title"], "variant": doc["variant"], "unit_price": doc["snapshot_price"],
                    "currency": "USD", "stock": 5, "shipping": doc.get("shipping", 0), "lead_hours": doc["lead_hours"],
                }).model_dump()
                tx.put("offers", offer["id"], offer, owner=sku, once=True)
        for item in fixture["evidence"]:
            doc = {**item, "kind": "review", "source": "Benchmark review fixture"}
            tx.put("evidence", doc["id"], doc, owner=doc["product_id"], once=True)


@contextmanager
def _override_offers(store, overrides):
    if not overrides:
        yield
        return
    originals = {}
    with store.transaction(True) as tx:
        for key, changes in overrides.items():
            offer = tx.require("offers", key)
            originals[key] = offer
            updated = OfferRecord.model_validate({**offer, **changes}).model_dump()
            tx.put("offers", key, updated, owner=offer["sku_id"])
    try:
        yield
    finally:
        with store.transaction(True) as tx:
            for key, offer in originals.items():
                tx.put("offers", key, offer, owner=offer["sku_id"])


def _run_cases(runtime, cases, k):
    worker = Worker(runtime)
    rows, durations = [], []
    scores, stages = defaultdict(list), defaultdict(list)
    for case in cases:
        with _override_offers(runtime.platform.store, case.get("offer_overrides", {})):
            task = runtime.runs.start_task(ACCOUNT)
            pending = None
            for position, label in enumerate(case["turns"], 1):
                prior = runtime.store.get("tasks", task["id"], ACCOUNT)
                started = perf_counter()
                if label.get("clarification"):
                    if not pending:
                        raise ValueError(f'{case["id"]} has a clarification without a preceding interrupt.')
                    runtime.runs.answer(ACCOUNT, pending["id"], pending["interrupt"]["id"], label["text"])
                    run = pending
                else:
                    values = Turn(text=label["text"], expected_version=prior["version"],
                                  address_id=label.get("address_id")).model_dump()
                    run = runtime.runs.start_turn(ACCOUNT, task["id"], values)
                worker.tick("runs")
                current = runtime.store.get("runs", run["id"], ACCOUNT)
                pending = current if current["state"] == "needs_input" else None
                duration = elapsed_ms(started)
                durations.append(duration)
                _record_stages(stages, current.get("result", {}).get("trace", {}))
                rows.append({"case": case["id"], "turn": position, "state": current["state"], "latency_ms": duration,
                             "failures": _grade_turn(label, current, prior, k, scores)})
    return {
        "cases": len(cases), "turns": len(rows), "failed_turns": sum(bool(x["failures"]) for x in rows), "top_k": k,
        "intent_accuracy": _ratio(scores["intent"]), "constraint_extraction_accuracy": _ratio(scores["constraints"]),
        "eligible_bundle_hit_rate": _ratio(scores["bundle_hit"]),
        "hard_constraint_satisfaction": _ratio(scores["eligible_plans"]),
        "multi_turn_retention": _ratio(scores["retention"]),
        "latency_ms": _distribution(durations), "stage_latency_ms": {
            stage: _distribution(values) for stage, values in sorted(stages.items())
        }, "results": rows,
    }


def evaluate_journeys(path, k=3, *, model_settings=None):
    path = Path(path)
    counters = Counter()
    cases = list(records(path, counters))
    if counters["invalid_rows"] or not cases:
        raise ValueError("Expected a nonempty JSONL file with valid shopping evaluation cases.")
    if len({case["id"] for case in cases}) != len(cases) or any(not case["turns"] for case in cases):
        raise ValueError("Expected unique case IDs and at least one turn in each case.")
    primary = model_settings or Settings()
    with TemporaryDirectory(prefix="shopping-eval-") as directory:
        folder = Path(directory).as_posix()
        settings = replace(primary, database=f"sqlite:///{folder}/agent.db",
                           platform_database=f"sqlite:///{folder}/platform.db", platform_url="", catalog_database="",
                           redis_url="", elastic_url="", clip_model="", catalog_images="", demo=True, embedded_worker=False)
        runtime = Runtime(settings, clock=lambda: EVALUATION_TIME)
        try:
            _load_catalog_fixture(runtime, path.with_suffix(".catalog.json"))
            return _run_cases(runtime, cases, k)
        finally:
            runtime.close()
