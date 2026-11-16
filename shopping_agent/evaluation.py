from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from .api import Turn
from .core import Settings, elapsed_ms
from .dataset import records
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
    if "suggestion" in label and result.get("suggested_action", {}).get("type") != label["suggestion"]:
        failures.append("unexpected action suggestion")
    forbidden = set(label.get("forbidden_skus", []))
    hard = {**actual_goal, **expected_goal}
    if forbidden or any(hard.get(field) is not None for field in ("budget", "deadline", "min_ram")) or hard.get("max_weights"):
        for plan in plans:
            passed = _hard_pass(plan, hard, forbidden)
            scores["eligible_plans"].append(passed)
            if not passed:
                failures.append("a returned plan violated a hard constraint")
    return failures


def _run_cases(runtime, cases, k):
    worker = Worker(runtime)
    rows, durations = [], []
    scores, stages = defaultdict(list), defaultdict(list)
    for case in cases:
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
                values = Turn(text=label["text"], expected_version=prior["version"]).model_dump()
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
    counters = Counter()
    cases = list(records(path, counters))
    if counters["invalid_rows"] or not cases:
        raise ValueError("Expected a nonempty JSONL file with valid shopping evaluation cases.")
    primary = model_settings or Settings()
    with TemporaryDirectory(prefix="shopping-eval-") as directory:
        folder = Path(directory).as_posix()
        settings = replace(primary, database=f"sqlite:///{folder}/agent.db",
                           platform_database=f"sqlite:///{folder}/platform.db", platform_url="",
                           redis_url="", elastic_url="", clip_model="", demo=True, embedded_worker=False)
        runtime = Runtime(settings, clock=lambda: EVALUATION_TIME)
        try:
            return _run_cases(runtime, cases, k)
        finally:
            runtime.close()
