from __future__ import annotations

import heapq
import itertools
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from .core import Fault, elapsed_ms, stable_id
from .platform import Quote

VIDEO_PORTS = ("hdmi", "dp")


def specification(item, field):
    """Resolve the catalog value unless structured authoritative claims disagree."""
    specs = item.get("specs")
    value = specs.get(field) if isinstance(specs, dict) else None
    claims = [claim["value"] for claim in item.get("spec_claims") or []
              if isinstance(claim, dict) and claim.get("field") == field and claim.get("authoritative") is True
              and claim.get("source") and "value" in claim]
    values = ([value] if value is not None else []) + claims
    if values and any(other != values[0] for other in values[1:]):
        return None, f"conflicting_{field}"
    if value is None:
        return None, f"unverified_{field}"
    return value, None


def candidate_rejection(item, goal, *, excluded=None, allowed_skus=None):
    sku = item.get("sku_id")
    if not sku:
        return "unverified_sku"
    if sku == excluded or (allowed_skus is not None and sku not in allowed_skus):
        return "sku_excluded"
    category = item["category"]
    ceiling = goal.get("max_weights", {}).get(category)
    if ceiling is not None:
        weight, reason = specification(item, "weight_g")
        if reason or type(weight) is not int:
            return reason or "unverified_weight_g"
        if weight > ceiling:
            return "weight_limit"
    if category == "laptop" and goal.get("min_ram"):
        memory, reason = specification(item, "ram_gb")
        if reason or type(memory) is not int:
            return reason or "unverified_ram_gb"
        if memory < goal["min_ram"]:
            return "min_ram"
    needs_dock = category == "dock" and (goal.get("hz") or goal.get("ports") or "laptop" in goal["categories"])
    if needs_dock:
        inputs, reason = specification(item, "inputs")
        if reason or not isinstance(inputs, dict):
            return reason or "unverified_inputs"
        required_ports = goal.get("ports") or VIDEO_PORTS
        modes = [mode for mode in inputs.values()
                 if isinstance(mode, dict) and type(mode.get("min")) is int and isinstance(mode.get("outputs"), dict)]
        if len(modes) != len(inputs):
            return "unverified_inputs"
        rates = [mode["outputs"][port] for mode in modes for port in required_ports if port in mode["outputs"]]
        if any(type(rate) is not int for rate in rates):
            return "unverified_inputs"
        if not any(rate >= (goal.get("hz") or 1) for rate in rates):
            return "dock_output"
    if category == "laptop" and "dock" in goal["categories"]:
        host, reason = specification(item, "video")
        if reason or not isinstance(host, dict) or any(type(level) is not int for level in host.values()):
            return reason or "unverified_video"
        if not host:
            return "host_output"
    return None


def compatibility(items, goal):
    hz = goal.get("hz")
    ports = goal.get("ports") or VIDEO_PORTS
    dock = next((x for x in items if x["category"] == "dock"), None)
    laptop = next((x for x in items if x["category"] == "laptop"), None)
    screen = next((x for x in items if x["category"] == "monitor"), None)
    if not dock or (not hz and not laptop):
        return True, [], None
    host, reason = specification(laptop, "video") if laptop else (goal.get("host_video"), None)
    if reason or not host:
        return False, [], reason or "unverified_video"
    if screen:
        monitor_ports, reason = specification(screen, "monitor_ports")
        if reason or not isinstance(monitor_ports, list):
            return False, [], reason or "unverified_monitor_ports"
        ports = [port for port in ports if port in monitor_ports]
    inputs, reason = specification(dock, "inputs")
    if reason:
        return False, [], reason
    for protocol, mode in inputs.items():
        if host.get(protocol, -1) < mode["min"]:
            continue
        for port in ports:
            output = mode["outputs"].get(port)
            if type(output) is int and output >= (hz or 1):
                sources = [(dock, f'{protocol} input delivers 4K at {output} Hz over {port.upper()}.')]
                if laptop:
                    sources.insert(0, (laptop, f'Host supports {protocol} at level {host[protocol]}.'))
                if screen:
                    sources.append((screen, f'Monitor accepts {port.upper()}.'))
                facts = [{
                    "id": stable_id(item["id"], protocol, port), "sku_id": item["id"],
                    "source": item["source"], "source_version": item["source_version"], "text": text,
                } for item, text in sources]
                return True, facts, None
    return False, [], "bundle_compatibility"


def ranking_preference(goal):
    return goal.get("sort_by") or ("weight" if goal.get("light") else "relevance")


def bundle_weight(pair, fixed):
    weights = [specification(item, "weight_g")[0] for _, item, _ in pair if item["category"] not in fixed]
    if all(type(weight) is int for weight in weights):
        return sum(weights)
    return float("inf")


class Planner:
    def __init__(self, catalog, platform):
        self.catalog, self.platform = catalog, platform

    def quote(self, account, address, lines, *, expected_skus=None):
        args = {"account_id": account, "address_id": address, "lines": lines}
        try:
            quote = Quote.model_validate(self.platform.call("commerce_quote", args)).model_dump()
        except (ValueError, TypeError) as error:
            raise Fault("invalid_quote", "The platform returned an invalid quote.", 502) from error
        expected = {(x["offer_id"], x["quantity"]) for x in lines}
        actual = {(x["offer_id"], x["quantity"]) for x in quote["lines"]}
        if (quote["account_id"] != account or quote["address_id"] != address or expected != actual
            or len(actual) != len(quote["lines"])
            or expected_skus and any(expected_skus[x["offer_id"]] != x["sku_id"] for x in quote["lines"])):
            raise Fault("invalid_quote", "The quote does not match the selected offers.", 502)
        return quote

    def _pools(self, goal, queries, fixed, visual, limit, excluded, allowed_skus):
        def fetch(category):
            unverified = []

            def reject(item):
                reason = candidate_rejection(item, goal, excluded=excluded.get(category), allowed_skus=allowed_skus)
                if reason and reason.startswith(("unverified_", "conflicting_")):
                    unverified.append(item)
                return reason

            fixed_item = fixed.get(category)
            if fixed_item:
                item = self.catalog.store.get("catalog", fixed_item["sku_id"])
                if not item or not item.get("active"):
                    return [], {"rejected": [{"id": fixed_item["sku_id"], "reason": "inactive_sku"}]}, []
                reason = reject(item)
                return ([] if reason else [item],
                        {"rejected": [{"id": item["id"], "reason": reason}] if reason else []}, unverified)
            alternate = queries.get(category, [])
            terms = category.replace("_", " ")
            if category == "laptop" and goal.get("light"):
                terms += " lightweight portable"
            if category == "dock" and goal.get("hz"):
                terms += f' USB-C 4K {goal["hz"]}Hz'
            extra = goal.get("queries", {}).get(category)
            docs, trace = self.catalog.search(extra or terms, None if category == "other" else category,
                                              alternates=alternate or ([terms] if extra else ()), limit=limit,
                                              pool=max(15, limit*2),
                                              visual=visual if len(goal["categories"]) == 1 else None, reject=reject)
            return docs, trace, unverified

        with ThreadPoolExecutor(max_workers=4) as executor:
            result = list(executor.map(fetch, goal["categories"]))
        return dict(zip(goal["categories"], result, strict=True))

    @staticmethod
    def _shortlist_offers(options, goal, fixed_offer=None):
        if fixed_offer:
            return [offer for offer in options if offer["id"] == fixed_offer]
        ranked = sorted(options, key=lambda x: (x["unit_price"]+x["shipping"], x["lead_hours"], x["id"]))
        if (goal.get("deadline") or ranking_preference(goal) == "delivery") and len(ranked) > 3:
            fastest = min(ranked, key=lambda x: (x["lead_hours"], x["unit_price"]+x["shipping"], x["id"]))
            return list({x["id"]: x for x in [fastest, *ranked]}.values())[:3]
        return ranked[:3]

    def find(self, account, goal, queries=None, fixed=None, visual=None, excluded=None,
             allowed_skus=None, allowed_sellers=None, diagnostics=None):
        queries, fixed, excluded = queries or {}, fixed or {}, excluded or {}
        diagnostics = {} if diagnostics is None else diagnostics
        diagnostics.update(sort_by=ranking_preference(goal), passes=[])
        for size in (10, 22):
            timings, rejected = {}, Counter()
            started = perf_counter()
            pools = self._pools(goal, queries, fixed, visual, size, excluded, allowed_skus)
            timings["pools"] = elapsed_ms(started)
            rejected.update(row["reason"] for _, trace, _ in pools.values() for row in trace.get("rejected", []))
            started = perf_counter()
            candidates = {doc["id"]: doc for docs, _, _ in pools.values() for doc in docs}
            offers = {}
            if candidates:
                offered = self.platform.call("commerce_offers", {"sku_ids": list(candidates)})
                for offer in offered:
                    key = offer["sku_id"]
                    if (key in candidates and offer["currency"] == "USD" and offer["stock"] >= 1
                        and (allowed_sellers is None or offer["seller"] in allowed_sellers)):
                        offers.setdefault(key, []).append(offer)
            options = []
            for cat, (items, _, _) in pools.items():
                slot = []
                for rank, item in enumerate(items):
                    available = self._shortlist_offers(offers.get(item["id"], []), goal,
                                                       fixed.get(cat, {}).get("offer_id"))
                    if not available:
                        rejected["no_eligible_offer"] += 1
                    slot.extend((rank, item, offer) for offer in available)
                options.append(slot)
            timings["offers"] = elapsed_ms(started)
            started = perf_counter()
            sample = self._pairs(options, goal, fixed, rejected)
            timings["combinations"] = elapsed_ms(started)
            started = perf_counter()
            plans = self._price(sample, account, goal, allowed_sellers, rejected)
            timings["quotes_and_final_rank"] = elapsed_ms(started)
            diagnostics["passes"].append({
                "slot_limit": size, "quotes_requested": len(sample), "returned_plans": len(plans),
                "rejections": dict(rejected), "timings_ms": timings,
            })
            if len(plans) >= 2 or size == 22:
                research = {doc["id"]: doc for docs, _, uncertain in pools.values() for doc in [*docs, *uncertain]}
                research_ids = list(dict.fromkeys([line["product_id"] for plan in plans for line in plan["items"]]))
                traces = {cat: trace for cat, (_, trace, _) in pools.items()}
                visible = [self._public(doc) for doc in list(research.values())[:16]]
                return plans, visible, traces, research_ids

    @staticmethod
    def _public(doc):
        return {k: doc.get(k) for k in ("id", "sku_id", "product_id", "title", "brand", "category", "variant",
                                       "source", "snapshot_price", "snapshot_currency", "rating", "images")}

    def _pairs(self, slots, goal, fixed, rejected):
        choice = ranking_preference(goal)

        def feasible():
            for pair in itertools.product(*slots):
                ids = [offer["id"] for _, _, offer in pair]
                if len(set(ids)) != len(ids):
                    continue
                valid, facts, reason = compatibility([item for _, item, _ in pair], goal)
                if not valid:
                    rejected[reason] += 1
                    continue
                amount = sum(offer["unit_price"] + offer["shipping"] for _, _, offer in pair)
                weight = bundle_weight(pair, fixed) if choice == "weight" else 0
                relevance = sum(rank for rank, _, _ in pair)
                lead = max(offer["lead_hours"] for _, _, offer in pair)
                primary = {"price": amount, "weight": weight, "relevance": relevance, "delivery": lead}[choice]
                estimate_penalty = int(bool(goal.get("budget") and amount > goal["budget"]))
                yield ((estimate_penalty, primary, amount, lead, relevance, ids), pair, facts, weight, relevance)

        return heapq.nsmallest(22, feasible(), key=lambda entry: entry[0])

    def _price(self, sample, account, goal, allowed_sellers, rejected):
        choice = ranking_preference(goal)

        def quote(entry):
            _, pair, facts, weight, relevance = entry
            lines = [{"offer_id": offer["id"], "quantity": 1} for _, _, offer in pair]
            try:
                price = self.quote(account, goal["address_id"], lines,
                                   expected_skus={o["id"]: i["id"] for _, i, o in pair})
            except Fault as error:
                if error.code == "out_of_stock":
                    return "out_of_stock", None
                raise
            if goal.get("budget") is not None and price["total"] > goal["budget"]:
                return "budget", None
            if goal.get("deadline") and price["eta"] > goal["deadline"]:
                return "deadline", None
            if allowed_sellers is not None and any(line["seller"] not in allowed_sellers for line in price["lines"]):
                return "seller", None
            identity = stable_id(*(x["offer_id"] for x in price["lines"]))
            primary = {"price": price["total"], "weight": weight, "delivery": price["eta"], "relevance": relevance}[choice]
            plan = {
                "id": identity, "quote": price, "evidence": facts,
                "items": [{**self._public(item), "specs": item["specs"], "offer_id": offer["id"]}
                          for _, item, offer in pair],
            }
            return None, ((primary, price["total"], price["eta"], relevance, identity), plan)

        with ThreadPoolExecutor(max_workers=5) as worker:
            results = list(worker.map(quote, sample))
        rejected.update(reason for reason, _ in results if reason)
        ranked = sorted((entry for _, entry in results if entry is not None), key=lambda entry: entry[0])
        chosen, seen = [], set()
        for _, plan in ranked:
            variants = tuple(sorted(item["sku_id"] for item in plan["items"]))
            if variants not in seen:
                chosen.append(plan)
                seen.add(variants)
        return chosen[:3]
