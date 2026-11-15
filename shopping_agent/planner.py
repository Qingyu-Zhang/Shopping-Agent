from __future__ import annotations

import heapq
import itertools
from concurrent.futures import ThreadPoolExecutor

from .core import Fault, stable_id
from .platform import Quote


def compatibility(items, goal):
    hz = goal.get("hz")
    ports = goal.get("ports") or ["hdmi", "dp"]
    dock = next((x for x in items if x["category"] == "dock"), None)
    laptop = next((x for x in items if x["category"] == "laptop"), None)
    screen = next((x for x in items if x["category"] == "monitor"), None)
    if not dock or (not hz and not laptop):
        return True, []
    host = laptop["specs"].get("video", {}) if laptop else goal.get("host_video")
    if host is None:
        return False, []
    if screen:
        ports = [port for port in ports if port in screen["specs"].get("monitor_ports", [])]
    for protocol, mode in dock["specs"].get("inputs", {}).items():
        if host.get(protocol, -1) < mode["min"]:
            continue
        for port in ports:
            output = mode["outputs"].get(port)
            if output is not None and output >= (hz or 1):
                facts = [{
                    "id": stable_id(item["id"], protocol, port), "sku_id": item["id"],
                    "source": item["source"], "source_version": item["source_version"],
                    "text": text,
                } for item, text in [
                    *(([(laptop, f'Host supports {protocol} at level {host[protocol]}.')] if laptop else [])),
                    (dock, f'{protocol} input delivers 4K at {output} Hz over {port.upper()}.'),
                ]]
                if screen:
                    facts.append({"id": stable_id(screen["id"], protocol, port), "sku_id": screen["id"],
                                  "source": screen["source"], "source_version": screen["source_version"],
                                  "text": f'Monitor accepts {port.upper()}.'})
                return True, facts
    return False, []


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

    def _pools(self, goal, queries, fixed, visual, limit):
        def fetch(category):
            fixed_item = fixed.get(category)
            if fixed_item:
                item = self.catalog.store.get("catalog", fixed_item["sku_id"])
                return ([item] if item and item.get("active") else []), {}
            alternate = queries.get(category, [])
            terms = category.replace("_", " ")
            if category == "laptop" and goal.get("light"):
                terms += " lightweight portable"
            if category == "dock" and goal.get("hz"):
                terms += f' USB-C 4K {goal["hz"]}Hz'
            extra = goal.get("queries", {}).get(category)
            return self.catalog.search(extra or terms, None if category == "other" else category,
                                       alternates=alternate or ([terms] if extra else ()), limit=limit,
                                       pool=max(15, limit*2), visual=visual if len(goal["categories"]) == 1 else None)
        with ThreadPoolExecutor(max_workers=4) as executor:
            result = list(executor.map(fetch, goal["categories"]))
        return dict(zip(goal["categories"], result, strict=True))

    @staticmethod
    def _eligible(item, goal):
        specs = item.get("specs", {})
        if not item.get("sku_id"):
            return False
        if item["category"] == "laptop" and goal.get("min_ram"):
            if specs.get("ram_gb", 0) < goal["min_ram"]:
                return False
        weights = goal.get("max_weights", {})
        ceiling = weights.get(item["category"])
        return ceiling is None or (specs.get("weight_g") is not None and specs["weight_g"] <= ceiling)

    def find(self, account, goal, queries=None, fixed=None, visual=None, excluded=None,
             allowed_skus=None, allowed_sellers=None):
        queries, fixed = queries or {}, fixed or {}
        excluded = excluded or {}
        for size in (10, 22):
            pools = self._pools(goal, queries, fixed, visual, size)
            eligible = {cat: [x for x in docs if self._eligible(x, goal) and x["sku_id"] != excluded.get(cat)
                              and (allowed_skus is None or x["sku_id"] in allowed_skus)]
                        for cat, (docs, _) in pools.items()}
            candidates = {doc["id"]: doc for group in eligible.values() for doc in group}
            offers = {}
            if candidates:
                offered = self.platform.call("commerce_offers", {"sku_ids": list(candidates)})
                for offer in offered:
                    key = offer.get("sku_id")
                    if (key in candidates and offer.get("currency") == "USD" and offer.get("stock", 0) >= 1
                        and (allowed_sellers is None or offer.get("seller") in allowed_sellers)):
                        offers.setdefault(key, []).append(offer)
            options = []
            for cat, items in eligible.items():
                options.append([
                    (rank, item, offer) for rank, item in enumerate(items)
                    for offer in sorted(offers.get(item["id"], []),
                                        key=lambda x: (x["unit_price"]+x["shipping"], x["lead_hours"], x["id"]))[:3]
                    if cat not in fixed or fixed[cat]["offer_id"] == offer["id"]
                ])
            sample = self._pairs(options, goal, fixed)
            plans = self._price(sample, account, goal, allowed_sellers)
            if len(plans) >= 2 or size == 22:
                research = {doc["id"]: doc for docs, _ in pools.values() for doc in docs}
                research_ids = list(dict.fromkeys([line["product_id"] for plan in plans for line in plan["items"]]))
                visible = [self._public(doc) for doc in list(research.values())[:16]]
                traces = {cat: trace for cat, (_, trace) in pools.items()}
                return plans, visible, traces, research_ids

    @staticmethod
    def _public(doc):
        return {k: doc.get(k) for k in ("id", "sku_id", "product_id", "title", "brand", "category", "variant",
                                       "source", "snapshot_price", "snapshot_currency", "rating", "images")}

    def _pairs(self, slots, goal, fixed):
        def feasible():
            for pair in itertools.product(*slots):
                ids = [offer["id"] for _, _, offer in pair]
                if len(set(ids)) != len(ids):
                    continue
                valid, facts = compatibility([item for _, item, _ in pair], goal)
                if valid:
                    amount = sum(offer["unit_price"] + offer["shipping"] for _, _, offer in pair)
                    weight = sum(item["specs"].get("weight_g", 2500) for _, item, _ in pair
                                 if item["category"] not in fixed)
                    estimate_penalty = int(bool(goal.get("budget") and amount > goal["budget"]))
                    relevance = sum(rank for rank, _, _ in pair)
                    yield ((estimate_penalty, weight if goal.get("light") else relevance, amount, ids),
                           pair, facts)
        return heapq.nsmallest(22, feasible(), key=lambda entry: entry[0])

    def _price(self, sample, account, goal, allowed_sellers):
        def quote(entry):
            _, pair, facts = entry
            lines = [{"offer_id": offer["id"], "quantity": 1} for _, _, offer in pair]
            try:
                price = self.quote(account, goal["address_id"], lines,
                                   expected_skus={o["id"]: i["id"] for _, i, o in pair})
            except Fault as error:
                if error.code == "out_of_stock":
                    return None
                raise
            if (goal.get("budget") is not None and price["total"] > goal["budget"]
                or goal.get("deadline") and price["eta"] > goal["deadline"]
                or allowed_sellers is not None and any(line["seller"] not in allowed_sellers for line in price["lines"])):
                return None
            identity = stable_id(*(x["offer_id"] for x in price["lines"]))
            return {
                "id": identity, "quote": price, "evidence": facts,
                "items": [{**self._public(item), "specs": item["specs"], "offer_id": offer["id"]}
                          for _, item, offer in pair],
            }

        with ThreadPoolExecutor(max_workers=5) as worker:
            results = list(worker.map(quote, sample))
        ranked = [plan for plan in results if plan]
        chosen, seen = [], set()
        for plan in ranked:
            variants = tuple(sorted(item["sku_id"] for item in plan["items"]))
            if variants not in seen:
                chosen.append(plan)
                seen.add(variants)
        return chosen[:3]
