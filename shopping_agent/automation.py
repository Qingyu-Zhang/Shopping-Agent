from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .checkout import Checkout, audit, store_quote
from .core import Fault, Input, Line, new_id, normalize_lines, stable_id


class SearchScope(Input):
    task_id: str
    plan_version: int = Field(ge=1)
    allowed_skus: list[str] | None = Field(default=None, min_length=1, max_length=40)
    allowed_sellers: list[str] | None = Field(default=None, min_length=1, max_length=20)


class WatchRequest(Input):
    mode: Literal["notify", "cart", "buy"]
    trigger: Literal["price", "stock", "schedule"] = "price"
    address_id: str = "home"
    lines: list[Line] | None = Field(default=None, min_length=1, max_length=20)
    search_scope: SearchScope | None = None
    threshold: int | None = Field(default=None, ge=1, le=100_000_000)
    total_budget: int | None = Field(default=None, ge=1, le=100_000_000)
    max_runs: int = Field(default=1, ge=1, le=100)
    every_days: int | None = Field(default=None, ge=1, le=365)
    expires_days: int = Field(default=30, ge=1, le=365)
    payment_method: str | None = Field(default=None, max_length=160)

    @model_validator(mode="after")
    def validate_rule(self):
        if (self.lines is None) == (self.search_scope is None):
            raise ValueError("Choose exact offers or a shopping task search scope.")
        if self.search_scope and self.mode != "notify" and (
            not self.search_scope.allowed_skus or not self.search_scope.allowed_sellers
        ):
            raise ValueError("Explicitly approve SKUs and sellers for automated cart or purchase writes.")
        if self.trigger == "price" and not self.threshold:
            raise ValueError("Provide a price threshold in cents.")
        if self.trigger == "schedule" and not self.every_days:
            raise ValueError("Provide an interval for scheduled automation.")
        if self.max_runs > 1 and not self.every_days:
            raise ValueError("Provide an interval for repeat execution.")
        if self.mode == "buy" and not (self.threshold and self.total_budget and self.payment_method):
            raise ValueError("Provide a per-order maximum, total budget, and saved payment method.")
        return self


class Automations:
    def __init__(self, store, platform, planner, settings, clock):
        self.store, self.platform, self.planner, self.settings, self.clock = store, platform, planner, settings, clock

    def create(self, account, request):
        profile = self.platform.call("commerce_account", {"account_id": account})
        if request.address_id not in profile.get("addresses", {}):
            raise Fault("address_not_found", "The selected address is unavailable.", 404)
        if request.mode == "buy":
            if not self.settings.autobuy_enabled or not self.settings.checkout_enabled:
                raise Fault("autobuy_disabled", "New automatic purchases are paused.", 503)
            if request.payment_method not in profile.get("payment_methods", {}):
                raise Fault("payment_method", "Choose a saved payment method.", 422)
        rule = request.model_dump()
        if request.lines:
            rule["lines"] = normalize_lines(rule["lines"])
        else:
            scope = request.search_scope
            snapshot = self.store.get("task_views", stable_id(scope.task_id, str(scope.plan_version)), account)
            if not snapshot or not snapshot.get("goal", {}).get("categories"):
                raise Fault("not_found", "The selected shopping task version was not found.", 404)
            rule["search_goal"] = snapshot["goal"]
        now, key = self.clock(), new_id()
        first = now + (request.every_days*86400 if request.trigger == "schedule" else 0)
        watch = {**rule, "id": key, "account_id": account, "created_at": now, "expires_at": now+request.expires_days*86400,
                 "state": "active", "version": 1, "cycle": 0, "spent": 0, "next_check": first}
        with self.store.transaction(True) as tx:
            tx.put("automations", key, watch, owner=account, state="active", due=min(first, watch["expires_at"]))
            for line in rule["lines"] or []:
                tx.put("watch_offers", stable_id(key, line["offer_id"]), {
                    "id": key, "account_id": account,
                }, owner=line["offer_id"], once=True)
            audit(tx, account, "automation_authorized", key, now, mode=request.mode)
        return watch

    def cancel(self, account, key):
        with self.store.transaction(True) as tx:
            watch = tx.require("automations", key, account, lock=True)
            if watch["state"] == "active":
                watch.update(state="revoked", version=watch["version"]+1)
                watch.pop("lease", None)
                tx.put("automations", key, watch, owner=account, state="revoked")
                audit(tx, account, "automation_revoked", key, self.clock())
            return watch

    def wake(self, event_id, offer_id):
        now = self.clock()
        with self.store.transaction(True) as tx:
            if not tx.put("received_offer_events", event_id, {"offer_id": offer_id, "at": now}, once=True):
                return False
            for ref in tx.list("watch_offers", offer_id, limit=5000):
                watch = tx.get("automations", ref["id"], ref["account_id"], lock=True)
                if (watch and watch["state"] == "active" and watch["trigger"] != "schedule"
                    and not watch.get("lease") and watch.get("not_before", 0) <= now):
                    watch["next_check"] = now
                    tx.put("automations", watch["id"], watch, owner=watch["account_id"], state="active", due=now)
            return True

    def step(self, watch, lease):
        now, key, account = self.clock(), watch["id"], watch["account_id"]
        quote = None
        try:
            if now < watch["expires_at"] and (watch["mode"] != "buy" or (
                self.settings.autobuy_enabled and self.settings.checkout_enabled
            )):
                quote = self._quote(account, watch)
        except Fault:
            pass
        with self.store.transaction(True) as tx:
            current = tx.require("automations", key, account, lock=True)
            if current.get("lease") != lease or current["version"] != watch["version"] or current["state"] != "active":
                return
            current.pop("lease", None)
            now = self.clock()
            if current["expires_at"] <= now:
                current["state"] = "expired"
            elif quote and quote["expires_at"] > now and (
                current["threshold"] is None or quote["total"] <= current["threshold"]
            ) and (current["total_budget"] is None or current["spent"]+quote["total"] <= current["total_budget"]):
                self._accept(tx, current, quote, now)
            else:
                current["next_check"] = now + 300
            due = min(current["next_check"], current["expires_at"]) if current["state"] == "active" else None
            tx.put("automations", key, current, owner=account, state=current["state"], due=due)

    def _quote(self, account, watch):
        if watch["lines"]:
            return self.planner.quote(account, watch["address_id"], watch["lines"])
        goal = {**watch["search_goal"], "address_id": watch["address_id"]}
        limits = [x for x in (goal.get("budget"), watch["threshold"],
                              None if watch["total_budget"] is None else watch["total_budget"]-watch["spent"])
                  if x is not None]
        if limits:
            goal["budget"] = min(limits)
            if goal["budget"] <= 0:
                return None
        scope = watch["search_scope"]
        plans, *_ = self.planner.find(account, goal, allowed_skus=scope["allowed_skus"],
                                      allowed_sellers=scope["allowed_sellers"])
        return plans[0]["quote"] if plans else None

    def _accept(self, tx, watch, quote, now):
        account = watch["account_id"]
        run_id = stable_id(watch["id"], str(watch["cycle"]))
        execution = {"id": run_id, "watch_id": watch["id"], "account_id": account, "total": quote["total"],
                     "mode": watch["mode"], "lines": [
                         {"offer_id": line["offer_id"], "quantity": line["quantity"]} for line in quote["lines"]
                     ], "at": now, "state": "accepted", "tries": 0}
        watch["cycle"] += 1
        watch["spent"] += quote["total"]
        if watch["cycle"] >= watch["max_runs"]:
            watch["state"] = "finished"
        else:
            watch["next_check"] = watch["not_before"] = now + watch["every_days"]*86400
        if watch["mode"] == "notify":
            title = "Price target reached" if watch["trigger"] == "price" else (
                "Items are in stock" if watch["trigger"] == "stock" else "Scheduled shopping reminder"
            )
            tx.put("notices", run_id, {"id": run_id, "watch_id": watch["id"], "title": title,
                   "at": now, "total": quote["total"]}, owner=account, once=True)
            execution["state"] = "succeeded"
            tx.put("automation_runs", run_id, execution, owner=account, once=True)
        elif watch["mode"] == "cart":
            tx.put("automation_runs", run_id, execution, owner=account, state="pending", due=now, once=True)
        else:
            receipt = store_quote(tx, quote, now)
            receipt["payment_method"] = watch["payment_method"]
            tx.put("quotes", receipt["id"], receipt, owner=account)
            result = Checkout.enqueue(tx, account, receipt, now, automation=run_id)
            execution.update(state="processing", checkout_id=result["id"])
            tx.put("automation_runs", run_id, execution, owner=account, once=True)
        audit(tx, account, "automation_accepted", run_id, now, watch_id=watch["id"], mode=watch["mode"])

    def cart_step(self, row, lease):
        account = row["account_id"]
        final = False
        try:
            cart = self.platform.call("commerce_cart", {"account_id": account})
            response = self.platform.call("commerce_cart_write", {
                "account_id": account, "operation_id": row["id"], "expected_version": cart["version"],
                "lines": row["lines"], "mode": "add",
            })
            result = {"state": "succeeded", "cart_version": response["version"]}
            final = True
        except Fault as error:
            final = error.status < 500 and error.code != "cart_conflict"
            result = {"state": "failed" if final else "accepted", "code": error.code}
        with self.store.transaction(True) as tx:
            latest = tx.require("automation_runs", row["id"], account, lock=True)
            if latest.get("lease") != lease:
                return
            latest.pop("lease", None)
            latest.update(**result, tries=latest["tries"]+1)
            tx.put("automation_runs", row["id"], latest, owner=account, state="finished" if final else "pending",
                   due=None if final else self.clock()+min(300, 2**min(latest["tries"], 8)))
