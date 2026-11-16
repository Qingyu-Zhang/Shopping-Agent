from __future__ import annotations

from .core import Fault, new_id, normalize_lines, stable_id
from .platform import Quote, quote_fingerprint


def store_quote(tx, quote, now, *, constraints=None, selection_trace=None):
    receipt = {
        "id": new_id(), "quote": quote, "terms": quote_fingerprint(quote),
        "created_at": now, "constraints": constraints or {},
    }
    if selection_trace:
        receipt["selection_trace"] = selection_trace
    tx.put("quotes", receipt["id"], receipt, owner=quote["account_id"])
    return receipt


def audit(tx, account, event, reference, now, *, once=False, **details):
    key = stable_id(account, event, reference) if once else new_id()
    tx.put("audit", key, {"id": key, "event": event, "reference": reference, "at": now, **details},
           owner=account, once=once)


class Checkout:
    def __init__(self, store, platform, planner, settings, clock):
        self.store, self.platform, self.planner = store, platform, planner
        self.settings, self.clock = settings, clock

    def prepare(self, account, address, lines, *, constraints=None, expected_skus=None, selection_trace=None):
        quote = self.planner.quote(account, address, normalize_lines(lines), expected_skus=expected_skus)
        now = self.clock()
        if quote["expires_at"] <= now:
            raise Fault("quote_expired", "The platform returned an expired quote.")
        limits = constraints or {}
        if limits.get("budget") and quote["total"] > limits["budget"]:
            raise Fault("budget_exceeded", "The updated quote exceeds the shopping budget.")
        if limits.get("deadline") and quote["eta"] > limits["deadline"]:
            raise Fault("delivery_changed", "The updated delivery estimate misses the deadline.")
        with self.store.transaction(True) as tx:
            receipt = store_quote(tx, quote, now, constraints=constraints, selection_trace=selection_trace)
            audit(tx, account, "quote_prepared", receipt["id"], now, platform_quote_id=quote["id"],
                  **(selection_trace or {}))
            return receipt

    def confirm(self, account, receipt_id, terms):
        with self.store.transaction(True) as tx:
            receipt = tx.require("quotes", receipt_id, account, lock=True)
            if receipt["terms"] != terms:
                raise Fault("quote_mismatch", "Confirm the exact terms shown to you.")
            key = stable_id(account, receipt_id)
            previous = tx.get("checkouts", key, account)
            if previous:
                return previous
            if not self.settings.checkout_enabled:
                raise Fault("checkout_disabled", "New checkouts are currently paused.", 503)
            now = self.clock()
            if receipt["quote"]["expires_at"] <= now:
                raise Fault("quote_expired", "Request an updated quote and confirm it.")
            limits = receipt["constraints"]
            if limits.get("budget") and receipt["quote"]["total"] > limits["budget"]:
                raise Fault("budget_exceeded", "The updated quote exceeds the task budget.")
            if limits.get("deadline") and receipt["quote"]["eta"] > limits["deadline"]:
                raise Fault("delivery_changed", "The updated quote misses the task deadline.")
            return self.enqueue(tx, account, receipt, now)

    @staticmethod
    def enqueue(tx, account, receipt, now, automation=None):
        key = stable_id(account, receipt["id"])
        row = {"id": key, "account_id": account, "receipt_id": receipt["id"], "state": "processing",
               "created_at": now, "updated_at": now, "tries": 0, "automation_run": automation}
        tx.put("checkouts", key, row, owner=account, state="pending", due=now, once=True)
        audit(tx, account, "checkout_accepted", key, now, receipt_id=receipt["id"], automation_run=automation,
              **receipt.get("selection_trace", {}))
        return row

    def step(self, row, lease):
        now = self.clock()
        account = row["account_id"]
        receipt = self.store.get("quotes", row["receipt_id"], account)
        quote = receipt["quote"]
        result = None
        try:
            result = self.platform.call("commerce_status", {"account_id": account, "attempt_id": row["id"]})
            if result is None:
                result = self.platform.call("commerce_submit", {
                    "account_id": account, "attempt_id": row["id"], "quote_id": quote["id"], "terms": receipt["terms"],
                    "payment_method": receipt.get("payment_method"),
                })
            if not isinstance(result, dict):
                raise Fault("platform_protocol", "The checkout status could not be verified.", 503)
            if result.get("state") == "succeeded":
                if not result.get("order_id") or type(result.get("total")) is not int or result["total"] != quote["total"]:
                    raise Fault("platform_protocol", "The checkout result does not match the confirmed quote.", 503)
            elif result.get("state") == "requote":
                replacement = Quote.model_validate(result.get("quote")).model_dump()
                same_lines = {(x["offer_id"], x["quantity"], x["sku_id"]) for x in quote["lines"]}
                other_lines = {(x["offer_id"], x["quantity"], x["sku_id"]) for x in replacement["lines"]}
                if (replacement["account_id"] != account or replacement["address_id"] != quote["address_id"]
                    or same_lines != other_lines or replacement["id"] == quote["id"]
                    or replacement["expires_at"] <= now):
                    raise Fault("platform_protocol", "The replacement quote could not be verified.", 503)
            elif result.get("state") not in {"failed", "processing"}:
                raise Fault("platform_protocol", "The checkout state could not be verified.", 503)
        except Fault as error:
            if error.status < 500:
                result = {"state": "failed", "code": error.code}
            else:
                result = None
        except (ValueError, TypeError, KeyError):
            result = None
        with self.store.transaction(True) as tx:
            current = tx.require("checkouts", row["id"], account, lock=True)
            if current.get("lease") != lease:
                return
            current.pop("lease", None)
            current.update(tries=current["tries"]+1, updated_at=self.clock())
            if not result or result["state"] == "processing":
                retry_at = self.clock() + min(300, 2 ** min(current["tries"], 8))
                tx.put("checkouts", row["id"], current, owner=account, state="pending", due=retry_at)
                return
            state = result["state"]
            current.update(state=state, code=result.get("code"), order_id=result.get("order_id"))
            if state == "requote":
                new = store_quote(tx, replacement, now, constraints=receipt.get("constraints"),
                                  selection_trace=receipt.get("selection_trace"))
                current["replacement"] = new
            tx.put("checkouts", row["id"], current, owner=account, state="finished")
            if run_id := current.get("automation_run"):
                execution = tx.require("automation_runs", run_id, account, lock=True)
                execution.update(state=state, order_id=current.get("order_id"))
                tx.put("automation_runs", run_id, execution, owner=account)
            audit(tx, account, "checkout_" + state, row["id"], now, code=current.get("code"),
                  receipt_id=receipt["id"], order_id=current.get("order_id"),
                  **receipt.get("selection_trace", {}))

    def preview_return(self, account, order_id, offer_id, quantity):
        args = {"account_id": account, "order_id": order_id, "offer_id": offer_id, "quantity": quantity}
        preview = self.platform.call("commerce_return_preview", args)
        if (not isinstance(preview, dict) or any(preview.get(k) != args[k] for k in ("order_id", "offer_id", "quantity"))
            or type(preview.get("refund")) is not int or preview["refund"] < 0 or preview.get("currency") != "USD"):
            raise Fault("platform_protocol", "The return estimate could not be verified.", 502)
        receipt = {"id": new_id(), "preview": preview, "expires_at": self.clock()+300}
        receipt["terms"] = quote_fingerprint(receipt)
        with self.store.transaction(True) as tx:
            tx.put("return_quotes", receipt["id"], receipt, owner=account)
        return receipt

    def confirm_return(self, account, receipt_id, terms, reason):
        if not self.settings.checkout_enabled:
            raise Fault("checkout_disabled", "New commerce writes are currently paused.", 503)
        with self.store.transaction(True) as tx:
            receipt = tx.require("return_quotes", receipt_id, account, lock=True)
            if receipt["terms"] != terms:
                raise Fault("quote_mismatch", "Confirm the exact return terms.")
            prior = tx.get("return_results", receipt_id, account)
            if prior:
                return prior
            if not receipt.get("started") and receipt["expires_at"] <= self.clock():
                raise Fault("quote_expired", "The return preview expired.")
            if receipt.get("reason", reason) != reason:
                raise Fault("operation_conflict", "The return already has a different reason.")
            receipt.update(started=True, reason=reason)
            tx.put("return_quotes", receipt_id, receipt, owner=account)
        args = {k: v for k, v in receipt["preview"].items() if k != "currency"}
        value = self.platform.call("commerce_return_request", {
            **args, "account_id": account, "reason": reason, "operation_id": stable_id(account, receipt_id),
        })
        with self.store.transaction(True) as tx:
            if tx.put("return_results", receipt_id, value, owner=account, once=True):
                audit(tx, account, "return_requested", receipt_id, self.clock())
        return value
