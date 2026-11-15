from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager, suppress
from typing import Any, Literal

import httpx
from pydantic import Field, ValidationError, model_validator

from .core import Fault, Input, Line, Settings, new_id, normalize_lines, stable_id, timestamp
from .seed import seed_platform
from .storage import Store


class AccountRequest(Input):
    account_id: str = Field(min_length=1, max_length=180)


class OfferRequest(Input):
    sku_ids: list[str] = Field(min_length=1, max_length=80)


class HistoryRequest(Input):
    offer_id: str


class QuoteRequest(AccountRequest):
    address_id: str
    lines: list[Line] = Field(min_length=1, max_length=20)


class CartRequest(AccountRequest):
    operation_id: str
    expected_version: int = Field(ge=0)
    lines: list[Line] = Field(min_length=1, max_length=20)
    mode: Literal["add", "remove"]


class AttemptRequest(AccountRequest):
    attempt_id: str


class SubmitRequest(AttemptRequest):
    quote_id: str
    terms: str
    payment_method: str | None = None


class OrderRequest(AccountRequest):
    order_id: str


class ReturnPreview(OrderRequest):
    offer_id: str
    quantity: int = Field(ge=1, le=20)


class ReturnRequest(ReturnPreview):
    operation_id: str
    refund: int = Field(ge=0)
    reason: str = Field(min_length=2, max_length=300)


class QuoteLine(Line):
    sku_id: str
    title: str
    variant: str
    seller: str
    unit_price: int = Field(ge=0)
    shipping: int = Field(ge=0)
    lead_hours: int = Field(ge=0)
    eta: float


class Quote(AccountRequest):
    id: str
    address_id: str
    lines: list[QuoteLine] = Field(min_length=1, max_length=20)
    subtotal: int = Field(ge=0)
    shipping: int = Field(ge=0)
    tax: int = Field(ge=0)
    discount: int = Field(ge=0)
    total: int = Field(ge=0)
    currency: Literal["USD"]
    expires_at: float
    eta: float

    @model_validator(mode="after")
    def check_totals(self):
        if (
            self.subtotal != sum(x.unit_price * x.quantity for x in self.lines)
            or self.shipping != sum(x.shipping for x in self.lines)
            or self.total != self.subtotal + self.shipping + self.tax - self.discount
            or self.eta != max(x.eta for x in self.lines)
        ):
            raise ValueError("Quote totals or delivery are inconsistent.")
        return self


class ToolEnvelope(Input):
    ok: bool
    value: Any = None
    error: dict | None = None


TOOL_INPUTS = {
    "commerce_account": AccountRequest,
    "commerce_offers": OfferRequest,
    "commerce_history": HistoryRequest,
    "commerce_quote": QuoteRequest,
    "commerce_cart": AccountRequest,
    "commerce_cart_write": CartRequest,
    "commerce_submit": SubmitRequest,
    "commerce_status": AttemptRequest,
    "commerce_orders": AccountRequest,
    "commerce_order": OrderRequest,
    "commerce_return_preview": ReturnPreview,
    "commerce_return_request": ReturnRequest,
}


def quote_fingerprint(quote: dict) -> str:
    import json
    return stable_id(json.dumps(quote, sort_keys=True, ensure_ascii=True, separators=(",", ":")))


class MockPlatform:
    def __init__(self, store: Store, clock=timestamp):
        self.store, self.clock = store, clock

    def call(self, name: str, arguments: dict):
        if name not in TOOL_INPUTS:
            raise Fault("unknown_tool", "This platform tool does not exist.", 404)
        return getattr(self, name.removeprefix("commerce_"))(TOOL_INPUTS[name].model_validate(arguments))

    def account(self, req):
        with self.store.transaction() as tx:
            return tx.require("accounts", req.account_id)

    def offers(self, req):
        with self.store.transaction() as tx:
            return [offer for sku in dict.fromkeys(req.sku_ids) for offer in tx.list("offers", sku, limit=20)]

    def history(self, req):
        with self.store.transaction() as tx:
            tx.require("offers", req.offer_id)
            return sorted(tx.list("price_history", req.offer_id, limit=5000), key=lambda x: x["at"])

    def _quote(self, tx, account, address, lines):
        profile = tx.require("accounts", account)
        if address not in profile["addresses"]:
            raise Fault("address_not_found", "The selected address is unavailable.", 404)
        now = self.clock()
        priced = []
        for line in normalize_lines(lines):
            offer = tx.require("offers", line["offer_id"])
            if offer["stock"] < line["quantity"]:
                raise Fault("out_of_stock", f'{offer["title"]} has insufficient stock.')
            extra_hours = 0 if address == "home" else 24
            lead_hours = offer["lead_hours"] + extra_hours
            priced.append({
                **line, **{k: offer[k] for k in ("sku_id", "title", "variant", "seller", "unit_price", "shipping")},
                "lead_hours": lead_hours, "eta": now + lead_hours * 3600,
            })
        subtotal = sum(x["unit_price"] * x["quantity"] for x in priced)
        shipping = sum(x["shipping"] for x in priced)
        quote = Quote(
            id=new_id(), account_id=account, address_id=address, lines=priced, subtotal=subtotal,
            shipping=shipping, tax=0, discount=0, total=subtotal+shipping, currency="USD",
            expires_at=now+300, eta=max(x["eta"] for x in priced),
        ).model_dump()
        tx.put("platform_quotes", quote["id"], quote, owner=account)
        return quote

    def quote(self, req):
        with self.store.transaction(True) as tx:
            return self._quote(tx, req.account_id, req.address_id, [x.model_dump() for x in req.lines])

    def cart(self, req):
        self.account(req)
        return self.store.get("carts", req.account_id) or {"version": 0, "lines": []}

    def cart_write(self, req):
        account = req.account_id
        identity = stable_id(account, req.operation_id)
        expected = req.model_dump()
        expected["lines"] = normalize_lines(expected["lines"])
        expected.pop("expected_version")
        with self.store.transaction(True) as tx:
            tx.require("accounts", account, lock=True)
            previous = tx.get("cart_ops", identity)
            if previous:
                if previous["request"] != expected:
                    raise Fault("operation_conflict", "This operation ID was used for a different cart change.")
                return previous["result"]
            cart = tx.get("carts", account) or {"version": 0, "lines": []}
            if req.expected_version != cart["version"]:
                raise Fault("cart_conflict", "The cart changed. Refresh it and retry.")
            amounts = {row["offer_id"]: row["quantity"] for row in cart["lines"]}
            for line in expected["lines"]:
                key, quantity = line["offer_id"], line["quantity"]
                if req.mode == "add":
                    offer = tx.require("offers", key)
                    count = amounts.get(key, 0) + quantity
                    if count > min(20, offer["stock"]):
                        raise Fault("out_of_stock", "The requested cart quantity is unavailable.")
                    amounts[key] = count
                else:
                    count = amounts.get(key, 0) - quantity
                    if count > 0:
                        amounts[key] = count
                    else:
                        amounts.pop(key, None)
            if len(amounts) > 20:
                raise Fault("cart_limit", "A cart can hold up to 20 distinct offers.")
            result = {"version": cart["version"]+1, "lines": [
                {"offer_id": key, "quantity": count} for key, count in sorted(amounts.items())
            ]}
            tx.put("carts", account, result, owner=account)
            tx.put("cart_ops", identity, {"request": expected, "result": result}, owner=account)
            return result

    def status(self, req):
        row = self.store.get("platform_attempts", req.attempt_id, req.account_id)
        return row and row["result"]

    def submit(self, req):
        account = req.account_id
        now = self.clock()
        lost = False
        with self.store.transaction(True) as tx:
            profile = tx.require("accounts", account, lock=True)
            existing = tx.get("platform_attempts", req.attempt_id)
            if existing:
                if existing["request"] != req.model_dump():
                    raise Fault("operation_conflict", "The checkout attempt was already used.")
                return existing["result"]
            quote = tx.require("platform_quotes", req.quote_id, account)
            if quote_fingerprint(quote) != req.terms:
                raise Fault("quote_mismatch", "The confirmed terms differ from the platform quote.")
            if req.payment_method and req.payment_method not in profile.get("payment_methods", {}):
                raise Fault("payment_method", "The saved payment method is no longer available.")
            scope = stable_id(account, req.quote_id)
            already = tx.get("platform_quote_use", scope)
            if already:
                raise Fault("quote_consumed", "The platform quote has already been submitted.")
            offers = {line["offer_id"]: tx.require("offers", line["offer_id"], lock=True)
                      for line in sorted(quote["lines"], key=lambda x: x["offer_id"])}
            extra_hours = 0 if quote["address_id"] == "home" else 24
            if any(offers[x["offer_id"]]["stock"] < x["quantity"] for x in quote["lines"]):
                result = {"state": "failed", "code": "out_of_stock"}
            elif quote["expires_at"] <= now or any(
                offer["unit_price"] != line["unit_price"] or offer["shipping"] != line["shipping"]
                or offer["lead_hours"] + extra_hours != line["lead_hours"]
                for line in quote["lines"] for offer in [offers[line["offer_id"]]]
            ):
                lines = [{k: line[k] for k in ("offer_id", "quantity")} for line in quote["lines"]]
                result = {"state": "requote", "quote": self._quote(tx, account, quote["address_id"], lines)}
            else:
                control = tx.get("controls", "payment", lock=True) or {}
                fault = control.get("mode")
                if fault:
                    tx.put("controls", "payment", {"mode": None})
                if fault == "decline":
                    result = {"state": "failed", "code": "payment_declined"}
                else:
                    for line in quote["lines"]:
                        offer = offers[line["offer_id"]]
                        offer["stock"] -= line["quantity"]
                        tx.put("offers", offer["id"], offer, owner=offer["sku_id"])
                        self._write_offer_event(tx, offer)
                    result = self._commit_order(tx, req, quote, now)
                    lost = fault == "lost_response"
            tx.put("platform_quote_use", scope, {"attempt_id": req.attempt_id}, owner=account)
            tx.put("platform_attempts", req.attempt_id, {"request": req.model_dump(), "result": result}, owner=account)
        if lost:
            raise Fault("unknown_payment", "Payment response was interrupted.", 503)
        return result

    def _commit_order(self, tx, req, quote, now):
        account, order_id = req.account_id, new_id()
        seller_lines = defaultdict(list)
        for line in quote["lines"]:
            seller_lines[line["seller"]].append(line)
        parcels = [{
            "id": new_id(), "seller": seller, "status": "processing", "eta": max(x["eta"] for x in lines),
            "offer_ids": [x["offer_id"] for x in lines],
        } for seller, lines in seller_lines.items()]
        order = {
            "id": order_id, "account_id": account, "attempt_id": req.attempt_id, "created_at": now,
            "total": quote["total"], "currency": "USD", "lines": quote["lines"], "parcels": parcels,
        }
        tx.put("orders", order_id, order, owner=account)
        tx.put("payments", req.attempt_id, {"order_id": order_id, "total": order["total"]}, owner=account)
        return {"state": "succeeded", "order_id": order_id, "total": order["total"]}

    def orders(self, req):
        self.account(req)
        return sorted(self.store.list("orders", req.account_id, limit=100), key=lambda x: x["created_at"], reverse=True)

    def order(self, req):
        with self.store.transaction() as tx:
            return tx.require("orders", req.order_id, req.account_id)

    def _return_preview(self, tx, req, order):
        line = next((x for x in order["lines"] if x["offer_id"] == req.offer_id), None)
        delivered = any(req.offer_id in x["offer_ids"] and x["status"] == "delivered" for x in order["parcels"])
        if not line or not delivered or self.clock() > order["created_at"] + 30*86400:
            raise Fault("return_ineligible", "This item is not currently eligible for return.")
        reserved = sum(x["quantity"] for x in tx.list("returns", order["id"], limit=500)
                       if x["offer_id"] == req.offer_id)
        if req.quantity + reserved > line["quantity"]:
            raise Fault("return_quantity", "The quantity exceeds the available return quantity.")
        return {"offer_id": req.offer_id, "quantity": req.quantity, "refund": line["unit_price"]*req.quantity,
                "currency": "USD", "order_id": order["id"]}

    def return_preview(self, req):
        with self.store.transaction() as tx:
            order = tx.require("orders", req.order_id, req.account_id)
            return self._return_preview(tx, req, order)

    def return_request(self, req):
        with self.store.transaction(True) as tx:
            order = tx.require("orders", req.order_id, req.account_id, lock=True)
            identity = stable_id(req.account_id, req.operation_id)
            previous = tx.get("return_ops", identity)
            if previous:
                if previous["request"] != req.model_dump():
                    raise Fault("operation_conflict", "This return operation ID was already used.")
                return previous["result"]
            preview = self._return_preview(tx, req, order)
            if preview["refund"] != req.refund:
                raise Fault("refund_changed", "Get a new return preview and confirm it.")
            result = {**preview, "id": new_id(), "status": "requested", "reason": req.reason}
            tx.put("returns", result["id"], result, owner=order["id"])
            tx.put("return_ops", identity, {"request": req.model_dump(), "result": result}, owner=req.account_id)
            return result

    def _write_offer_event(self, tx, offer):
        key = new_id()
        at = self.clock()
        tx.put("price_history", key, {"id": key, **offer, "at": at}, owner=offer["id"])
        tx.put("offer_events", key, {"id": key, "offer_id": offer["id"], "at": at},
               owner=offer["id"], due=at, state="pending")

    def change_offer(self, offer_id, changes):
        with self.store.transaction(True) as tx:
            offer = tx.require("offers", offer_id, lock=True)
            offer.update(changes, updated_at=self.clock())
            tx.put("offers", offer_id, offer, owner=offer["sku_id"])
            self._write_offer_event(tx, offer)
            return offer

    def payment_fault(self, mode):
        with self.store.transaction(True) as tx:
            tx.put("controls", "payment", {"mode": mode})

    def update_parcel(self, account_id, order_id, parcel_id, status):
        with self.store.transaction(True) as tx:
            order = tx.require("orders", order_id, account_id, lock=True)
            parcel = next((x for x in order["parcels"] if x["id"] == parcel_id), None)
            if not parcel:
                raise Fault("not_found", "Parcel not found.", 404)
            transitions = ["processing", "shipped", "delivered"]
            if transitions.index(status) < transitions.index(parcel["status"]):
                raise Fault("parcel_state", "The parcel cannot move backwards.")
            parcel["status"] = status
            tx.put("orders", order_id, order, owner=account_id)
            return parcel


class RemotePlatform:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.headers = {"Authorization": f"Bearer {settings.platform_token}"}
        self.http = httpx.Client(timeout=8, headers=self.headers)
        if settings.platform_transport not in {"http", "mcp"}:
            raise ValueError("PLATFORM_TRANSPORT must be http or mcp.")

    async def _mcp(self, name, args):
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        async with httpx.AsyncClient(timeout=8, headers=self.headers) as client:
            async with streamable_http_client(self.settings.platform_url, http_client=client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, {"payload": args})
                    if result.isError:
                        raise Fault("platform_protocol", "The MCP tool returned a protocol error.", 502)
                    return result.structuredContent

    def call(self, name, arguments):
        args = TOOL_INPUTS[name].model_validate(arguments).model_dump()
        try:
            if self.settings.platform_transport == "mcp":
                envelope = asyncio.run(asyncio.wait_for(self._mcp(name, args), timeout=12))
            else:
                response = self.http.post(self.settings.platform_url.rstrip("/") + "/tools/" + name, json=args)
                response.raise_for_status()
                envelope = response.json()
            if isinstance(envelope, dict) and envelope.get("ok") is True and "value" in envelope:
                return envelope["value"]
            if isinstance(envelope, dict) and envelope.get("ok") is False:
                failure = envelope["error"]
                raise Fault(failure["code"], failure["message"], failure["status"])
            raise ValueError("Invalid platform response")
        except Fault:
            raise
        except Exception as error:
            raise Fault("platform_unavailable", "The commerce platform response could not be verified.", 503) from error


def boot_platform(settings: Settings, clock=timestamp):
    store = Store(settings.platform_database)
    store.initialize()
    if settings.demo:
        seed_platform(store)
    return MockPlatform(store, clock)


def create_platform_app(settings: Settings | None = None):
    from secrets import compare_digest
    from fastapi import Body, FastAPI, Request
    from fastapi.responses import JSONResponse
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings

    settings = settings or Settings.from_env()
    platform = boot_platform(settings)
    mcp = FastMCP(
        "Shopping commerce sandbox", stateless_http=True, json_response=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["127.0.0.1:*", "localhost:*", "platform:*"],
            allowed_origins=["http://127.0.0.1:*", "http://localhost:*"],
        ),
    )
    mcp_app = mcp.streamable_http_app()

    def envelope(name, payload):
        try:
            return {"ok": True, "value": platform.call(name, payload)}
        except Fault as error:
            return {"ok": False, "error": {"code": error.code, "message": str(error), "status": error.status}}
        except ValidationError:
            return {"ok": False, "error": {"code": "invalid_input", "message": "Invalid tool input.", "status": 422}}

    def handler(name):
        async def call(payload: Input) -> ToolEnvelope:
            result = await asyncio.to_thread(envelope, name, payload.model_dump())
            return ToolEnvelope.model_validate(result)
        call.__annotations__["payload"] = TOOL_INPUTS[name]
        return call

    for name in TOOL_INPUTS:
        mcp.add_tool(handler(name), name=name, description=f"Platform operation {name}.", structured_output=True)

    async def publish_events():
        headers = {"Authorization": f"Bearer {settings.platform_token}"}
        async with httpx.AsyncClient(timeout=5, headers=headers) as client:
            while True:
                try:
                    rows = await asyncio.to_thread(platform.store.list, "offer_events", state="pending", limit=100)
                    for row in rows:
                        response = await client.post(settings.offer_event_url, json={
                            "event_id": row["id"], "offer_id": row["offer_id"],
                        })
                        response.raise_for_status()
                        await asyncio.to_thread(acknowledge_event, row["id"])
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(2)

    def acknowledge_event(key):
        with platform.store.transaction(True) as tx:
            tx.delete("offer_events", key)

    @asynccontextmanager
    async def lifespan(_):
        publisher = asyncio.create_task(publish_events()) if settings.offer_event_url else None
        try:
            async with mcp.session_manager.run():
                yield
        finally:
            if publisher:
                publisher.cancel()
                with suppress(asyncio.CancelledError):
                    await publisher
            platform.store.engine.dispose()

    app = FastAPI(title="Shopping platform sandbox", lifespan=lifespan)
    app.state.platform = platform

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        expected = f"Bearer {settings.platform_token}"
        if request.url.path != "/health" and not compare_digest(request.headers.get("authorization", ""), expected):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return await call_next(request)

    @app.get("/health")
    def health():
        return {"status": "ready"}

    @app.post("/tools/{name}")
    def invoke(name: str, payload: dict = Body()):
        return envelope(name, payload)

    if settings.demo:
        @app.patch("/demo/offers/{offer_id}")
        def offer_change(offer_id: str, payload: dict = Body()):
            class OfferChange(Input):
                unit_price: int | None = Field(default=None, ge=0)
                stock: int | None = Field(default=None, ge=0)
                lead_hours: int | None = Field(default=None, ge=0)
            return platform.change_offer(offer_id, OfferChange.model_validate(payload).model_dump(exclude_none=True))

        @app.put("/demo/payment")
        def payment(mode: Literal["lost_response", "decline"]):
            platform.payment_fault(mode)
            return {"mode": mode}

        @app.patch("/demo/accounts/{account_id}/orders/{order_id}/parcels/{parcel_id}")
        def parcel(account_id: str, order_id: str, parcel_id: str, status: Literal["shipped", "delivered"]):
            return platform.update_parcel(account_id, order_id, parcel_id, status)

    app.mount("/", mcp_app)
    return app
