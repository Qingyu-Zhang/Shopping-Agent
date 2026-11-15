import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from secrets import compare_digest
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import jwt
from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import Field, field_validator, model_validator

from .automation import WatchRequest
from .core import Fault, Input, Line, Settings, normalize_lines, stable_id, timestamp
from .platform import MockPlatform
from .runtime import Runtime, Worker


class Turn(Input):
    text: str = Field(min_length=2, max_length=2200)
    request_id: str | None = Field(default=None, min_length=1, max_length=180)
    address_id: str | None = Field(default=None, min_length=1, max_length=180)
    expected_version: int | None = Field(default=None, ge=0)
    plan_version: int | None = Field(default=None, ge=1)
    timezone: str = "America/New_York"
    image_base64: str | None = Field(default=None, max_length=10_700_000)

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value):
        try:
            ZoneInfo(value)
        except (ValueError, ZoneInfoNotFoundError):
            raise ValueError("Use an IANA timezone.") from None
        return value


class Clarification(Input):
    interrupt_id: str
    text: str = Field(min_length=1, max_length=1200)


class Selection(Input):
    task_id: str
    plan_id: str
    plan_version: int = Field(ge=1)


class CheckoutQuote(Input):
    address_id: str | None = None
    selection: Selection | None = None
    cart_version: int | None = Field(default=None, ge=0)
    lines: list[Line] | None = Field(default=None, min_length=1, max_length=20)

    @model_validator(mode="after")
    def source(self):
        if sum(x is not None for x in (self.selection, self.cart_version, self.lines)) != 1:
            raise ValueError("Choose a plan, a cart version, or exact offer lines.")
        return self


class Confirm(Input):
    terms: str = Field(min_length=64, max_length=64)


class CartMutation(Input):
    operation_id: str = Field(min_length=1, max_length=160)
    expected_version: int = Field(ge=0)
    mode: Literal["add", "remove"] = "add"
    selection: Selection | None = None
    lines: list[Line] | None = Field(default=None, min_length=1, max_length=20)

    @model_validator(mode="after")
    def source(self):
        if (self.selection is None) == (self.lines is None):
            raise ValueError("Choose a plan or exact offer lines.")
        return self


class ReturnConfirm(Confirm):
    reason: str = Field(min_length=2, max_length=300)


class Profile(Input):
    lightweight: bool | None = None
    address_id: str | None = Field(default=None, min_length=1, max_length=180)


class OfferEvent(Input):
    event_id: str = Field(min_length=1, max_length=180)
    offer_id: str = Field(min_length=1, max_length=160)


class OfferChange(Input):
    unit_price: int | None = Field(default=None, ge=0, le=100_000_000)
    stock: int | None = Field(default=None, ge=0, le=10_000)
    lead_hours: int | None = Field(default=None, ge=0, le=720)


def public(row):
    return {k: v for k, v in row.items() if k not in {"lease", "input", "request", "resume"}}


def create_app(settings: Settings | None = None, clock=timestamp):
    settings = settings or Settings.from_env()
    if not settings.demo and len(settings.auth_jwt_secret) < 32:
        raise ValueError("Set AUTH_JWT_SECRET to a secure value in non-demo deployments.")

    @asynccontextmanager
    async def lifespan(app):
        runtime = Runtime(settings, clock)
        app.state.runtime = runtime
        worker = Worker(runtime) if settings.embedded_worker else None
        if worker:
            worker.start()
        try:
            yield
        finally:
            if worker:
                await asyncio.to_thread(worker.close)
            runtime.close()

    app = FastAPI(title="Shopping Agent V2", version="0.1.0", lifespan=lifespan)
    bearer = HTTPBearer(auto_error=False)

    def get_runtime(request: Request):
        return request.app.state.runtime

    def authenticate(auth: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
        if auth:
            token = auth.credentials
            if settings.demo and token in {"demo-alex", "demo-sam"}:
                return token.removeprefix("demo-")
            if settings.auth_jwt_secret:
                try:
                    claims = jwt.decode(token, settings.auth_jwt_secret, algorithms=["HS256"],
                                        issuer=settings.auth_issuer, audience=settings.auth_audience,
                                        options={"require": ["exp", "sub", "iss", "aud"]})
                    account = claims["sub"]
                    if isinstance(account, str) and 1 <= len(account) <= 180:
                        return account
                except jwt.PyJWTError:
                    pass
        raise Fault("unauthorized", "A valid account token is required.", 401)

    Account = Annotated[str, Depends(authenticate)]
    UseRuntime = Annotated[Runtime, Depends(get_runtime)]

    @app.exception_handler(Fault)
    async def error_response(_, error):
        return JSONResponse({"code": error.code, "message": str(error)}, status_code=error.status)

    @app.get("/health")
    def health(runtime: UseRuntime):
        try:
            runtime.ready()
        except Exception:
            raise Fault("not_ready", "One or more configured services are unavailable.", 503)
        return {"status": "ready"}

    @app.get("/v1/me")
    def me(account: Account, runtime: UseRuntime):
        profile = runtime.platform.call("commerce_account", {"account_id": account})
        if profile.get("id") != account:
            raise Fault("platform_protocol", "Account data could not be verified.", 502)
        return profile

    @app.get("/v1/preferences")
    def get_preferences(account: Account, runtime: UseRuntime):
        return runtime.store.get("preferences", account, account) or {}

    @app.put("/v1/preferences")
    def set_preferences(body: Profile, account: Account, runtime: UseRuntime):
        if body.address_id and body.address_id not in me(account, runtime).get("addresses", {}):
            raise Fault("address_not_found", "The selected address is unavailable.", 404)
        data = body.model_dump(exclude_none=True)
        with runtime.store.transaction(True) as tx:
            tx.put("preferences", account, data, owner=account)
        return data

    @app.post("/v1/tasks", status_code=201)
    def create_task(account: Account, runtime: UseRuntime):
        return runtime.runs.start_task(account)

    def required(runtime, section, key, account):
        with runtime.store.transaction() as tx:
            return tx.require(section, key, account)

    @app.get("/v1/tasks")
    def tasks(account: Account, runtime: UseRuntime):
        return sorted(runtime.store.list("tasks", account, limit=100), key=lambda x: x["updated_at"], reverse=True)

    @app.get("/v1/tasks/{task_id}")
    def task(task_id: str, account: Account, runtime: UseRuntime):
        return required(runtime, "tasks", task_id, account)

    @app.post("/v1/tasks/{task_id}/turns", status_code=202)
    def turn(task_id: str, body: Turn, account: Account, runtime: UseRuntime):
        if runtime.redis:
            from redis.exceptions import RedisError
            try:
                key = "shopping:turns:" + stable_id(account)
                result = runtime.redis.eval(
                    "local n=redis.call('INCR',KEYS[1]); if n==1 then redis.call('EXPIRE',KEYS[1],60) end; return n",
                    1, key,
                )
                if result > 35:
                    raise Fault("rate_limit", "Too many shopping requests. Try again shortly.", 429)
            except RedisError:
                raise Fault("rate_limit_unavailable", "The shopping service is briefly unavailable.", 503)
        return public(runtime.runs.start_turn(account, task_id, body.model_dump()))

    @app.get("/v1/runs/{run_id}")
    def run(run_id: str, account: Account, runtime: UseRuntime):
        return public(required(runtime, "runs", run_id, account))

    @app.post("/v1/runs/{run_id}/clarification", status_code=202)
    def clarify(run_id: str, body: Clarification, account: Account, runtime: UseRuntime):
        return public(runtime.runs.answer(account, run_id, body.interrupt_id, body.text))

    @app.get("/v1/runs/{run_id}/events")
    def events(run_id: str, request: Request, account: Account, runtime: UseRuntime,
               last_event_id: Annotated[int, Header(ge=0)] = 0):
        required(runtime, "runs", run_id, account)

        async def stream():
            cursor = last_event_id
            while not await request.is_disconnected():
                row = await asyncio.to_thread(runtime.store.get, "runs", run_id, account)
                updates = [event for event in row["events"] if event["id"] > cursor]
                for event in updates:
                    cursor = event["id"]
                    payload = dict(event)
                    if event is updates[-1] and row["state"] in {"completed", "needs_input", "failed"}:
                        payload.update({name: row[name] for name in ("result", "interrupt", "error") if name in row})
                    yield f"id: {cursor}\nevent: {event['state']}\ndata: {json.dumps(payload)}\n\n"
                if row["state"] in {"completed", "needs_input", "failed"}:
                    break
                if not updates:
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/v1/catalog/search")
    def search(q: Annotated[str, Query(min_length=2, max_length=240)], account: Account,
               runtime: UseRuntime, category: str | None = None):
        docs, _ = runtime.catalog.search(q, category)
        return [runtime.planner._public(doc) for doc in docs]

    @app.get("/v1/products/{product_id}/evidence")
    def evidence(product_id: str, account: Account, runtime: UseRuntime,
                 q: Annotated[str, Query(min_length=2, max_length=240)] = "quality experience"):
        from .graph import public_evidence
        return [public_evidence(row) for row in runtime.catalog.evidence([product_id], q)]

    def selected(runtime, account, selection):
        view = required(runtime, "task_views", stable_id(selection.task_id, str(selection.plan_version)), account)
        plan = next((x for x in view["plans"] if x["id"] == selection.plan_id), None)
        if plan is None:
            raise Fault("not_found", "The selected plan was not found.", 404)
        lines = [{"offer_id": x["offer_id"], "quantity": x["quantity"]} for x in plan["quote"]["lines"]]
        return plan, lines, view["goal"]

    @app.get("/v1/cart")
    def cart(account: Account, runtime: UseRuntime):
        return runtime.platform.call("commerce_cart", {"account_id": account})

    @app.post("/v1/cart/changes")
    def cart_change(body: CartMutation, account: Account, runtime: UseRuntime):
        lines = selected(runtime, account, body.selection)[1] if body.selection else [line.model_dump() for line in body.lines]
        return runtime.platform.call("commerce_cart_write", {
            "account_id": account, "operation_id": body.operation_id, "expected_version": body.expected_version,
            "mode": body.mode, "lines": normalize_lines(lines),
        })

    @app.post("/v1/quotes", status_code=201)
    def propose_quote(body: CheckoutQuote, account: Account, runtime: UseRuntime):
        limits = {}
        expected_skus = None
        address = body.address_id
        if body.selection:
            plan, lines, limits = selected(runtime, account, body.selection)
            expected_skus = {x["offer_id"]: x["sku_id"] for x in plan["quote"]["lines"]}
            address = address or plan["quote"]["address_id"]
        elif body.cart_version is not None:
            value = cart(account, runtime)
            if value["version"] != body.cart_version:
                raise Fault("cart_conflict", "The cart changed. Refresh it and retry.")
            lines = value["lines"]
        else:
            lines = [line.model_dump() for line in body.lines]
        return runtime.checkout.prepare(account, address or "home", lines, constraints=limits, expected_skus=expected_skus)

    @app.post("/v1/quotes/{receipt_id}/confirm", status_code=202)
    def confirm(receipt_id: str, body: Confirm, account: Account, runtime: UseRuntime):
        return public(runtime.checkout.confirm(account, receipt_id, body.terms))

    @app.get("/v1/checkouts/{attempt_id}")
    def checkout(attempt_id: str, account: Account, runtime: UseRuntime):
        return public(required(runtime, "checkouts", attempt_id, account))

    @app.get("/v1/orders")
    def orders(account: Account, runtime: UseRuntime):
        return [x for x in runtime.platform.call("commerce_orders", {"account_id": account})
                if x.get("account_id") == account]

    @app.get("/v1/orders/{order_id}")
    def order(order_id: str, account: Account, runtime: UseRuntime):
        value = runtime.platform.call("commerce_order", {"account_id": account, "order_id": order_id})
        if value.get("id") != order_id or value.get("account_id") != account:
            raise Fault("not_found", "The requested order was not found.", 404)
        return value

    @app.post("/v1/orders/{order_id}/return-quotes", status_code=201)
    def preview_return(order_id: str, body: Line, account: Account, runtime: UseRuntime):
        return runtime.checkout.preview_return(account, order_id, body.offer_id, body.quantity)

    @app.post("/v1/return-quotes/{receipt_id}/confirm")
    def submit_return(receipt_id: str, body: ReturnConfirm, account: Account, runtime: UseRuntime):
        return runtime.checkout.confirm_return(account, receipt_id, body.terms, body.reason)

    @app.get("/v1/offers/{offer_id}/price-history")
    def price_history(offer_id: str, account: Account, runtime: UseRuntime):
        return runtime.platform.call("commerce_history", {"offer_id": offer_id})

    @app.post("/v1/automations", status_code=201)
    def create_automation(body: WatchRequest, account: Account, runtime: UseRuntime):
        return runtime.automations.create(account, body)

    @app.get("/v1/automations")
    def automations(account: Account, runtime: UseRuntime):
        return [public(x) for x in runtime.store.list("automations", account, limit=100)]

    @app.delete("/v1/automations/{watch_id}")
    def cancel(watch_id: str, account: Account, runtime: UseRuntime):
        return public(runtime.automations.cancel(account, watch_id))

    @app.get("/v1/automations/{watch_id}/runs")
    def automation_runs(watch_id: str, account: Account, runtime: UseRuntime):
        required(runtime, "automations", watch_id, account)
        return [public(x) for x in runtime.store.list("automation_runs", account, limit=200) if x["watch_id"] == watch_id]

    @app.get("/v1/notices")
    def notices(account: Account, runtime: UseRuntime):
        return sorted(runtime.store.list("notices", account, limit=100), key=lambda x: x["at"], reverse=True)

    @app.post("/internal/offer-events")
    def offer_event(body: OfferEvent, runtime: UseRuntime, authorization: Annotated[str, Header()] = ""):
        if not compare_digest(authorization, "Bearer " + settings.platform_token):
            raise Fault("unauthorized", "A valid platform token is required.", 401)
        return {"accepted": runtime.automations.wake(body.event_id, body.offer_id)}

    if settings.demo:
        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        def index():
            return Path(__file__).with_name("demo.html").read_text(encoding="utf-8")

        def mock(runtime):
            if not isinstance(runtime.platform, MockPlatform):
                raise Fault("remote_platform", "Run this operation on the connected platform sandbox.")
            return runtime.platform

        @app.patch("/demo/offers/{offer_id}")
        def demo_offer(offer_id: str, body: OfferChange, account: Account, runtime: UseRuntime):
            return mock(runtime).change_offer(offer_id, body.model_dump(exclude_none=True))

        @app.put("/demo/payment")
        def demo_payment(mode: Literal["lost_response", "decline"], account: Account, runtime: UseRuntime):
            mock(runtime).payment_fault(mode)
            return {"mode": mode}

        @app.patch("/demo/orders/{order_id}/parcels/{parcel_id}")
        def demo_parcel(order_id: str, parcel_id: str, status: Literal["shipped", "delivered"],
                        account: Account, runtime: UseRuntime):
            return mock(runtime).update_parcel(account, order_id, parcel_id, status)

    return app
