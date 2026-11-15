from __future__ import annotations

import logging
import sqlite3
from threading import Event, Thread

from langgraph.types import Command
from sqlalchemy.engine import make_url

from .automation import Automations
from .catalog import Catalog
from .checkout import Checkout, audit
from .core import Fault, Settings, new_id, stable_id, timestamp
from .graph import ShoppingGraph
from .models import Models
from .planner import Planner
from .platform import RemotePlatform, boot_platform
from .seed import seed_catalog
from .storage import Store

LOG = logging.getLogger(__name__)


class Runs:
    def __init__(self, store, graph, clock):
        self.store, self.graph, self.clock = store, graph, clock

    def start_task(self, account):
        key = new_id()
        task = {"id": key, "version": 0, "goal": {}, "plans": [], "updated_at": self.clock()}
        with self.store.transaction(True) as tx:
            tx.put("tasks", key, task, owner=account)
        return task

    def start_turn(self, account, task_id, data):
        now = self.clock()
        unique = data.pop("request_id") or new_id()
        run_id = stable_id(account, task_id, unique)
        with self.store.transaction(True) as tx:
            task = tx.require("tasks", task_id, account, lock=True)
            if existing := tx.get("runs", run_id, account):
                if existing["request"] != data:
                    raise Fault("operation_conflict", "The request ID was used for another turn.")
                return existing
            requested = data["expected_version"]
            if requested is not None and requested != task["version"]:
                raise Fault("task_conflict", "The task was changed in another session. Refresh and retry.")
            view = task if data["plan_version"] is None else tx.require(
                "task_views", stable_id(task_id, str(data["plan_version"])), account
            )
            preferences = tx.get("preferences", account, account) or {}
            address = data["address_id"] or task["goal"].get("address_id") or preferences.get("address_id") or "home"
            previous = dict(task["goal"])
            if "light" not in previous and preferences.get("lightweight") is not None:
                previous["light"] = preferences["lightweight"]
            row = {"id": run_id, "task_id": task_id, "account_id": account, "base_version": task["version"],
                   "state": "queued", "events": [{"id": 1, "state": "queued"}],
                   "request": data, "created_at": now,
                   "input": {"text": data["text"], "now": now, "timezone": data["timezone"],
                             "previous": previous, "prior_plans": view["plans"],
                             "address": address, "account": account, "image": data["image_base64"]}}
            tx.put("runs", run_id, row, owner=account, state="pending", due=now)
            return row

    def answer(self, account, run_id, interrupt_id, answer):
        with self.store.transaction(True) as tx:
            row = tx.require("runs", run_id, account, lock=True)
            if row.get("resume") == {"id": interrupt_id, "answer": answer}:
                return row
            if row["state"] != "needs_input" or row["interrupt"]["id"] != interrupt_id:
                raise Fault("run_state", "This clarification is no longer pending.")
            row["resume"] = {"id": interrupt_id, "answer": answer}
            self.add_event(row, "queued")
            tx.put("runs", run_id, row, owner=account, state="pending", due=self.clock())
            return row

    @staticmethod
    def add_event(row, state):
        row["state"] = state
        if not row["events"] or row["events"][-1]["state"] != state:
            row["events"] = [*row["events"][-39:], {"id": row["events"][-1]["id"]+1 if row["events"] else 1,
                                                   "state": state}]

    def step(self, row, lease):
        run_id, account = row["id"], row["account_id"]
        config = {"configurable": {"thread_id": run_id}, "recursion_limit": 18}
        try:
            snap = self.graph.get_state(config)
            if snap.interrupts:
                pending = snap.interrupts[0]
                if row.get("resume", {}).get("id") == pending.id:
                    self.graph.invoke(Command(resume={pending.id: row["resume"]["answer"]}), config, durability="sync")
            elif snap.next:
                self.graph.invoke(None, config, durability="sync")
            elif not snap.values:
                self.graph.invoke(row["input"], config, durability="sync")
            snap = self.graph.get_state(config)
            if snap.interrupts:
                pending = snap.interrupts[0]
                self._finish(row, lease, "needs_input", interrupt={"id": pending.id, **pending.value})
            else:
                self._commit(row, lease, snap.values)
        except Fault as error:
            self._finish(row, lease, "failed", error={"code": error.code, "message": str(error)})
        except Exception:
            LOG.exception("Shopping workflow failed run=%s", run_id)
            self._finish(row, lease, "failed", error={"code": "run_failed", "message": "The shopping workflow could not finish."})

    def _commit(self, row, lease, state):
        account, key = row["account_id"], row["id"]
        with self.store.transaction(True) as tx:
            task = tx.require("tasks", row["task_id"], account, lock=True)
            current = tx.require("runs", key, account, lock=True)
            if current.get("lease") != lease:
                return
            current.pop("lease", None)
            current.pop("interrupt", None)
            if task["version"] != row["base_version"]:
                current["error"] = {"code": "task_conflict", "message": "The task changed while this turn was running."}
                self.add_event(current, "failed")
            else:
                result = state["result"]
                next_version = task["version"]+1
                action = result.get("suggested_action")
                if action:
                    action = {**action, "plan_version": row["request"]["plan_version"] or task["version"]}
                if not result.get("preserve_plans"):
                    task["plans"] = result["plans"]
                    task["goal"] = state["goal"]
                task.update(version=next_version, updated_at=self.clock())
                if action:
                    result = {**result, "suggested_action": action}
                current["result"] = {**result, "goal": task["goal"], "task_version": next_version}
                self.add_event(current, "completed")
                tx.put("tasks", task["id"], task, owner=account)
                tx.put("task_views", stable_id(task["id"], str(next_version)), task, owner=account)
                audit(tx, account, "turn_completed", key, self.clock(), task_id=task["id"], version=next_version)
            tx.put("runs", key, current, owner=account, state="finished")

    def _finish(self, row, lease, state, **fields):
        with self.store.transaction(True) as tx:
            current = tx.require("runs", row["id"], row["account_id"], lock=True)
            if current.get("lease") != lease:
                return
            current.pop("lease", None)
            current.update(fields)
            self.add_event(current, state)
            tx.put("runs", row["id"], current, owner=row["account_id"], state=state)


class Runtime:
    def __init__(self, settings: Settings, clock=timestamp, *, setup_graph=True):
        self.settings, self.clock = settings, clock
        self.store = Store(settings.database)
        self.store.initialize()
        if settings.demo:
            seed_catalog(self.store)
        self.platform = RemotePlatform(settings) if settings.platform_url else boot_platform(settings, clock)
        self.redis = None
        if settings.redis_url:
            from redis import Redis
            self.redis = Redis.from_url(settings.redis_url, socket_timeout=1, socket_connect_timeout=1)
        self.models = Models(settings, cache=self.redis)
        self.catalog = Catalog(self.store, self.models, settings)
        self.planner = Planner(self.catalog, self.platform)
        self.checkout = Checkout(self.store, self.platform, self.planner, settings, clock)
        self.automations = Automations(self.store, self.platform, self.planner, settings, clock)
        self.checkpoint_connection = None
        if setup_graph:
            self.runs = Runs(self.store, ShoppingGraph(self.catalog, self.models, self.planner,
                                                       self._checkpointer()).graph, clock)

    def _checkpointer(self):
        url = make_url(self.settings.database)
        if self.store.sqlite:
            from langgraph.checkpoint.sqlite import SqliteSaver
            destination = ":memory:" if url.database in {None, "", ":memory:"} else url.database + ".checkpoints"
            self.checkpoint_connection = sqlite3.connect(destination, check_same_thread=False, timeout=20)
            return SqliteSaver(self.checkpoint_connection)
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
        from sqlalchemy import text

        self.checkpoint_connection = ConnectionPool(
            conninfo=url.set(drivername="postgresql").render_as_string(hide_password=False),
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            min_size=1, max_size=6, open=False,
        )
        self.checkpoint_connection.open(wait=True)
        saver = PostgresSaver(self.checkpoint_connection)
        with self.store.transaction(True) as tx:
            tx.connection.execute(text("SELECT pg_advisory_xact_lock(25316924)"))
            saver.setup()
        return saver

    def ready(self):
        from sqlalchemy import text
        with self.store.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        if self.settings.elastic_url:
            for bucket in ("catalog", "evidence"):
                self.catalog.http.head(f"/{self.settings.index}-{bucket}").raise_for_status()
        if self.redis:
            self.redis.ping()

    def close(self):
        if self.checkpoint_connection:
            self.checkpoint_connection.close()
        if self.redis:
            self.redis.close()
        self.models.http.close()
        self.catalog.http.close()
        if isinstance(self.platform, RemotePlatform):
            self.platform.http.close()
        else:
            self.platform.store.engine.dispose()
        self.store.engine.dispose()


class Worker:
    def __init__(self, runtime):
        self.runtime = runtime
        self.stop = Event()
        self.threads = []
        self.handlers = {
            "runs": ("pending", runtime.runs.step),
            "checkouts": ("pending", runtime.checkout.step),
            "automations": ("active", runtime.automations.step),
            "automation_runs": ("pending", runtime.automations.cart_step),
        }

    def tick(self, bucket):
        if bucket == "offer_events":
            return self._events()
        state, handler = self.handlers[bucket]
        runtime = self.runtime
        with runtime.store.transaction(True) as tx:
            found = tx.list(bucket, state=state, due=runtime.clock(), limit=1, lock=True)
            if not found:
                return False
            row = found[0]
            lease = new_id()
            row["lease"] = lease
            if bucket == "runs":
                Runs.add_event(row, "running")
            tx.put(bucket, row["id"], row, owner=row["account_id"], state=state, due=runtime.clock()+420)
        handler(row, lease)
        return True

    def _events(self):
        runtime = self.runtime
        if isinstance(runtime.platform, RemotePlatform):
            return False
        for row in runtime.platform.store.list("offer_events", state="pending", limit=100):
            runtime.automations.wake(row["id"], row["offer_id"])
            with runtime.platform.store.transaction(True) as tx:
                tx.delete("offer_events", row["id"])
        return False

    def _loop(self, bucket):
        while not self.stop.is_set():
            try:
                active = self.tick(bucket)
            except Exception:
                LOG.exception("Worker step failed bucket=%s", bucket)
                active = False
            if not active:
                self.stop.wait(0.5 if bucket in {"runs", "checkouts"} else 2)

    def start(self):
        for bucket in [*self.handlers, "offer_events"]:
            thread = Thread(target=self._loop, args=(bucket,), name="worker-" + bucket, daemon=True)
            thread.start()
            self.threads.append(thread)

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=30)
