from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Column, Float, Index, JSON, MetaData, String, Table, create_engine, delete, event, select
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import make_url
from sqlalchemy.pool import StaticPool

from .core import Fault

metadata = MetaData()
records = Table(
    "shopping_records", metadata,
    Column("bucket", String(40), primary_key=True),
    Column("key", String(180), primary_key=True),
    Column("owner", String(180), nullable=False),
    Column("state", String(32), nullable=False),
    Column("due", Float),
    Column("value", JSON().with_variant(JSONB(), "postgresql"), nullable=False),
)
Index("ix_records_owner", records.c.bucket, records.c.owner)
Index("ix_records_due", records.c.bucket, records.c.state, records.c.due)


class Store:
    def __init__(self, url: str):
        url = make_url(url)
        if url.drivername == "postgresql":
            url = url.set(drivername="postgresql+psycopg")
        self.sqlite = url.get_backend_name() == "sqlite"
        options = {"pool_pre_ping": True}
        if self.sqlite:
            options["connect_args"] = {"check_same_thread": False, "timeout": 20}
            if url.database in {None, "", ":memory:"}:
                options["poolclass"] = StaticPool
            else:
                Path(url.database).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, **options)
        if self.sqlite:
            @event.listens_for(self.engine, "connect")
            def configure(connection, _):
                connection.execute("PRAGMA busy_timeout=20000")
                connection.execute("PRAGMA journal_mode=WAL")
        if not self.sqlite and url.get_backend_name() != "postgresql":
            raise ValueError("Use PostgreSQL or local SQLite.")

    def initialize(self):
        with self.transaction(True) as tx:
            if not self.sqlite:
                from sqlalchemy import text
                tx.connection.execute(text("SELECT pg_advisory_xact_lock(25316923)"))
            metadata.create_all(tx.connection)

    @contextmanager
    def transaction(self, write=False):
        with self.engine.connect() as connection:
            if write and self.sqlite:
                connection.exec_driver_sql("BEGIN IMMEDIATE")
            else:
                connection.begin()
            try:
                yield Transaction(connection, self.sqlite)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def get(self, bucket, key, owner=None):
        with self.transaction() as tx:
            return tx.get(bucket, key, owner)

    def list(self, bucket, owner=None, **query):
        with self.transaction() as tx:
            return tx.list(bucket, owner, **query)


class Transaction:
    def __init__(self, connection, sqlite):
        self.connection, self.sqlite = connection, sqlite

    def get(self, bucket, key, owner=None, *, lock=False):
        query = select(records.c.value).where(records.c.bucket == bucket, records.c.key == key)
        if owner is not None:
            query = query.where(records.c.owner == owner)
        if lock and not self.sqlite:
            query = query.with_for_update()
        return self.connection.execute(query).scalar_one_or_none()

    def require(self, bucket, key, owner=None, *, lock=False):
        value = self.get(bucket, key, owner, lock=lock)
        if value is None:
            raise Fault("not_found", "The requested resource was not found.", 404)
        return value

    def list(self, bucket, owner=None, *, state=None, due=None, limit=100, lock=False, after=None):
        query = select(records.c.value).where(records.c.bucket == bucket)
        if owner is not None:
            query = query.where(records.c.owner == owner)
        if state is not None:
            query = query.where(records.c.state == state)
        if due is not None:
            query = query.where(records.c.due <= due).order_by(records.c.due)
        if after is not None:
            query = query.where(records.c.key > after)
        if lock and not self.sqlite:
            query = query.with_for_update(skip_locked=True)
        return list(self.connection.execute(query.order_by(records.c.key).limit(limit)).scalars())

    def put(self, bucket, key, value, *, owner="", state="", due=None, once=False):
        insert = sqlite_insert if self.sqlite else pg_insert
        row = {"bucket": bucket, "key": key, "owner": owner, "state": state, "due": due, "value": value}
        stmt = insert(records).values(**row)
        cols = [records.c.bucket, records.c.key]
        if once:
            stmt = stmt.on_conflict_do_nothing(index_elements=cols)
        else:
            stmt = stmt.on_conflict_do_update(index_elements=cols, set_={
                name: getattr(stmt.excluded, name) for name in ("owner", "state", "due", "value")
            })
        return bool(self.connection.execute(stmt).rowcount)

    def delete(self, bucket, key):
        self.connection.execute(delete(records).where(records.c.bucket == bucket, records.c.key == key))
