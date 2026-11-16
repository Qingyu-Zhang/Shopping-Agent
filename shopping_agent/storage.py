from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Column, Float, Index, JSON, MetaData, String, Table, create_engine, delete, event, or_, select
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
    def __init__(self, url: str, *, read_only=False):
        url = make_url(url)
        if url.drivername == "postgresql":
            url = url.set(drivername="postgresql+psycopg")
        self.sqlite = url.get_backend_name() == "sqlite"
        self.read_only = read_only
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
                if self.read_only:
                    connection.execute("PRAGMA query_only=ON")
                else:
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
        if write and self.read_only:
            raise ValueError("This database connection is read-only.")
        with self.engine.connect() as connection:
            if write and self.sqlite:
                connection.exec_driver_sql("BEGIN IMMEDIATE")
            else:
                connection.begin()
            try:
                if self.read_only and not self.sqlite:
                    connection.exec_driver_sql("SET TRANSACTION READ ONLY")
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

    def get_many(self, bucket, keys):
        with self.transaction() as tx:
            return tx.get_many(bucket, keys)

    def catalog_for_products(self, product_ids):
        relation = (or_(records.c.owner.in_(product_ids), records.c.key.in_(product_ids)) if self.read_only
                    else records.c.value["product_id"].as_string().in_(product_ids))
        query = select(records.c.value).where(
            records.c.bucket == "catalog", relation,
        )
        with self.transaction() as tx:
            return list(tx.connection.execute(query).scalars())

    def task_audit(self, owner, task_id, limit):
        query = select(records.c.value).where(
            records.c.bucket == "audit", records.c.owner == owner,
            records.c.value["task_id"].as_string() == task_id,
        ).order_by(records.c.value["at"].as_float().desc(), records.c.key.desc()).limit(limit)
        with self.engine.connect() as connection:
            return list(reversed(connection.execute(query).scalars().all()))


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

    def get_many(self, bucket, keys):
        query = select(records.c.key, records.c.value).where(records.c.bucket == bucket, records.c.key.in_(keys))
        return dict(self.connection.execute(query).all())

    def list(self, bucket, owner=None, *, state=None, due=None, limit=100, lock=False, after=None):
        query = select(records.c.value).where(records.c.bucket == bucket)
        if isinstance(owner, (list, tuple)):
            query = query.where(records.c.owner.in_(owner))
        elif owner is not None:
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

    def put_many(self, bucket, documents, *, owner_field=None, once=False):
        rows = [{"bucket": bucket, "key": doc["id"], "owner": doc[owner_field] if owner_field else "",
                 "state": "", "due": None, "value": doc} for doc in documents]
        if not rows:
            return 0
        insert = sqlite_insert if self.sqlite else pg_insert
        stmt = insert(records).values(rows)
        cols = [records.c.bucket, records.c.key]
        if once:
            stmt = stmt.on_conflict_do_nothing(index_elements=cols)
        else:
            stmt = stmt.on_conflict_do_update(index_elements=cols, set_={
                "owner": stmt.excluded.owner, "value": stmt.excluded.value,
            })
        return len(self.connection.execute(stmt.returning(records.c.key)).all())

    def delete(self, bucket, key):
        self.connection.execute(delete(records).where(records.c.bucket == bucket, records.c.key == key))
