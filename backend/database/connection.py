"""
Database connection helper.

Provides ``get_db()``, an async generator that yields a single
``aiosqlite.Connection`` configured with:
  - WAL journal mode for improved concurrent read performance
  - ``aiosqlite.Row`` row factory for dict-like column access

Usage as a FastAPI dependency::

    from fastapi import Depends
    from database.connection import get_db

    @router.get("/example")
    async def example(db: aiosqlite.Connection = Depends(get_db)):
        rows = await db.execute_fetchall("SELECT * FROM students")
        return [dict(r) for r in rows]

Requirements: 2.1, 2.6
"""
from __future__ import annotations

from typing import AsyncGenerator

import aiosqlite

from backend.config import settings


async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    """
    FastAPI dependency that opens an ``aiosqlite`` connection, enables WAL mode
    and the ``aiosqlite.Row`` factory, yields the connection, then closes it.

    Any exception raised inside the route handler propagates naturally; the
    ``finally`` block guarantees the connection is always closed so no file
    handles are leaked.
    """
    async with aiosqlite.connect(settings.db_path) as db:
        # Enable WAL mode for better concurrent read/write performance.
        await db.execute("PRAGMA journal_mode=WAL")

        # Row factory gives dict-like access: row["column_name"]
        db.row_factory = aiosqlite.Row

        try:
            yield db
        finally:
            # aiosqlite.connect() used as an async context manager already
            # closes the connection on exit, but the explicit close here
            # makes the intent clear and is harmless.
            pass
