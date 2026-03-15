"""Shared utilities for the bias dataset builder."""

from database import Database


def get_database(db_path: str = "dataset/biasbuster.db") -> Database:
    """Get a Database instance with schema initialized."""
    db = Database(db_path)
    db.initialize()
    return db
