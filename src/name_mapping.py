"""Persistent name mapping database for PHI anonymization.

Stores the mapping between real names and anonymized IDs for patients
and employees. Never exported — stays local for in-office reference.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional


class NameMappingDB:
    """SQLite-backed persistent store for name mappings."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS patients (
                anonymized_id TEXT PRIMARY KEY,
                real_name TEXT NOT NULL,
                first_seen TEXT,
                last_seen TEXT,
                source_files TEXT
            );
            CREATE TABLE IF NOT EXISTS employees (
                anonymized_id TEXT PRIMARY KEY,
                real_name TEXT NOT NULL,
                first_seen TEXT,
                last_seen TEXT,
                source_files TEXT
            );
        """)
        conn.commit()
        conn.close()

    def upsert_patient(
        self,
        anonymized_id: str,
        real_name: str,
        source_file: str,
    ) -> None:
        self._upsert("patients", anonymized_id, real_name, source_file)

    def upsert_employee(
        self,
        anonymized_id: str,
        real_name: str,
        source_file: str,
    ) -> None:
        self._upsert("employees", anonymized_id, real_name, source_file)

    def _upsert(
        self,
        table: str,
        anonymized_id: str,
        real_name: str,
        source_file: str,
    ) -> None:
        conn = self._get_conn()
        now = datetime.now().isoformat()
        existing = conn.execute(
            f"SELECT * FROM {table} WHERE anonymized_id = ?",
            (anonymized_id,),
        ).fetchone()

        if existing:
            existing_files = (existing["source_files"] or "").split(",")
            if source_file not in existing_files:
                existing_files.append(source_file)
            conn.execute(
                f"""UPDATE {table}
                    SET real_name = ?, last_seen = ?, source_files = ?
                    WHERE anonymized_id = ?""",
                (
                    real_name,
                    now,
                    ",".join(existing_files),
                    anonymized_id,
                ),
            )
        else:
            conn.execute(
                f"""INSERT INTO {table}
                    (anonymized_id, real_name, first_seen, last_seen, source_files)
                    VALUES (?, ?, ?, ?, ?)""",
                (anonymized_id, real_name, now, now, source_file),
            )

        conn.commit()
        conn.close()

    def lookup_real(self, anonymized_id: str, table: str = "patients") -> Optional[str]:
        """Get real name from anonymized ID."""
        conn = self._get_conn()
        row = conn.execute(
            f"SELECT real_name FROM {table} WHERE anonymized_id = ?",
            (anonymized_id,),
        ).fetchone()
        conn.close()
        return row["real_name"] if row else None

    def lookup_anonymized(
        self, real_name: str, table: str = "patients"
    ) -> Optional[str]:
        """Get anonymized ID from real name."""
        conn = self._get_conn()
        row = conn.execute(
            f"SELECT anonymized_id FROM {table} WHERE real_name = ?",
            (real_name,),
        ).fetchone()
        conn.close()
        return row["anonymized_id"] if row else None

    def get_all(self, table: str = "patients") -> list[dict]:
        """Get all mappings."""
        conn = self._get_conn()
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_source_files(
        self, anonymized_id: str, table: str = "patients"
    ) -> list[str]:
        """Get all source files associated with an anonymized ID."""
        conn = self._get_conn()
        row = conn.execute(
            f"SELECT source_files FROM {table} WHERE anonymized_id = ?",
            (anonymized_id,),
        ).fetchone()
        conn.close()
        if row and row["source_files"]:
            return row["source_files"].split(",")
        return []
