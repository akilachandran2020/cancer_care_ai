"""SQLite access layer. Scripts read episodes through load_episodes(), which uses the database when it exists and
falls back to the processed CSV files, so the pipeline works before step 10 has been run."""
import sqlite3
import pandas as pd
from config import DB_PATH, PROCESSED


def connect(path=None):
    path = path or DB_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run python src/10_build_database.py")
    return sqlite3.connect(path)


def load_episodes(round_, use_db=True):
    """One row per cancer hospitalization for 'nss75' (2017-18) or 'nss80' (2025)."""
    if use_db and DB_PATH.exists():
        with connect() as con:
            return pd.read_sql_query("SELECT * FROM episodes WHERE survey = ?", con, params=(round_,)).drop(columns="survey")
    return pd.read_csv(PROCESSED / f"cancer_episodes_{round_}.csv")


def query(sql, params=()):
    with connect() as con:
        return pd.read_sql_query(sql, con, params=params)
