"""Step 10 - Load the two episode files into a local SQLite database (data/db/cancer_care.db).

Why SQLite: the analysis is a single-user, file-based workflow on 4,889 episode rows; SQLite needs no server, is
included in Python, enforces column types and allows SQL for the group summaries used in the report. The database
holds survey microdata, so it is git-ignored like the CSV files and is never published.

Tables:  episodes(survey, <31 episode columns>), data_dictionary(variable, definition, ...), build_log
Views:   v_che_by_survey, v_che_by_hospital
"""
import datetime, json, sqlite3
import pandas as pd
from config import PROCESSED, DB_PATH, OUTPUTS, ROOT

ROUNDS = {"nss75": "2017-18", "nss80": "2025"}


def build():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    frames = []
    for r in ROUNDS:
        f = PROCESSED / f"cancer_episodes_{r}.csv"
        if not f.exists():
            raise SystemExit(f"Missing {f.name}: run python src/01_build_cancer_episodes.py first")
        frames.append(pd.read_csv(f).assign(survey=r))
    # union of columns: the 2017-18 file has extra columns, absent ones become NULL
    both = pd.concat(frames, ignore_index=True)
    cols = ["survey"] + [c for c in both.columns if c != "survey"]
    both = both[cols]
    with sqlite3.connect(DB_PATH) as con:
        both.to_sql("episodes", con, index=False)
        dd = ROOT / "data_dictionary.csv"
        if dd.exists():
            pd.read_csv(dd).to_sql("data_dictionary", con, index=False)
        con.executescript("""
            CREATE INDEX ix_ep_survey ON episodes(survey);
            CREATE INDEX ix_ep_facility ON episodes(survey, facility_type);
            CREATE INDEX ix_ep_design ON episodes(survey, stratum, psu);
            CREATE VIEW v_che_by_survey AS
              SELECT survey, COUNT(*) AS n, ROUND(100.0 * AVG(che), 1) AS che_pct,
                     ROUND(100.0 * SUM(che * weight) / SUM(weight), 1) AS che_weighted_pct,
                     ROUND(AVG(oop_medical), 0) AS mean_oope
              FROM episodes GROUP BY survey;
            CREATE VIEW v_che_by_hospital AS
              SELECT survey, facility_type, COUNT(*) AS n, ROUND(100.0 * AVG(che), 1) AS che_pct,
                     ROUND(AVG(oop_medical), 0) AS mean_oope
              FROM episodes GROUP BY survey, facility_type;
            CREATE TABLE build_log(built_at TEXT, survey TEXT, rows_loaded INTEGER, source_file TEXT);
        """)
        now = datetime.datetime.now().isoformat(timespec="seconds")
        con.executemany("INSERT INTO build_log VALUES (?,?,?,?)",
                        [(now, r, len(f), f"cancer_episodes_{r}.csv") for r, f in zip(ROUNDS, frames)])
    return frames


def validate(frames):
    """Row counts and column totals in the database must equal the CSV files."""
    checks = {}
    with sqlite3.connect(DB_PATH) as con:
        for r, f in zip(ROUNDS, frames):
            n, che, oop = con.execute("SELECT COUNT(*), SUM(che), SUM(oop_medical) FROM episodes WHERE survey=?", (r,)).fetchone()
            ok = (n == len(f)) and (che == f["che"].sum()) and abs(oop - f["oop_medical"].sum()) < 1e-6
            checks[r] = {"rows_db": n, "rows_csv": len(f), "che_sum_match": bool(che == f["che"].sum()), "all_match": bool(ok)}
            if not ok:
                raise SystemExit(f"Database check failed for {r}: {checks[r]}")
        queries = {
            "che_by_survey": pd.read_sql_query("SELECT * FROM v_che_by_survey", con).to_dict("records"),
            "che_by_hospital": pd.read_sql_query("SELECT * FROM v_che_by_hospital", con).to_dict("records"),
            "tables": [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY 1")],
            "indexes": [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'ix_%' ORDER BY 1")],
            "schema_columns": {t: [r[1] for r in con.execute(f"PRAGMA table_info({t})")] for t in ("episodes",)},
        }
    out = {"database": str(DB_PATH.relative_to(ROOT)), "size_kb": round(DB_PATH.stat().st_size / 1024, 1), "integrity_checks": checks, **queries}
    json.dump(out, open(OUTPUTS / "database_summary.json", "w"), indent=2, default=float)
    return out


if __name__ == "__main__":
    frames = build()
    out = validate(frames)
    print(f"Wrote {out['database']} ({out['size_kb']} KB); tables/views: {out['tables']}")
    print(json.dumps(out["integrity_checks"], indent=1))
    for row in out["che_by_survey"]:
        print(row)
