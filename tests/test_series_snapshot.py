#!/usr/bin/env python3
"""Snapshot guard for the Layer-2 metric contract + series pivot.

`compute_run_metrics(run)` is the single source of truth for metric VALUES, and
`per_run_series` is a pure pivot over it. This test pins their combined output:
if a refactor (or a new metric) changes any chart value, this fails loudly.

Regenerate the fixture intentionally with:  python tests/test_series_snapshot.py --update

    python tests/test_series_snapshot.py          # check
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from storage.sqlite_store import set_active_db_path, resolve_db_path  # noqa: E402
from analysis.layer1_raw import list_runs                            # noqa: E402
from analysis.metric_series import per_run_series                           # noqa: E402

FIXTURE = os.path.join(ROOT, "tests", "fixtures", "series_snapshot.json")
DBS = ["demo"]  # the committed sample database (private project DBs are gitignored)


def _current() -> dict:
    out = {}
    for db in DBS:
        set_active_db_path(resolve_db_path(db))
        # round-trip through JSON so the comparison is order-insensitive on keys
        out[db] = json.loads(json.dumps(per_run_series(list_runs(2000)), sort_keys=True))
    return out


def test_series_snapshot():
    """per_run_series output must match the pinned fixture for every project."""
    baseline = json.load(open(FIXTURE))
    current = _current()
    for db in baseline:
        assert current[db] == baseline[db], f"series output changed for '{db}'"


if __name__ == "__main__":
    if "--update" in sys.argv:
        json.dump(_current(), open(FIXTURE, "w"), sort_keys=True, indent=0)
        print("fixture updated.")
    else:
        try:
            test_series_snapshot()
            print("PASS — per_run_series matches the snapshot for all projects.")
        except AssertionError as e:
            print("FAIL —", e)
            sys.exit(1)
