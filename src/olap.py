"""Run the sample OLAP queries and print / save each result.

Usage:
    python src/olap.py            # all operations
    python src/olap.py pivot      # one operation
"""
import sys

import pandas as pd

from db import OUTPUTS, ROOT, get_engine, load_named_queries


def main():
    queries = load_named_queries(ROOT / "sql" / "02_olap_queries.sql")
    wanted = sys.argv[1:] or list(queries)
    engine = get_engine()
    OUTPUTS.mkdir(exist_ok=True)
    for name in wanted:
        result = pd.read_sql(queries[name], engine)
        result.to_csv(OUTPUTS / f"olap_{name}.csv", index=False)
        print(f"\n=== {name.upper()} ({len(result)} rows) ===")
        print(result.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
