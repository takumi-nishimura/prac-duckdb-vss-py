import random
from datetime import datetime
from pathlib import Path

import duckdb

db_name = "test_duckdb"
db_path = Path(__file__).parent.parent / f"data/{db_name}.db"
print(f"Database path: {db_path}")

if db_path.exists():
    print(f"Database {db_name} already exists at {db_path}.")
    con = duckdb.connect(db_path)
else:
    con = duckdb.connect(db_path)
    print(f"Database {db_name} created at {db_path}.")


def table_exists(con, table_name):
    """
    Check if a table exists in the database.
    """
    try:
        con.sql(f"SELECT * FROM {table_name} LIMIT 1")
        return True
    except duckdb.CatalogException:
        return False


table_name = "access_log"
if table_exists(con, table_name):
    print(f"Table {table_name} already exists.")
else:
    print(f"Creating table {table_name}...")
    con.sql(
        f"""
        CREATE TABLE {table_name} (
            asctime TIMESTAMP,
            id INTEGER,
        )
        """
    )
    print(f"Table {table_name} created.")

__now_time = datetime.now()
__id = random.randint(0, 100)
print(f"Current time: {__now_time}")
print(f"UUID: {__id}")

con.sql(
    f"""
    INSERT INTO {table_name} (asctime, id)
    VALUES ('{__now_time}', {__id})
    """
)
print(f"Inserted data into {table_name}.")

query = f"SELECT * FROM {table_name}"
result = con.sql(query)
print("Query result:")
print(result)
