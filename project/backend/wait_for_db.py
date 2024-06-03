import os
import time
import psycopg2
from psycopg2 import OperationalError

def wait_for_db():
    while True:
        try:
            conn = psycopg2.connect(
                dbname=os.getenv("POSTGRES_DB"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
                host="db",
                port="5432"
            )
            conn.close()
            break
        except OperationalError:
            print("Database not ready, waiting...")
            time.sleep(1)

if __name__ == "__main__":
    wait_for_db()
