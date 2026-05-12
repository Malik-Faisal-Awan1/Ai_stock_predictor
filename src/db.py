import sqlite3
import pandas as pd

def get_db_connection():
    conn = sqlite3.connect("runs.db")
    with conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS runs 
            (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, model TEXT, accuracy REAL, time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
        )
    return conn

def log_run(ticker, model, accuracy):
    conn = get_db_connection()
    with conn:
        conn.execute("INSERT INTO runs (ticker, model, accuracy) VALUES (?, ?, ?)", (ticker, model, float(accuracy)))
    conn.close()

def get_run_history():
    conn = get_db_connection()
    df = None
    try:
        df = pd.read_sql_query("SELECT * FROM runs ORDER BY time DESC LIMIT 10", conn)
    finally:
        conn.close()
    return df
