import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import streamlit as st

# -------------------------------------
# GET DATABASE URL
# -------------------------------------
def get_database_url():
    # Streamlit Cloud
    if "db_url" in st.secrets:
        return st.secrets["db_url"]

    # Local environment
    return os.getenv("DATABASE_URL")


DATABASE_URL = get_database_url()

if not DATABASE_URL:
    raise ValueError("Database URL not found. Set it in secrets.toml or env variable.")

# -------------------------------------
# CREATE ENGINE
# -------------------------------------
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

# -------------------------------------
# ENGINE ACCESS (for old code)
# -------------------------------------
def get_engine():
    return engine

# -------------------------------------
# HELPER FUNCTIONS
# -------------------------------------
def get_connection():
    return engine.connect()


def execute_query(query, params=None):
    with engine.connect() as conn:
        result = conn.execute(text(query), params or {})
        conn.commit()
        return result


def fetch_all(query, params=None):
    with engine.connect() as conn:
        result = conn.execute(text(query), params or {})
        return result.fetchall()


def fetch_one(query, params=None):
    with engine.connect() as conn:
        result = conn.execute(text(query), params or {})
        return result.fetchone()