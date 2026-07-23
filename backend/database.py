import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Load .env here too: database.py is imported before engine.py (which also calls
# load_dotenv), so DATABASE_URL must be available this early.
load_dotenv()

# Durable storage for users, wishlist, search history and interactions.
# Local dev defaults to SQLite; set DATABASE_URL to a hosted Postgres
# (Neon / Supabase / Render) so data survives redeploys and file resets.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./backend/freeme.db")

# Some providers still hand out legacy "postgres://" URLs; SQLAlchemy wants "postgresql://".
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

_is_sqlite = DATABASE_URL.startswith("sqlite")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    pool_pre_ping=True,  # revive connections serverless Postgres (Neon) drops when idle
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
