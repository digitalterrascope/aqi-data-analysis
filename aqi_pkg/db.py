from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
import dotenv
import os

ENGINE_URL = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"

engine = create_engine(
    ENGINE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
)

SessionLocal = scoped_session(
    sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )
)

def get_session():
    """
    Returns a SQLAlchemy session.
    Caller is responsible for closing it.
    """
    return SessionLocal()
