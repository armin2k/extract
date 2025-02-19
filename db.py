# db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base
import os

# For a simple setup, we’ll use SQLite.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///balancesheets.db")

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)