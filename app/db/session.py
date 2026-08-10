from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings


def create_session_factory(
    database_url: str,
) -> sessionmaker[Session]:
    engine = create_engine(
        database_url,
        pool_pre_ping=True,
    )

    return sessionmaker(
        bind=engine,
        class_=Session,
        expire_on_commit=False,
    )


SessionLocal = create_session_factory(settings.database_url)


def get_session(
    session_factory: sessionmaker[Session],
) -> Generator[Session, None, None]:
    session = session_factory()

    try:
        yield session
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    yield from get_session(SessionLocal)