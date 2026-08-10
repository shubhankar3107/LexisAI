from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.core.config import settings


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Provide a PostgreSQL test session isolated by transaction rollback."""

    engine = create_engine(
        settings.test_database_url,
        pool_pre_ping=True,
    )

    with engine.connect() as connection:
        database_name = connection.execute(
            text("SELECT current_database()")
        ).scalar_one()

    if database_name != "lexisai_test":
        engine.dispose()

        pytest.fail(
            "Refusing to run database integration tests against "
            f"unexpected database: {database_name!r}"
        )

    session = Session(
        bind=engine,
        expire_on_commit=False,
    )

    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()