from app.db.unit_of_work import UnitOfWork


class FakeSession:
    def __init__(self):
        self.commit_called = False
        self.rollback_called = False

    def commit(self):
        self.commit_called = True

    def rollback(self):
        self.rollback_called = True


def test_commit():
    session = FakeSession()
    unit_of_work = UnitOfWork(session)

    unit_of_work.commit()

    assert session.commit_called is True
    assert session.rollback_called is False


def test_rollback():
    session = FakeSession()
    unit_of_work = UnitOfWork(session)

    unit_of_work.rollback()

    assert session.rollback_called is True
    assert session.commit_called is False
