import hashlib
from io import BytesIO

from app.services.checksum import ChecksumCalculator


def test_calculate_checksum():
    content = b"LexisAI"
    file = BytesIO(content)

    checksum = ChecksumCalculator().calculate(file)

    expected = hashlib.sha256(content).hexdigest()

    assert checksum == expected


def test_calculate_checksum_reads_complete_file():
    content = b"LexisAI test content" * 1000
    file = BytesIO(content)

    checksum = ChecksumCalculator().calculate(file)

    assert checksum == hashlib.sha256(content).hexdigest()
