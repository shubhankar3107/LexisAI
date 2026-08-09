import hashlib
from typing import BinaryIO


class ChecksumCalculator:
    def calculate(
        self,
        file: BinaryIO,
    ) -> str:
        hasher = hashlib.sha256()

        while chunk := file.read(1024 * 1024):
            hasher.update(chunk)

        return hasher.hexdigest()
