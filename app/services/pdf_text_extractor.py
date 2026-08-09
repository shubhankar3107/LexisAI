from typing import BinaryIO

from pypdf import PdfReader

from app.services.document_text_extractor import DocumentTextExtractor


class PdfTextExtractor(DocumentTextExtractor):

    def extract(
        self,
        file: BinaryIO,
    ) -> tuple[str, int]:
        reader = PdfReader(file)

        pages = reader.pages

        text = "\n".join(page.extract_text() or "" for page in pages)

        return text, len(pages)
