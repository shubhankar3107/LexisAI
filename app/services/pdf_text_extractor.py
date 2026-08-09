from typing import BinaryIO

from pypdf import PdfReader

from app.services.document_text_extractor import (
    DocumentTextExtractor,
    PageText,
)


class PdfTextExtractor(DocumentTextExtractor):

    def extract(
        self,
        file: BinaryIO,
    ) -> list[PageText]:
        reader = PdfReader(file)

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            content = page.extract_text() or ""

            pages.append(
                PageText(
                    page_number=page_number,
                    content=content,
                )
            )

        return pages