from io import BytesIO

from reportlab.pdfgen import canvas

from app.services.pdf_text_extractor import PdfTextExtractor
import pytest


def create_pdf(text: str) -> BytesIO:
    buffer = BytesIO()

    pdf = canvas.Canvas(buffer)
    pdf.drawString(100, 750, text)
    pdf.save()

    buffer.seek(0)

    return buffer


def test_extract_returns_page_text():
    extractor = PdfTextExtractor()

    pdf = create_pdf("Hello world")

    result = extractor.extract(pdf)

    assert len(result) == 1
    assert result[0].page_number == 1
    assert result[0].content.strip() == "Hello world"


def create_multi_page_pdf() -> BytesIO:
    buffer = BytesIO()

    pdf = canvas.Canvas(buffer)

    pdf.drawString(100, 750, "Page one text")
    pdf.showPage()

    pdf.drawString(100, 750, "Page two text")
    pdf.showPage()

    pdf.save()

    buffer.seek(0)

    return buffer


def test_extract_returns_text_for_each_page():
    extractor = PdfTextExtractor()

    pdf = create_multi_page_pdf()

    result = extractor.extract(pdf)

    assert len(result) == 2

    assert result[0].page_number == 1
    assert result[0].content.strip() == "Page one text"

    assert result[1].page_number == 2
    assert result[1].content.strip() == "Page two text"


def test_extract_returns_empty_content_for_blank_page():
    buffer = BytesIO()

    pdf = canvas.Canvas(buffer)

    pdf.drawString(100, 750, "Page one text")
    pdf.showPage()

    # Intentionally blank page
    pdf.showPage()

    pdf.save()

    buffer.seek(0)

    extractor = PdfTextExtractor()

    result = extractor.extract(buffer)

    assert len(result) == 2

    assert result[0].page_number == 1
    assert result[0].content.strip() == "Page one text"

    assert result[1].page_number == 2
    assert result[1].content == ""


def test_extract_preserves_multiple_lines():
    buffer = BytesIO()

    pdf = canvas.Canvas(buffer)

    pdf.drawString(100, 750, "First line")
    pdf.drawString(100, 730, "Second line")
    pdf.drawString(100, 710, "Third line")

    pdf.save()

    buffer.seek(0)

    extractor = PdfTextExtractor()

    result = extractor.extract(buffer)

    assert len(result) == 1
    assert result[0].page_number == 1

    content = result[0].content

    assert "First line" in content
    assert "Second line" in content
    assert "Third line" in content


def test_extract_raises_for_invalid_pdf():
    extractor = PdfTextExtractor()

    invalid_pdf = BytesIO(b"this is not a valid PDF")

    with pytest.raises(Exception):
        extractor.extract(invalid_pdf)