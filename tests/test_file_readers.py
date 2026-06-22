import io
import unittest

from pypdf import PdfWriter

from streamlit_app import _extract_pdf_text, _read_uploaded_file


class _Upload:
    def __init__(self, name: str, data: bytes, content_type: str = "") -> None:
        self.name = name
        self._data = data
        self.type = content_type

    def getvalue(self) -> bytes:
        return self._data


def _minimal_text_pdf(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("ascii")
    stream = b"BT /F1 12 Tf 72 720 Td (" + escaped + b") Tj ET"
    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        b"5 0 obj << /Length " + str(len(stream)).encode("ascii") + b" >> stream\n" + stream + b"\nendstream endobj\n",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(out.tell())
        out.write(obj)
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.write(f"{offset:010d} 00000 n \n".encode("ascii"))
    out.write(f"trailer << /Root 1 0 R /Size {len(objects) + 1} >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    return out.getvalue()


class FileReaderTest(unittest.TestCase):
    def test_reads_text_upload(self) -> None:
        text, metadata = _read_uploaded_file(_Upload("policy.md", b"# Policy\nUse GraphRAG.", "text/markdown"))

        self.assertIn("Use GraphRAG", text)
        self.assertEqual(metadata["source_format"], "md")
        self.assertEqual(metadata["parser"], "utf-8")

    def test_extracts_text_layer_from_pdf(self) -> None:
        text, metadata = _extract_pdf_text(_minimal_text_pdf("Hello PDF Text"), "sample.pdf")

        self.assertIn("[Page 1]", text)
        self.assertIn("Hello PDF Text", text)
        self.assertEqual(metadata["source_format"], "pdf")
        self.assertEqual(metadata["pdf_pages"], 1)
        self.assertEqual(metadata["pdf_pages_with_text"], 1)

    def test_rejects_pdf_without_text_layer(self) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        out = io.BytesIO()
        writer.write(out)

        with self.assertRaisesRegex(ValueError, "No extractable text"):
            _extract_pdf_text(out.getvalue(), "blank.pdf")


if __name__ == "__main__":
    unittest.main()
