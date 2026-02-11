import fitz  # PyMuPDF
import re
from typing import List, Dict

WARNING_WORDS = ["WARNING", "CAUTION", "NOTICE", "DANGER"]

LIST_PATTERN = r"^\s*(\d+\.|\-|\•|\*)"


def classify_block(text: str, font_size: float) -> str:
    """
    Classify block type based on content + formatting.
    """

    text_strip = text.strip()

    # Warning detection
    for w in WARNING_WORDS:
        if text_strip.upper().startswith(w):
            return "warning"

    # List detection
    if re.match(LIST_PATTERN, text_strip):
        return "list"

    # Heading detection using font size or uppercase pattern
    if font_size > 12 or text_strip.isupper():
        if len(text_strip.split()) < 10:
            return "heading"

    return "paragraph"


def parse_pdf_to_structured_blocks(pdf_path: str) -> List[Dict]:
    """
    Parses PDF and returns structured blocks.
    """

    doc = fitz.open(pdf_path)
    blocks_output = []

    for page_number, page in enumerate(doc, start=1):

        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" not in block:
                continue

            block_text = ""
            font_sizes = []

            for line in block["lines"]:
                for span in line["spans"]:
                    block_text += span["text"] + " "
                    font_sizes.append(span["size"])

            block_text = block_text.strip()

            if not block_text:
                continue

            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 10

            block_type = classify_block(block_text, avg_font_size)

            blocks_output.append({
                "text": block_text,
                "type": block_type,
                "page": page_number,
                "font_size": round(avg_font_size, 2)
            })

    doc.close()

    return blocks_output
