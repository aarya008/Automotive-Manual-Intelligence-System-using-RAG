import fitz
from typing import List, Dict
from langchain_core.documents import Document
from pdf_parser import parse_pdf_to_structured_blocks
import re

# -------- Vehicle System Keywords -------- #

VEHICLE_SYSTEM_KEYWORDS = {
    "engine": ["engine", "coolant", "oil", "radiator", "motor"],
    "brake": ["brake", "parking brake", "disc", "pad", "abs"],
    "fuel": ["fuel", "petrol", "diesel", "tank", "gas"],
    "steering": ["steering", "wheel alignment", "power steering"],
    "electrical": ["battery", "wiring", "horn", "light", "fuse", "alternator"],
    "transmission": ["transmission", "clutch", "gear", "gearbox"],
    "suspension": ["suspension", "shock", "strut", "spring"],
    "safety": ["airbag", "seat belt", "warning light", "crash"],
    "climate": ["air conditioning", "heating", "hvac", "climate"],
    "tire": ["tire", "tyre", "wheel", "pressure"],
}

SAFETY_WORDS = ["warning", "caution", "notice", "danger"]
PROCEDURE_PATTERN = r"^\s*(\d+\.|\-|\•)"


# -------- Metadata Detection Functions -------- #

def detect_vehicle_system(text: str) -> List[str]:
    """Detect all vehicle systems mentioned in text."""
    text_lower = text.lower()
    systems = set()
    
    for system, keywords in VEHICLE_SYSTEM_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                systems.add(system)
    
    return list(systems) if systems else ["general"]


def detect_safety_level(text: str) -> str:
    """Detect highest safety level in text."""
    text_lower = text.lower()
    priority = {"danger": 4, "warning": 3, "caution": 2, "notice": 1}
    
    found_level = "none"
    max_priority = 0
    
    for word in SAFETY_WORDS:
        if word in text_lower and priority.get(word, 0) > max_priority:
            found_level = word.upper()
            max_priority = priority[word]
    
    return found_level


def detect_content_type(text: str) -> str:
    """Detect the type of content."""
    text_lower = text.lower()
    
    if "maintenance" in text_lower or "inspection" in text_lower or "service" in text_lower:
        return "maintenance"
    if "how to" in text_lower or "procedure" in text_lower or re.search(r"\d+\.\s", text):
        return "procedure"
    if "emergency" in text_lower or "breakdown" in text_lower:
        return "emergency"
    if "specification" in text_lower or "capacity" in text_lower:
        return "specification"
    if "troubleshooting" in text_lower or "problem" in text_lower:
        return "troubleshooting"
    
    return "information"


def has_procedure_steps(text: str) -> bool:
    """Check if text contains numbered or bulleted steps."""
    lines = text.split('\n')
    step_count = sum(1 for line in lines if re.match(PROCEDURE_PATTERN, line.strip()))
    return step_count >= 2


# -------- TOC Extraction -------- #

def extract_toc_with_ranges(pdf_path: str):
    """Extract TOC with page ranges."""
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    total_pages = len(doc)
    doc.close()

    if not toc:
        raise ValueError("No clickable TOC found in PDF. Please use section_extractor.py instead.")
    
    sections = []
    for i, (level, title, page) in enumerate(toc):
        # Calculate end page
        if i + 1 < len(toc):
            end_page = toc[i + 1][2] - 1
        else:
            end_page = total_pages
        
        sections.append({
            "title": title.strip(),
            "level": level,
            "start_page": page,
            "end_page": end_page
        })
    
    return sections


def extract_text_for_section(blocks, start_page, end_page):
    """Extract all text from blocks within page range."""
    text_parts = []
    for block in blocks:
        page = block.get("page", 0)
        if start_page <= page <= end_page:
            text_parts.append(block["text"])
    
    return "\n".join(text_parts)


def extract_section_name(content_text):
    """Extract actual section name from content."""
    lines = content_text.split('\n')
    
    for line in lines[:20]:
        line = line.strip()
        
        if not line or line.isdigit() or len(line) < 3:
            continue
        
        if any(char.isdigit() for char in line) and len(line) < 15:
            continue
        
        if line.isupper() and len(line) > 3:
            return ' '.join(line.split())
    
    return None


# -------- Chunking -------- #

def split_into_chunks(text: str, max_words: int = 800) -> List[str]:
    """Split text into chunks at paragraph boundaries."""
    if len(text.split()) <= max_words:
        return [text]
    
    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if current_chunk and len((current_chunk + para).split()) > max_words:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


# -------- Main Function -------- #

def extract_hybrid_chunks(pdf_path: str, blocks: list, chunk_large_sections: bool = True, max_chunk_size: int = 800):
    """
    Extract chunks from car manual using TOC structure + semantic metadata.
    
    Args:
        pdf_path: Path to PDF file
        blocks: List of structured blocks from PDF
        chunk_large_sections: Split large sections into smaller chunks
        max_chunk_size: Max words per chunk (if chunking enabled)
    
    Returns:
        List of chunk dictionaries with metadata
    """
    
    # Parse PDF into blocks
    
    
    # Extract TOC with page ranges
    toc_sections = extract_toc_with_ranges(pdf_path)
    
    chunks = []
    
    for section in toc_sections:
        # Extract text for this section
        section_text = extract_text_for_section(blocks, section["start_page"], section["end_page"])
        
        if not section_text.strip():
            continue
        
        # Get clean section name
        extracted_name = extract_section_name(section_text)
        section_name = extracted_name if extracted_name else section["title"]
        
        # Split into chunks if needed
        if chunk_large_sections:
            text_chunks = split_into_chunks(section_text, max_chunk_size)
        else:
            text_chunks = [section_text]
        
        # Create chunks with metadata
        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunks.append({
                "content": chunk_text,
                "section_name": section_name,
                "section_level": section["level"],
                "start_page": section["start_page"],
                "end_page": section["end_page"],
                "chunk_index": chunk_idx,
                "total_chunks_in_section": len(text_chunks),
                # Semantic metadata
                "vehicle_systems": detect_vehicle_system(chunk_text),
                "content_type": detect_content_type(chunk_text),
                "safety_level": detect_safety_level(chunk_text),
                "has_procedure": has_procedure_steps(chunk_text)
            })
    
    return chunks


# -------- Convert to LangChain Documents -------- #

def docs_from_hybrid_chunks(chunks: List[Dict]) -> List[Document]:
    """Convert chunks to LangChain Documents."""
    documents = []
    
    for chunk in chunks:
        documents.append(
            Document(
                page_content=chunk["content"],
                metadata={
                    "section_name": chunk["section_name"],
                    "section_level": chunk["section_level"],
                    "start_page": chunk["start_page"],
                    "end_page": chunk["end_page"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks_in_section"],
                    "vehicle_systems": chunk["vehicle_systems"],
                    "content_type": chunk["content_type"],
                    "safety_level": chunk["safety_level"],
                    "has_procedure": chunk["has_procedure"],
                    "is_safety_critical": chunk["safety_level"] != "none",
                    "is_procedural": chunk["has_procedure"] or chunk["content_type"] in ["procedure", "maintenance"],
                }
            )
        )
    
    return documents


