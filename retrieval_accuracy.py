import json
import os
import shutil
import rag_text as rag
from langchain_community.vectorstores.utils import filter_complex_metadata
from hybrid_extractor import extract_hybrid_chunks, docs_from_hybrid_chunks
from pdf_parser import parse_pdf_to_structured_blocks

embeddings = rag.get_embeddings()

pdf_path = r"C:\Users\Aarya\Downloads\Swift_Owner's_Manual.pdf"
blocks = parse_pdf_to_structured_blocks(pdf_path)
chunks = extract_hybrid_chunks(pdf_path, blocks)
documents = docs_from_hybrid_chunks(chunks)
documents = filter_complex_metadata(documents)

print(f"Total chunks: {len(documents)}")

sections = set(d.metadata.get("section_name", "") for d in documents)
print(f"Unique sections ({len(sections)}):")
for s in sorted(sections):
    print(f"  - {s}")

persist_dir = r"E:\Workspace\Projects\Owners_manual\.venv\RAG_pipeline_text\db"
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)

vectordb = rag.build_vectorstore(
    documents=documents,
    embeddings=embeddings,
    persist_dir=persist_dir
)

retriever = rag.get_retriever(vectordb, k=3)

with open(r"E:\Workspace\Projects\Owners_manual\.venv\Automotive Manual Intelligence System using RAG\test_questions.txt", "r") as f:
    test_questions = [json.loads(line) for line in f]

print(f"\nLoaded {len(test_questions)} test questions")

correct = 0
failed = []

print("\n" + "="*60)
print("RETRIEVAL TEST RESULTS")
print("="*60)

for i, item in enumerate(test_questions, 1):
    docs = retriever.invoke(item["q"])
    retrieved_section = docs[0].metadata.get("section_name", "")
    
    if item["section"].lower() in retrieved_section.lower() or retrieved_section.lower() in item["section"].lower():
        correct += 1
        print(f"[{i}] [PASS] {item['q'][:50]}...")
        print(f"       -> {retrieved_section}")
    else:
        failed.append((item, retrieved_section))
        print(f"[{i}] [FAIL] {item['q'][:50]}...")
        print(f"       Expected: {item['section']}, Got: {retrieved_section}")
    print()

accuracy = (correct / len(test_questions)) * 100
print("="*60)
print(f"Total Questions: {len(test_questions)}")
print(f"Correct: {correct}")
print(f"Failed: {len(failed)}")
print(f"Accuracy: {accuracy:.2f}%")
print("="*60)
