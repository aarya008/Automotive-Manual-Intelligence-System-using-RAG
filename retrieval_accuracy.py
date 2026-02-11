import rag_text as rag
embeddings = rag.get_embeddings()

vectordb = rag.build_vectorstore(
    documents=None,
    embeddings=embeddings,
    persist_dir=r"E:\Workspace\Projects\Owners_manual\.venv\RAG_pipeline_text\db"  
)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})


test_questions = [
    {"q": "What fuel should I use?", "section": "Fuel"},
    {"q": "How to check brake fluid?", "section": "Brake"},
    {"q": "What does warning mean?", "section": "Safety"},
    {"q": "How do I open the hood of the car?", "section": "Maintenance"},
    {"q": "What should I do if a warning light turns on?", "section": "Safety"},
    {"q": "How often should periodic maintenance be done?", "section": "Maintenance"},
    {"q": "Where is the fuse box located?", "section": "Electrical"},
    {"q": "How do I adjust the steering wheel position?", "section": "Steering"},
    {"q": "What is the capacity of the fuel tank?", "section": "Fuel"},
    {"q": "What checks should be done before driving the vehicle?", "section": "Inspection"}
]

section_map = {
    "Fuel": ["FUEL RECOMMENDATION"],
    "Maintenance": ["INSPECTION AND MAINTENANCE"],
    "Safety": ["BEFORE DRIVING", "WARNING"],
    "Electrical": ["ELECTRICAL"],
    "Steering": ["STEERING"],
    "Inspection": ["INSPECTION AND MAINTENANCE"],
    "Brake": ["BRAKE"],
}

correct = 0

for item in test_questions:
    for i, item in enumerate(test_questions, 1):
    # Retrieve documents
        docs = retriever.invoke(item["q"])
    
    # Get top result metadata
    top_doc = docs[0]
    retrieved_section = top_doc.metadata.get("section_name", "")

    print("Q:", item["q"])
    print("Expected:", item["section"])
    print("Retrieved:", retrieved_section)
    print()

    if item["section"].lower() in retrieved_section.lower():
        returned_section = retrieved_section
    if retrieved_section in section_map[item["section"]]:
            correct += 1    

accuracy = correct / len(test_questions)*100
print("Retrieval Accuracy:", accuracy, "%")
