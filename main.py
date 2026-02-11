import rag_text as rag

PERSIST_DIR = "db"
PDF_PATH = "C:\\Users\\Aarya\\Downloads\\Swift_Owner's_Manual.pdf"

# --- Build index ONCE ---
embeddings = rag.get_embeddings()
chunks = rag.load_and_split_pdf(PDF_PATH)

vectordb = rag.build_vectorstore(chunks, embeddings, PERSIST_DIR)

retriever = rag.get_retriever(vectordb)

llm = rag.get_llm()

history = rag.init_history()

# --- Chat loop ---
while True:
    query = input("Ask a question (or 'exit'): ")
    if query.lower() == "exit":
        break

    answer, history = rag.run_rag(
        user_question=query,
        retriever=retriever,
        llm=llm,
        history=history
    )

    print("\nAnswer:", answer)
