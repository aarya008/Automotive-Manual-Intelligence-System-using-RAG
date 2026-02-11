print("Loading RAG module...")
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from pdf_parser import parse_pdf_to_structured_blocks
from hybrid_extractor import extract_hybrid_chunks, docs_from_hybrid_chunks
   
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("Google_api_key")
def get_llm():
    return ChatGoogleGenerativeAI(
        api_key=API_KEY,
        model="gemini-2.5-flash",
        temperature=0.1
    )


def get_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": device}
    )


def load_and_split_pdf(path: str):
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF file not found at {path}")    
    blocks = parse_pdf_to_structured_blocks(path)
    chunks = extract_hybrid_chunks(path, blocks)
    documents = docs_from_hybrid_chunks(chunks)
    print("Documents count:", len(documents))
    return documents

    

   


def build_vectorstore(documents, embeddings, persist_dir: str):
    if os.path.exists(persist_dir):
        print("Loading existing vector DB...")
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    else:
        print("Creating new vector DB...")
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir
        )

    return vectordb
    


def get_retriever(vectordb, k: int = 5):
    return vectordb.as_retriever(search_kwargs={"k": k})


def init_history():
    return []

def update_history(history, user_msg, assistant_msg):
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    return history

def history_to_text(history, limit=4):
    """
    Keep history short and controlled
    """
    trimmed = history[-limit:]
    return "\n".join(
        f"{h['role']}: {h['content']}" for h in trimmed
    )
def retrieve_context(query, retriever):
    docs = retriever.invoke(query)
    return "\n".join(d.page_content for d in docs)

def load_system_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(context, history, question):
    SYSTEM_PROMPT_PATH = r"E:\Workspace\Projects\Owners_manual\.venv\RAG_pipeline_text\system_prompt.txt"

    system_prompt = load_system_prompt(SYSTEM_PROMPT_PATH)

    return f"""
{system_prompt}        

Conversation so far:
{history}

Context:
{context}

Question:
{question}

"""

def generate_answer(llm, prompt):
    return llm.invoke(prompt).content

def run_rag(user_question, retriever, llm, history):

    docs = retriever.invoke(user_question)

    # Format context with metadata (updated for hybrid extractor)
    context_blocks = []
    for d in docs:
        meta = d.metadata

        block = f"""
Section: {meta.get('section_name', '')}
Start_page: {meta.get('start_page', '')}
End_page: {meta.get('end_page', '')}
Vehicle_systems: {meta.get('vehicle_systems', [])}
Content_type: {meta.get('content_type', '')}
Safety_level: {meta.get('safety_level', 'none')}
Has_procedure: {meta.get('has_procedure', False)}

{d.page_content}
"""
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    history_text = history_to_text(history)

    prompt = build_prompt(context, history_text, user_question)

    answer = generate_answer(llm, prompt)

    history = update_history(history, user_question, answer)

    return answer, history

print("RAG module loaded.")



