import streamlit as st
import tempfile
import os
import shutil
import rag_text as rag

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Automotive Manual Assistant",
    page_icon="🚗",
    layout="wide"
)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
def clear_temp_db():
    PERSIST_DIR = "/tmp/temp_db"
    if os.path.exists(PERSIST_DIR):
        try:
            shutil.rmtree(PERSIST_DIR)
            st.write("✅ Cleared persistent vector store from temp_db")
        except Exception as e:
            st.warning(f"Could not remove temp_db: {e}")


def init_session_state():
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0
    if "last_retrieved_docs" not in st.session_state:
        st.session_state.last_retrieved_docs = []


init_session_state()

# register cleanup callback to clear built DB at session end
if "session_cleanup_registered" not in st.session_state:
    st.session_state.session_cleanup_registered = True
    try:
        st.on_session_end(clear_temp_db)
    except Exception:
        # Fallback: degrade gracefully if Streamlit version doesn't support on_session_end
        pass

# if no session state data, ensure stale DB is removed on app start
if st.session_state.retriever is None:
    clear_temp_db()

# ─────────────────────────────────────────────
# LOAD MODELS ONCE (cached across reruns)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load embeddings and LLM once. Cached by Streamlit."""
    embeddings = rag.get_embeddings()
    llm = rag.get_llm()
    return embeddings, llm

# ─────────────────────────────────────────────
# PDF PROCESSING
# ─────────────────────────────────────────────
def process_pdf(uploaded_file):
    """Save uploaded file to temp, build vector DB, return retriever."""
    PERSIST_DIR = "/tmp/temp_db"

    # Write uploaded bytes to a real temp file (fitz needs a file path)
    suffix = ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        documents = rag.load_and_split_pdf(tmp_path)
        embeddings, _ = load_models()
        vectordb = rag.build_vectorstore(
            documents=documents,
            embeddings=embeddings,
            persist_dir=PERSIST_DIR,
            
        )
        retriever = rag.get_retriever(vectordb)
        return retriever, len(documents)
    finally:
        os.unlink(tmp_path)  # always clean up temp file

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🚗 Manual Assistant")
    st.markdown("---")

    # PDF uploader
    uploaded_file = st.file_uploader(
        "Upload your vehicle owner's manual",
        type=["pdf"],
        help="PDF with a clickable Table of Contents works best"
    )

    if uploaded_file is not None:
        # Show upload button only if it's a new file
        is_new_file = uploaded_file.name != st.session_state.pdf_name

        if is_new_file:
            if st.button("⚡ Process Manual", type="primary", use_container_width=True):
                with st.spinner("Parsing PDF and building index... (~30 sec for large manuals)"):
                    try:
                        retriever, chunk_count = process_pdf(uploaded_file)

                        # Update session state
                        st.session_state.retriever = retriever
                        st.session_state.history = []
                        st.session_state.chat_messages = []
                        st.session_state.pdf_name = uploaded_file.name
                        st.session_state.chunk_count = chunk_count
                        st.session_state.last_retrieved_docs = []

                        st.success(f"✅ Ready! Indexed {chunk_count} chunks.")
                        st.rerun()

                    except ValueError as e:
                        st.error(f"❌ PDF Error: {e}\n\nMake sure your PDF has a clickable Table of Contents.")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {e}")
        else:
            st.success(f"✅ **{uploaded_file.name}** is loaded")

    st.markdown("---")

    # Current file info
    if st.session_state.pdf_name:
        st.markdown("**📄 Current Manual**")
        st.caption(st.session_state.pdf_name)
        st.caption(f"Chunks indexed: {st.session_state.chunk_count}")

        if st.button("🗑️ Clear & Upload New", use_container_width=True):
            st.session_state.retriever = None
            st.session_state.history = []
            st.session_state.chat_messages = []
            st.session_state.pdf_name = None
            st.session_state.chunk_count = 0
            st.session_state.last_retrieved_docs = []
            # Clean up DB folder
            if os.path.exists("./temp_db"):
                shutil.rmtree("./temp_db")
            st.rerun()
    else:
        st.info("No manual loaded yet.")

    st.markdown("---")
    st.caption("Powered by Gemini 2.5 Flash + HuggingFace Embeddings")

# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
st.title("🚗 Automotive Manual Intelligence")

if st.session_state.retriever is None:
    # Empty state
    st.markdown("### 👈 Upload a PDF manual from the sidebar to get started")
    st.markdown("""
    **What this app does:**
    - Parses your vehicle owner's manual using intelligent TOC-based chunking
    - Builds a semantic search index over all sections
    - Answers your questions with source citations from the actual manual

    **Tips for best results:**
    - Use the official PDF from the manufacturer's website
    - Your PDF must have a clickable Table of Contents
    - Ask specific questions: *"How do I reset the tire pressure warning?"*
    """)

else:
    # Load models (cached, instant after first load)
    _, llm = load_models()

    # ── Chat history display ──
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Sources expander for last answer ──
    if st.session_state.last_retrieved_docs:
        with st.expander("📄 Sources from last answer", expanded=False):
            for i, doc in enumerate(st.session_state.last_retrieved_docs):
                meta = doc.metadata
                safety = meta.get("safety_level", "none")
                safety_badge = f"⚠️ {safety}" if safety != "none" else ""
                st.markdown(
                    f"**[{i+1}] {meta.get('section_name', 'Unknown')}** "
                    f"— Pages {meta.get('start_page', '?')}–{meta.get('end_page', '?')} "
                    f"{safety_badge}"
                )
                st.caption(f"Systems: {meta.get('vehicle_systems', '')} | "
                           f"Type: {meta.get('content_type', '?')}")
                st.markdown("---")

    # ── Chat input ──
    user_query = st.chat_input("Ask anything about your vehicle manual...")

    if user_query:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_messages.append({"role": "user", "content": user_query})

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching manual..."):
                try:
                    # Get retrieved docs separately so we can show sources
                    docs = st.session_state.retriever.invoke(user_query)
                    st.session_state.last_retrieved_docs = docs

                    answer, updated_history = rag.run_rag(
                        user_question=user_query,
                        retriever=st.session_state.retriever,
                        llm=llm,
                        history=st.session_state.history
                    )

                    st.markdown(answer)
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                    st.session_state.history = updated_history

                except Exception as e:
                    error_msg = f"Something went wrong: {e}"
                    st.error(error_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})

        st.rerun()
