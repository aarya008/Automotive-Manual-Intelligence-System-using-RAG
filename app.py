import streamlit as st
import tempfile
import os
import shutil
import rag_text as rag

st.set_page_config(
    page_title="AutoManual AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def clear_temp_db():
    if os.path.exists("/tmp/temp_db"):
        try:
            shutil.rmtree("/tmp/temp_db")
        except Exception:
            pass

def init_session_state():
    defaults = {
        "retriever": None,
        "history": [],
        "chat_messages": [],
        "pdf_name": None,
        "chunk_count": 0,
        "last_retrieved_docs": []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

if "cleanup_registered" not in st.session_state:
    st.session_state.cleanup_registered = True
    try:
        st.on_session_end(clear_temp_db)
    except Exception:
        pass

if st.session_state.retriever is None:
    clear_temp_db()


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    embeddings = rag.get_embeddings()
    llm = rag.get_llm()
    return embeddings, llm


# ─────────────────────────────────────────────
# PDF PROCESSING
# ─────────────────────────────────────────────
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        documents = rag.load_and_split_pdf(tmp_path)
        embeddings, _ = load_models()
        vectordb = rag.build_vectorstore(
            documents=documents,
            embeddings=embeddings,
            persist_dir="/tmp/temp_db",
        )
        return rag.get_retriever(vectordb), len(documents)
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

    st.markdown("## 🚗 Owners Manual Assistant ")
    st.caption("RAG-powered Q&A for your vehicle's manual. Upload your PDF and start asking!")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Owner's Manual (PDF)",
        type=["pdf"],
        help="Must have a clickable Table of Contents"
    )

    if uploaded_file:
        is_new = uploaded_file.name != st.session_state.pdf_name
        if is_new:
            if st.button("⚡ Process Manual", type="primary", use_container_width=True):
                with st.spinner("Parsing PDF and building index…"):
                    try:
                        retriever, count = process_pdf(uploaded_file)
                        st.session_state.retriever = retriever
                        st.session_state.history = []
                        st.session_state.chat_messages = []
                        st.session_state.pdf_name = uploaded_file.name
                        st.session_state.chunk_count = count
                        st.session_state.last_retrieved_docs = []
                        st.rerun()
                    except ValueError as e:
                        st.error(f"PDF Error: {e}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.success(f"✅ {uploaded_file.name}")

    st.divider()

    if st.session_state.pdf_name:
        st.markdown("**Active Manual**")
        st.caption(st.session_state.pdf_name)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", st.session_state.chunk_count)
        with col2:
            st.metric("Messages", len(st.session_state.chat_messages) // 2)

        st.divider()
        if st.button("🗑️ Clear Session", use_container_width=True):
            for key in ["retriever", "history", "chat_messages", "pdf_name", "last_retrieved_docs"]:
                st.session_state[key] = [] if isinstance(st.session_state[key], list) else None
            st.session_state.chunk_count = 0
            clear_temp_db()
            st.rerun()
    else:
        st.info("No manual loaded yet.")


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────

if st.session_state.retriever is None:

    st.markdown("# Ask Your Car Manual")
    st.markdown("##### Get instant, accurate answers from your vehicle's owner manual — powered by RAG.")
    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🔍 Semantic Search")
        st.caption("Understands your question the way you naturally phrase it — not just keyword matching.")
    with c2:
        st.markdown("### 📌 Source Citations")
        st.caption("Every answer cites the exact section and page numbers from the manual.")
    with c3:
        st.markdown("### ⚡ TOC-Aware Chunking")
        st.caption("Respects the document's Table of Contents for precise, context-aware retrieval.")

    st.divider()
    st.markdown("#### 💬 Example questions")

    q1, q2, q3 = st.columns(3)
    with q1:
        st.code("How do I jump start the car?", language=None)
        st.code("What does the ABS warning light mean?", language=None)
    with q2:
        st.code("How often should I change the engine oil?", language=None)
        st.code("How do I engage reverse on a manual gearbox?", language=None)
    with q3:
        st.code("My engine makes noise after cold start — normal?", language=None)
        st.code("What is the correct tyre pressure?", language=None)

else:
    _, llm = load_models()

    st.markdown("### 💬 Chat with your Manual")

    if st.session_state.last_retrieved_docs:
        with st.expander("📎 Sources used in last answer", expanded=False):
            for i, doc in enumerate(st.session_state.last_retrieved_docs):
                m = doc.metadata
                safety = m.get("safety_level", "none")
                cols = st.columns([3, 1, 1, 1])
                cols[0].markdown(f"**{m.get('section_name', 'Unknown')}**")
                cols[1].caption(f"p.{m.get('start_page','?')}–{m.get('end_page','?')}")
                cols[2].caption(m.get("content_type", "—").capitalize())
                if safety != "none":
                    cols[3].warning(safety)
                else:
                    cols[3].caption("—")
                st.divider()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask anything about your vehicle manual…")

    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Searching manual…"):
                try:
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
                    err = f"Something went wrong: {e}"
                    st.error(err)
                    st.session_state.chat_messages.append({"role": "assistant", "content": err})

        st.rerun()