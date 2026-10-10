"""AI Document Intelligence Platform: Streamlit front end."""
import re
import time

import streamlit as st

from config import LLM_MODEL
from document_loader import SUPPORTED_TYPES, process_uploaded_files
from export_utils import chat_to_pdf, chat_to_text, text_to_pdf
from quiz_generator import generate_quiz
from rag_pipeline import answer_question
from suggested_questions import generate_suggested_questions
from summarizer import generate_summary
from vector_store import clear_vector_store, create_and_save_vector_store

st.set_page_config(
    page_title="AI Document Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

STYLE = """
<style>
.stApp {
    background: linear-gradient(135deg, #0b132b 0%, #1c2541 55%, #3a506b 100%);
    background-attachment: fixed;
}
[data-testid="stSidebar"] { background-color: rgba(20, 20, 30, 0.85); }
h3 { color: #4FC3F7 !important; }

.stTabs [role="tablist"] { display: flex; width: 100%; gap: 20px; }
button[data-baseweb="tab"] {
    flex: 1;
    font-size: 22px !important;
    font-weight: 700 !important;
    min-height: 70px !important;
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 15px !important;
    transition: 0.3s;
}
button[data-baseweb="tab"]:hover { transform: translateY(-2px); }
button[aria-selected="true"] {
    background: rgba(255, 255, 255, 0.15) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

DEFAULT_STATE = {
    "chat_history": [],
    "vector_store": None,
    "full_raw_text": "",
    "summary": None,
    "quiz_data": None,
    "suggested_qs": [],
    "pending_question": "",
    "num_docs": 0,
    "num_chunks": 0,
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_everything() -> None:
    clear_vector_store()
    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = value


def run_generator(label: str, func, state_key: str) -> None:
    """Run a text-generation feature on the processed documents and store its result."""
    if not st.session_state.full_raw_text:
        st.warning("Please process documents first.")
        return
    try:
        with st.spinner(f"Generating {label}..."):
            st.session_state[state_key] = func(st.session_state.full_raw_text)
        st.success(f"✅ {label.capitalize()} generated!")
    except Exception as exc:
        st.error(f"Failed to generate {label}: {exc}")


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("🔍 View Source Citations"):
        for idx, src in enumerate(sources, start=1):
            st.markdown(f"**Source {idx}:** `{src['source']}` (Page {src['page']})")
            st.markdown(f"*Similarity Score:* `{src['similarity_score']}`")
            st.caption(f"_{src['content'][:250]}..._")
            st.divider()


def format_quiz_markdown(quiz: str) -> str:
    """Put each A) B) C) D) option and the answer label on their own lines."""
    quiz = re.sub(r"\s*\b([A-D])\)", r"\n\n\1)", quiz)
    return quiz.replace("Correct Answer:", "\n\n### Correct Answer:\n")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("📂 Document Management")
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, DOCX, TXT, CSV)",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
    )

    if st.button("Process Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            try:
                with st.spinner("Processing and splitting documents..."):
                    chunks = process_uploaded_files(uploaded_files)
                with st.spinner("Generating embeddings (this may take a moment)..."):
                    store = create_and_save_vector_store(chunks)

                st.session_state.vector_store = store
                st.session_state.num_docs = len(uploaded_files)
                st.session_state.num_chunks = len(chunks)
                st.session_state.full_raw_text = "\n\n".join(c.page_content for c in chunks)
                # Results from a previous document set no longer apply.
                st.session_state.summary = None
                st.session_state.quiz_data = None
                st.session_state.suggested_qs = []
                st.session_state.chat_history = []
                st.success("✅ Documents processed successfully!")
            except Exception as exc:
                st.error(f"Error during processing: {exc}")

    st.divider()
    st.header("📊 Executive Summary")
    if st.button("Generate Summary", use_container_width=True):
        run_generator("summary", generate_summary, "summary")

    st.divider()
    st.header("📝 Quiz Generator")
    if st.button("Generate Quiz", use_container_width=True):
        run_generator("quiz", generate_quiz, "quiz_data")

    st.divider()
    st.header("💡 Suggested Questions")
    if st.button("Generate Suggested Questions", use_container_width=True):
        run_generator("suggested questions", generate_suggested_questions, "suggested_qs")

    st.divider()
    st.header("⚙️ System Status")
    if st.session_state.vector_store is not None:
        st.success("Database Status: Loaded 🟢")
        if st.button("Clear Database", use_container_width=True):
            reset_everything()
            st.rerun()
    else:
        st.error("Database Status: Empty 🔴")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    has_history = bool(st.session_state.chat_history)
    col_txt, col_pdf = st.columns(2)
    with col_txt:
        st.download_button(
            "📥 Export TXT",
            chat_to_text(st.session_state.chat_history) if has_history else "",
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True,
            disabled=not has_history,
        )
    with col_pdf:
        st.download_button(
            "📄 Export PDF",
            chat_to_pdf(st.session_state.chat_history) if has_history else b"",
            file_name="chat_history.pdf",
            mime="application/pdf",
            use_container_width=True,
            disabled=not has_history,
        )

# --------------------------------------------------------------- main page
st.title("🧠 AI Document Intelligence Platform")
st.markdown(
    """
### Powered by RAG + FAISS + Mistral

Upload multiple documents, generate summaries, create quizzes,
and chat with your data using Retrieval-Augmented Generation.
"""
)
st.divider()

m1, m2, m3 = st.columns(3)
m1.metric("Documents", st.session_state.num_docs)
m2.metric("Chunks", st.session_state.num_chunks)
m3.metric("Model", LLM_MODEL.capitalize())

tab_chat, tab_summary, tab_quiz = st.tabs(
    ["💬 Chat & QA", "📊 Executive Summary", "🎓 Quiz Results"]
)

with tab_chat:
    if st.session_state.suggested_qs:
        st.subheader("💡 Suggested Questions")
        cols = st.columns(min(len(st.session_state.suggested_qs), 3))
        for i, question in enumerate(st.session_state.suggested_qs):
            if cols[i % len(cols)].button(question, key=f"sq_{i}"):
                st.session_state.pending_question = question
        st.divider()

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            render_sources(chat["sources"])
            st.caption(f"⏱ Response Time: {chat['response_time']:.2f} sec")

with tab_summary:
    if st.session_state.summary:
        with st.container(border=True):
            st.subheader("📊 Executive Summary")
            st.markdown(st.session_state.summary)
        st.download_button(
            "📕 Download Summary PDF",
            text_to_pdf("Executive Summary", st.session_state.summary),
            file_name="Executive_Summary.pdf",
            mime="application/pdf",
        )
    else:
        st.info("Process your documents, then click 'Generate Summary' in the sidebar.")

with tab_quiz:
    if st.session_state.quiz_data:
        st.markdown(format_quiz_markdown(st.session_state.quiz_data))
        st.download_button(
            "📕 Download Quiz PDF",
            text_to_pdf("Generated Quiz", st.session_state.quiz_data),
            file_name="Document_Quiz.pdf",
            mime="application/pdf",
        )
    else:
        st.info("Click 'Generate Quiz' in the sidebar to create a test from your documents.")

# The chat box sits at the top level so Streamlit pins it to the bottom of the page.
typed = st.chat_input("Ask a question about your documents...")
query = typed or st.session_state.pending_question
st.session_state.pending_question = ""

if query:
    if st.session_state.vector_store is None:
        st.warning("Please upload and process documents before asking questions.")
    else:
        with tab_chat:
            with st.chat_message("user"):
                st.write(query)
            with st.chat_message("assistant"):
                with st.spinner("Searching documents and formulating answer..."):
                    try:
                        started = time.time()
                        answer, sources = answer_question(
                            query,
                            st.session_state.vector_store,
                            history=st.session_state.chat_history,
                        )
                        st.session_state.chat_history.append(
                            {
                                "question": query,
                                "answer": answer,
                                "sources": sources,
                                "response_time": time.time() - started,
                            }
                        )
                    except Exception as exc:
                        st.error(f"Error while answering (is Ollama running?): {exc}")
                        st.stop()
        st.rerun()
