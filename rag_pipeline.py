"""Retrieval-Augmented Generation: search FAISS, then answer from the hits only."""
from langchain_core.prompts import PromptTemplate

from config import NO_ANSWER_MESSAGE, TOP_K
from llm import get_llm

PROMPT = PromptTemplate(
    input_variables=["history", "context", "question"],
    template="""You are an expert document assistant. Answer the question using ONLY the context below.
If the context does not contain enough information, reply exactly with:
"{no_answer}"
Never use outside knowledge and never invent facts.

Previous conversation (for follow-up questions only):
{history}

Context:
{context}

Question:
{question}

Answer:""",
    partial_variables={"no_answer": NO_ANSWER_MESSAGE},
)


def _format_history(history: list[dict] | None, turns: int = 3) -> str:
    if not history:
        return "(none)"
    return "\n".join(
        f"User: {item['question']}\nAssistant: {item['answer']}"
        for item in history[-turns:]
    )


def answer_question(question: str, vector_store, history=None, top_k: int = TOP_K):
    """Return (answer, sources). Each source carries a 0-1 similarity score."""
    # Search with the bare question so earlier chat turns don't dilute retrieval.
    results = vector_store.similarity_search_with_score(question, k=top_k)
    if not results:
        return NO_ANSWER_MESSAGE, []

    context_parts = []
    sources = []
    for doc, distance in results:
        meta = doc.metadata
        context_parts.append(
            f"---\nDocument: {meta.get('source')}, Page: {meta.get('page')}\n"
            f"Content: {doc.page_content}"
        )
        sources.append(
            {
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page", "N/A"),
                "chunk_id": meta.get("chunk_id", "N/A"),
                # FAISS returns L2 distance: smaller is closer, so invert it.
                "similarity_score": round(1 / (1 + float(distance)), 4),
                "content": doc.page_content,
            }
        )

    prompt = PROMPT.format(
        history=_format_history(history),
        context="\n\n".join(context_parts),
        question=question,
    )
    response = get_llm(temperature=0.0).invoke(prompt)
    return response.content, sources
