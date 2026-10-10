"""Suggested questions a reader might want to ask about the documents."""
import re

from langchain_core.prompts import PromptTemplate

from config import QUESTIONS_CHAR_LIMIT
from llm import get_llm

MAX_QUESTIONS = 8

PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""From the document text below, write 5 to 8 insightful, analytical questions
a reader could ask to understand the content in depth. Cover main findings, risks,
numerical comparisons (if any) and conclusions.
Output ONLY the questions, one per line, with no numbering, bullets or extra formatting.

Document text:
{text}""",
)

_LEADING_MARKERS = re.compile(r"^[\s\-*•\d.)]+")


def generate_suggested_questions(full_text: str) -> list[str]:
    text = full_text[:QUESTIONS_CHAR_LIMIT]
    raw = get_llm(temperature=0.5).invoke(PROMPT.format(text=text)).content

    questions = []
    for line in raw.splitlines():
        cleaned = _LEADING_MARKERS.sub("", line).strip()
        if cleaned:
            questions.append(cleaned)
    return questions[:MAX_QUESTIONS]
