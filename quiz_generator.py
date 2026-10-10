"""Automatic quiz generation: 10 multiple choice, 5 true/false, 5 short answer."""
from langchain_core.prompts import PromptTemplate

from config import QUIZ_CHAR_LIMIT
from llm import get_llm

PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are an expert educator.

Write a quiz based STRICTLY on the document below. Do not add facts that are not in it.
Follow the format below exactly, and output clean Markdown only.

# Multiple Choice Questions (10 Questions)

Question 1:
[question]

A) option

B) option

C) option

D) option

Correct Answer:
[letter]

--------------------------------------------------

(repeat for all 10 questions)

# True / False Questions (5 Questions)

Question 1:
[statement]

Correct Answer:
[True or False]

--------------------------------------------------

(repeat for all 5 questions)

# Short Answer Questions (5 Questions)

Question 1:
[question]

Correct Answer:
[short answer]

--------------------------------------------------

(repeat for all 5 questions)

Do not summarize the document, and do not add notes, outlines or study material.

Document text:
{text}
""",
)


def generate_quiz(full_text: str) -> str:
    text = full_text[:QUIZ_CHAR_LIMIT]
    return get_llm(temperature=0.3).invoke(PROMPT.format(text=text)).content
