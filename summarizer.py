"""Executive summary generation."""
from langchain_core.prompts import PromptTemplate

from config import SUMMARY_CHAR_LIMIT
from llm import get_llm

PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are a senior business analyst and document intelligence expert.

Read the document below carefully and write a detailed executive summary.

Guidelines:
- Synthesize and explain the content; do not just list topics or repeat headings.
- Cover every major section of the document in proportion to its weight.
- Capture the main ideas, arguments, findings, technical details and conclusions.
- Avoid generic statements and avoid long bullet lists inside the summary.

Use exactly this structure:

# Executive Summary
8 to 12 detailed paragraphs. This is the longest section (about 70% of the output).

# Key Findings
The most important findings and insights.

# Important Topics
5 to 10 topics at most.

# Recommendations / Action Items
Practical actions or conclusions that follow from the document.

Document:
{text}
""",
)


def generate_summary(full_text: str) -> str:
    text = full_text[:SUMMARY_CHAR_LIMIT]
    return get_llm(temperature=0.2).invoke(PROMPT.format(text=text)).content
