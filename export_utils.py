"""Helpers that turn chat history, summaries and quizzes into downloadable files."""
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def _safe(text: str) -> str:
    """Escape XML characters (reportlab parses paragraphs as markup) and keep line breaks."""
    return escape(str(text)).replace("\n", "<br/>")


def chat_to_text(chat_history: list[dict]) -> str:
    blocks = [
        f"Question:\n{item['question']}\n\nAnswer:\n{item['answer']}\n\n{'=' * 60}\n"
        for item in chat_history
    ]
    return "\n".join(blocks)


def chat_to_pdf(chat_history: list[dict]) -> BytesIO:
    styles = getSampleStyleSheet()
    story = []
    for item in chat_history:
        story.append(Paragraph(f"<b>Question:</b> {_safe(item['question'])}", styles["Normal"]))
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"<b>Answer:</b> {_safe(item['answer'])}", styles["Normal"]))
        story.append(Spacer(1, 20))
    return _build(story)


def text_to_pdf(title: str, body: str) -> BytesIO:
    styles = getSampleStyleSheet()
    story = [
        Paragraph(_safe(title), styles["Title"]),
        Spacer(1, 20),
        Paragraph(_safe(body), styles["BodyText"]),
    ]
    return _build(story)


def _build(story) -> BytesIO:
    buffer = BytesIO()
    SimpleDocTemplate(buffer).build(story)
    buffer.seek(0)
    return buffer
