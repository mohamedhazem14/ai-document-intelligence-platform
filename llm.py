"""Single place where the local LLM is created."""
from langchain_ollama import ChatOllama

from config import LLM_MODEL


def get_llm(temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=temperature)
