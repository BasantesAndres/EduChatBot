# src/educhat/tools.py

from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from .rag_store import load_vector_store


@lru_cache(maxsize=1)
def _get_retriever():
    """
    Carga el vector store y devuelve un retriever de similitud.
    Se cachea en memoria para no recargarlo cada vez.
    """
    vectordb = load_vector_store()
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    return retriever


def course_rag_search(query: str) -> str:
    """
    Busca en los documentos del curso (sílabo, UC1, quizzes, etc.)
    usando el vector store, y devuelve un contexto concatenado en texto.
    Compatible con las versiones nuevas de LangChain donde el retriever
    es un Runnable (se usa .invoke en lugar de .get_relevant_documents).
    """
    retriever = _get_retriever()

    # En LangChain >= 0.2 los retrievers son Runnables:
    # docs = retriever.invoke(query) devuelve List[Document]
    docs = retriever.invoke(query)

    # Por seguridad, normalizamos a lista
    if isinstance(docs, Document):
        docs = [docs]
    elif not isinstance(docs, list):
        docs = list(docs)

    if not docs:
        return ""

    chunks: List[str] = [d.page_content for d in docs if getattr(d, "page_content", None)]
    context = "\n\n---\n\n".join(chunks)
    return context
