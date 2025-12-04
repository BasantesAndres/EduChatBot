# src/educhat/rag_store.py

from typing import List
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_DIR = "data/processed/faiss"


def build_vector_store(texts: List[str], persist_dir: str = FAISS_DIR):
    """
    Crea un vector store FAISS a partir de una lista de strings (texts),
    lo guarda en disco y lo devuelve.
    """
    # Modelo de embeddings de HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    # Convertimos cada texto en un Document de LangChain
    docs = [Document(page_content=t) for t in texts if t.strip()]

    # Creamos el índice FAISS en memoria
    vectordb = FAISS.from_documents(docs, embeddings)

    # Guardamos en disco
    os.makedirs(persist_dir, exist_ok=True)
    vectordb.save_local(persist_dir)

    return vectordb


def load_vector_store(persist_dir: str = FAISS_DIR):
    """
    Carga el vector store FAISS desde disco y lo devuelve.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    vectordb = FAISS.load_local(
        persist_dir,
        embeddings,
        allow_dangerous_deserialization=True,  # hace falta en versiones nuevas
    )
    return vectordb
