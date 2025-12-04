# src/educhat/build_rag.py

from .data_loader import load_txt_files
from .rag_store import build_vector_store

def main():
    texts = load_txt_files("data/raw")
    print(f"Loaded {len(texts)} documents from data/raw")
    build_vector_store(texts)
    print("✅ Vector store built in data/processed/chroma")

if __name__ == "__main__":
    main()
