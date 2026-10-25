"""FAISS vector store management."""
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def build_vector_store(
    documents: List[Document],
    embedding_model: str = "text-embedding-3-small",
    save_path: str | Path | None = None,
) -> FAISS:
    """Build FAISS index from documents and optionally persist it."""
    embeddings = OpenAIEmbeddings(model=embedding_model)
    store = FAISS.from_documents(documents, embeddings)

    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        store.save_local(str(save_path))

    return store


def load_vector_store(
    store_path: str | Path,
    embedding_model: str = "text-embedding-3-small",
) -> FAISS:
    """Load a persisted FAISS index."""
    embeddings = OpenAIEmbeddings(model=embedding_model)
    return FAISS.load_local(
        str(store_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def add_documents(
    store: FAISS,
    documents: List[Document],
    save_path: str | Path | None = None,
) -> FAISS:
    """Add new documents to an existing store."""
    store.add_documents(documents)
    if save_path:
        store.save_local(str(save_path))
    return store
