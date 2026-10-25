"""Document retrieval from FAISS vector store."""
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


class Retriever:
    def __init__(self, store: FAISS, top_k: int = 5):
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve top-k relevant documents for a query."""
        return self.store.similarity_search(query, k=self.top_k)

    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """Retrieve documents with similarity scores."""
        return self.store.similarity_search_with_score(query, k=self.top_k)

    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into a context string."""
        parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", f"Document {i}")
            parts.append(f"[來源 {i}: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)
