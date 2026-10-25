"""Full RAG pipeline: retrieve → compose prompt → generate."""
from dataclasses import dataclass, field
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from .retriever import Retriever
from .vector_store import load_vector_store


SYSTEM_PROMPT = """你是一個知識問答助理。請根據以下提供的參考資料，回答使用者的問題。
如果參考資料中沒有相關資訊，請誠實說明你不知道，不要捏造答案。
請只使用參考資料中的內容進行回答。"""


@dataclass
class RAGResponse:
    query: str
    answer: str
    retrieved_docs: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    sources: List[str] = field(default_factory=list)


class RAGPipeline:
    def __init__(
        self,
        store_path: str,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        top_k: int = 5,
        embedding_model: str = "text-embedding-3-small",
    ):
        store = load_vector_store(store_path, embedding_model)
        self.retriever = Retriever(store, top_k=top_k)
        self.llm = self._build_llm(llm_provider, llm_model, temperature)

    def _build_llm(self, provider: str, model: str, temperature: float):
        if provider == "anthropic":
            return ChatAnthropic(model=model, temperature=temperature)
        return ChatOpenAI(model=model, temperature=temperature)

    def build_prompt(self, query: str, context: str) -> list:
        return [
            ("system", SYSTEM_PROMPT),
            ("human", f"參考資料：\n\n{context}\n\n問題：{query}"),
        ]

    def query(self, user_input: str) -> RAGResponse:
        import time

        start = time.perf_counter()

        docs = self.retriever.retrieve(user_input)
        context = self.retriever.format_context(docs)
        messages = self.build_prompt(user_input, context)
        response = self.llm.invoke(messages)

        latency_ms = (time.perf_counter() - start) * 1000

        return RAGResponse(
            query=user_input,
            answer=response.content,
            retrieved_docs=[d.page_content for d in docs],
            latency_ms=latency_ms,
            sources=[str(d.metadata.get("source", "")) for d in docs],
        )
