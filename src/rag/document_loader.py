"""Load and chunk documents for the RAG knowledge base."""
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)


LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
}
# TextLoader otherwise decodes with the platform default (cp950 on zh-TW
# Windows), and DirectoryLoader(silent_errors=True) would silently skip every
# UTF-8 Chinese file that fails to decode.
LOADER_KWARGS = {
    ".txt": {"encoding": "utf-8"},
    ".md": {"encoding": "utf-8"},
}
SUPPORTED_EXTENSIONS = tuple(LOADER_MAP)


def load_file(path: str | Path) -> List[Document]:
    """Load a single supported file (used by the API upload endpoint)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in LOADER_MAP:
        raise ValueError(f"Unsupported file type {ext!r}; expected one of {SUPPORTED_EXTENSIONS}")
    return LOADER_MAP[ext](str(path), **LOADER_KWARGS.get(ext, {})).load()


def load_documents(docs_dir: str | Path, glob: str = "**/*") -> List[Document]:
    """Load all supported documents from a directory."""
    docs_dir = Path(docs_dir)
    documents: List[Document] = []

    for ext, loader_cls in LOADER_MAP.items():
        loader = DirectoryLoader(
            str(docs_dir),
            glob=f"**/*{ext}",
            loader_cls=loader_cls,
            loader_kwargs=LOADER_KWARGS.get(ext),
            silent_errors=True,
        )
        documents.extend(loader.load())

    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    return splitter.split_documents(documents)


def load_and_chunk(
    docs_dir: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Document]:
    docs = load_documents(docs_dir)
    return chunk_documents(docs, chunk_size, chunk_overlap)
