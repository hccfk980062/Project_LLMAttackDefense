"""Initialize FAISS vector store from documents directory."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.document_loader import load_and_chunk
from src.rag.vector_store import build_vector_store
from src.utils import get_logger, load_config

logger = get_logger("setup_vector_store")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS vector store")
    parser.add_argument("--docs-dir", default="data/raw/documents", help="Documents directory")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rag_cfg = cfg["rag"]

    logger.info(f"Loading documents from: {args.docs_dir}")
    chunks = load_and_chunk(
        args.docs_dir,
        chunk_size=rag_cfg["chunk_size"],
        chunk_overlap=rag_cfg["chunk_overlap"],
    )
    logger.info(f"Loaded {len(chunks)} chunks.")

    logger.info("Building FAISS index...")
    build_vector_store(
        chunks,
        embedding_model=rag_cfg["embedding_model"],
        save_path=rag_cfg["vector_store_path"],
    )
    logger.info(f"Vector store saved to: {rag_cfg['vector_store_path']}")


if __name__ == "__main__":
    main()
