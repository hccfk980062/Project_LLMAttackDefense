"""Knowledge-base document endpoints: upload (with optional L1/L2 pre-ingestion
scan) and listing."""
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from api.schemas.response import ChunkVerdict, DocumentListResponse, DocumentSource, UploadResponse

router = APIRouter(prefix="/documents", tags=["Documents"])

# Deliberately outside data/raw/documents/: that directory feeds
# setup_vector_store.py AND build_detector_dataset.py, so a (possibly poisoned)
# upload there would leak into the L2 training data on the next rebuild.
# Uploaded chunks live in the persisted FAISS index until it is rebuilt.
UPLOAD_DIR = Path("data/uploads")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def _store(request: Request):
    rag = request.app.state.rag_pipeline
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    # The API may wrap the retriever (e.g. PoisoningRetriever); .store falls through.
    return rag.retriever.store


def scan_chunks(defense, chunks) -> list[ChunkVerdict]:
    """Run the L1 structural scan and (when trained) the L2 sliding-window scan
    over each chunk — the same checks the pipeline applies to retrieved docs,
    used here as a gate *before* a document enters the knowledge base."""
    texts = [c.page_content for c in chunks]
    l2_results = defense.l2.scan_documents(texts, threshold=defense.context_threshold) if (defense.l2 and texts) else []
    verdicts = []
    for i, text in enumerate(texts):
        by, reasons = [], []
        l1 = defense.l1.scan_context(text)
        if l1.is_blocked:
            by.append("L1")
            reasons.append(f"{l1.reason}: {l1.triggered_pattern[:60]}")
        if l2_results and l2_results[i].is_injection:
            by.append("L2")
            reasons.append(f"injection_detected: p={l2_results[i].confidence:.3f}")
        verdicts.append(ChunkVerdict(
            index=i, preview=text[:120], flagged=bool(by), flagged_by=by, reasons=reasons,
        ))
    return verdicts


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    scan: bool = Query(True, description="Scan chunks with L1/L2 before ingestion"),
    reject_flagged: bool = Query(True, description="Do not ingest chunks the scan flags"),
):
    """Add a .txt/.md/.pdf/.docx file to the knowledge base (FAISS index)."""
    from src.rag.document_loader import SUPPORTED_EXTENSIONS, chunk_documents, load_file
    from src.rag.vector_store import add_documents

    store = _store(request)
    name = Path(file.filename or "").name
    ext = Path(name).suffix.lower()
    if not name or ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type; expected {SUPPORTED_EXTENSIONS}")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_BYTES} bytes")
    if not data.strip():
        raise HTTPException(status_code=400, detail="Empty file")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved = UPLOAD_DIR / name
    saved.write_bytes(data)

    cfg = request.app.state.config["rag"]
    try:
        docs = await run_in_threadpool(load_file, saved)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")
    chunks = chunk_documents(docs, cfg["chunk_size"], cfg["chunk_overlap"])
    for c in chunks:
        c.metadata["source"] = str(saved)
        c.metadata["uploaded"] = True

    defense = request.app.state.defense_pipeline
    verdicts: list[ChunkVerdict] = []
    scanned = bool(scan and defense is not None)
    if scanned:
        verdicts = await run_in_threadpool(scan_chunks, defense, chunks)

    keep = [c for c, v in zip(chunks, verdicts) if not v.flagged] if (scanned and reject_flagged) else chunks
    if keep:
        await run_in_threadpool(add_documents, store, keep, cfg["vector_store_path"])

    return UploadResponse(
        filename=name,
        saved_path=str(saved),
        n_chunks=len(chunks),
        n_added=len(keep),
        n_rejected=len(chunks) - len(keep),
        scanned=scanned,
        chunks=verdicts,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(request: Request):
    """List the knowledge base's source files and their chunk counts."""
    store = _store(request)
    docs = store.docstore._dict.values()
    counts = Counter(str(d.metadata.get("source", "unknown")) for d in docs)
    return DocumentListResponse(
        total_chunks=sum(counts.values()),
        documents=[DocumentSource(source=s, n_chunks=n) for s, n in sorted(counts.items())],
    )
