import hashlib
import os

import chromadb
import pdfplumber
import docx
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")


def extract_text_from_pdf(file_path):
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                pages.append((i, text))
    return pages


def extract_text_from_docx(file_path):
    document = docx.Document(file_path)
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    pages = []
    group_size = 3
    for i in range(0, len(paragraphs), group_size):
        group = "\n".join(paragraphs[i:i + group_size])
        pages.append((i // group_size + 1, group))
    return pages


def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    pages = []
    chunk_size = 800
    for i, start in enumerate(range(0, len(content), chunk_size), start=1):
        pages.append((i, content[start:start + chunk_size]))
    return pages


def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def process_file(file_path, filename=None, collection_name="pdf_docs"):
    """
    Ingests a file into ChromaDB without resetting the collection.
    Supports multi-file — each file's chunks are tagged with its filename.
    """
    if filename is None:
        filename = os.path.basename(file_path)

    collection = chroma_client.get_or_create_collection(name=collection_name)
    pages = extract_text(file_path)
    all_chunks, all_metadatas, all_ids = [], [], []

    for page_num, page_text in pages:
        for chunk in chunk_text(page_text):
            chunk_id = hashlib.md5(
                f"{filename}-{page_num}-{chunk[:50]}".encode()
            ).hexdigest()
            all_chunks.append(chunk)
            all_metadatas.append({"source": filename, "page": page_num})
            all_ids.append(chunk_id)

    if not all_chunks:
        return 0

    embeddings = embedding_model.encode(all_chunks).tolist()
    collection.upsert(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=all_metadatas,
    )
    return len(all_chunks)


def remove_file_chunks(filename, collection_name="pdf_docs"):
    """Removes all chunks belonging to a specific file from ChromaDB.

    Filters in plain Python rather than via ChromaDB's `where=` argument —
    same reasoning as retrieve_chunks()/_sample_chunks() in rag.py and
    insights.py: `where=` was found to unreliably come back empty for
    chunks that demonstrably existed in the collection. Using it here too
    meant "Remove" could report success while leaving the file's chunks
    in place, so they'd keep showing up in later answers even after being
    deleted from the sidebar.
    """
    try:
        collection = chroma_client.get_or_create_collection(name=collection_name)
        got = collection.get(include=["metadatas"])
        ids = got.get("ids") or []
        metas = got.get("metadatas") or []
        matching_ids = [
            _id for _id, meta in zip(ids, metas) if meta.get("source") == filename
        ]
        if matching_ids:
            collection.delete(ids=matching_ids)
        return True
    except Exception:
        return False


def reset_collection(collection_name="pdf_docs"):
    """Wipes the entire collection — used for Clear All.

    Belt-and-suspenders: some ChromaDB persistent-client setups can silently
    no-op delete_collection() (e.g. a stale in-process handle to the old
    collection segment), which previously was masked by a bare
    `except Exception: pass` — so a failed wipe looked identical to a
    successful one, and old chunks kept answering queries after "Clear all
    documents". We now also explicitly delete every id inside the collection
    before dropping it, and end by recreating a guaranteed-empty collection
    so callers can never observe a half-cleared state.
    """
    try:
        collection = chroma_client.get_or_create_collection(name=collection_name)
        existing = collection.get(include=[])
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    # Always leave behind a fresh, empty collection rather than relying on
    # the next process_file() call to lazily (and maybe inconsistently)
    # recreate it.
    chroma_client.get_or_create_collection(name=collection_name)