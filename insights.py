import json

from ingest import chroma_client
from rag import groq_client

INSIGHTS_MODEL = "openai/gpt-oss-120b"   


def _sample_chunks(filenames, collection_name="pdf_docs", chunks_per_file=6):
    """
    Pulls chunks per file and LABELS each chunk with its source document name.
    This is the key fix: when two documents are uploaded, each chunk now clearly
    states which document it came from, so the LLM can distinguish them.

    Note: this filters in plain Python rather than via ChromaDB's `where=`
    argument, mirroring the fix in rag.py's retrieve_chunks(). `where=` was
    found there to unreliably come back empty for chunks that demonstrably
    existed in the collection; using it here too would silently starve
    "Extract topics" / "Generate mind map" of content the same way.
    """
    collection = chroma_client.get_or_create_collection(name=collection_name)
    try:
        got = collection.get(include=["documents", "metadatas"])
    except Exception:
        return []
    all_docs = got.get("documents") or []
    all_metas = got.get("metadatas") or []

    by_source = {}
    for doc, meta in zip(all_docs, all_metas):
        src = meta.get("source")
        if src in filenames:
            by_source.setdefault(src, []).append(doc)

    labeled_chunks = []
    for fname in filenames:
        for doc in by_source.get(fname, [])[:chunks_per_file]:
            # Each chunk is clearly labeled with its source file
            labeled_chunks.append(f"[Document: {fname}]\n{doc}")
    return labeled_chunks


def _call_llm_json(prompt):
    """Calls the LLM expecting a raw JSON response and parses it defensively."""
    response = groq_client.chat.completions.create(
        model=INSIGHTS_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def extract_topics(filenames, collection_name="pdf_docs"):
    """
    Returns topics per document when multiple files are uploaded,
    so topics from Document A and Document B are never mixed together.
    Returns: [{"name": ..., "summary": ..., "document": ...}]
    """
    chunks = _sample_chunks(filenames, collection_name)
    if not chunks:
        return []

    combined_text = "\n\n".join(chunks)[:12000]
    multi_doc = len(filenames) > 1

    if multi_doc:
        doc_list = ", ".join(filenames)
        instruction = (
            f"The excerpts come from {len(filenames)} separate documents: {doc_list}. "
            "Each excerpt is labeled with [Document: filename]. "
            "Identify the 3-4 most important topics PER DOCUMENT — do NOT mix topics "
            "across documents. Include a 'document' field showing which file each topic "
            "belongs to."
        )
        structure = (
            '{"topics": [{"name": "Topic name", "summary": "One-sentence summary", '
            '"document": "exact filename here"}]}'
        )
    else:
        instruction = "Identify the 5-8 most important topics discussed overall."
        structure = (
            '{"topics": [{"name": "Topic name", "summary": "One-sentence summary", '
            '"document": "' + (filenames[0] if filenames else "") + '"}]}'
        )

    prompt = (
        f"Analyze the following document excerpts. {instruction}\n"
        "Respond with ONLY valid JSON, no markdown fences, no preamble, "
        f"in exactly this structure:\n{structure}\n\n"
        f"Document excerpts:\n{combined_text}"
    )

    data = _call_llm_json(prompt)
    if not data:
        return []
    return data.get("topics", [])


def generate_mindmap_data(filenames, collection_name="pdf_docs"):
    """
    Generates a mind map that clearly separates content by document
    when multiple files are uploaded. Each document gets its own top-level
    branch so content is never mixed.
    """
    chunks = _sample_chunks(filenames, collection_name)
    if not chunks:
        return None

    combined_text = "\n\n".join(chunks)[:12000]
    multi_doc = len(filenames) > 1

    if multi_doc:
        doc_list = ", ".join(filenames)
        instruction = (
            f"The excerpts come from {len(filenames)} separate documents: {doc_list}. "
            "Each excerpt is labeled with [Document: filename]. "
            f"Create a mind map where the TOP-LEVEL BRANCHES are the document names "
            f"({doc_list}), and the children under each branch are the key topics "
            "from ONLY that document. Never put a topic from Document A under Document B."
        )
        central = "Uploaded Documents"
    else:
        instruction = (
            "Build a mind map with 4-7 branches covering the main topics, "
            "each with 2-4 short child points."
        )
        central = filenames[0] if filenames else "Document"

    prompt = (
        f"Based on the document excerpts below, build a mind map. {instruction}\n"
        "Respond with ONLY valid JSON, no markdown fences, no preamble, "
        "in exactly this structure:\n"
        '{"central_topic": "Short title", "branches": '
        '[{"name": "Branch name", "children": ["child 1", "child 2"]}]}\n'
        f"Each child should be just a few words.\n\n"
        f"Document excerpts:\n{combined_text}"
    )

    return _call_llm_json(prompt)


def _clean_label(text):
    """Strip characters that break Mermaid mindmap syntax."""
    return (
        str(text)
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace(":", "-")
        .replace('"', "'")
        .strip()
    )


def mindmap_to_mermaid(mindmap_data):
    """Converts the nested topic dict into Mermaid mindmap syntax."""
    central = _clean_label(mindmap_data.get("central_topic", "Documents"))
    lines = ["mindmap", f"  root(({central}))"]
    for branch in mindmap_data.get("branches", []):
        lines.append(f"    {_clean_label(branch.get('name', 'Branch'))}")
        for child in branch.get("children", []):
            lines.append(f"      {_clean_label(child)}")
    return "\n".join(lines)


def generate_followup(query, answer):
    """
    After an answer is given, generates:
      - a one-sentence plain-language summary of that answer
      - 3 short, specific follow-up questions the user could ask next
    This is what powers the "summary + suggested next questions" shown
    under every chat response.
    Returns: {"summary": str, "questions": [str, str, str]} or None on failure.
    """
    prompt = (
        "You just answered a user's question about their uploaded document(s). "
        "Based on the question and answer below, do two things:\n"
        "1. Write a ONE-sentence plain-language summary of the answer (not a repeat "
        "of the question).\n"
        "2. Suggest exactly 3 short, specific follow-up questions the user could "
        "ask next to dig deeper into the same topic. Avoid vague questions like "
        "'tell me more'.\n\n"
        f"Question: {query}\n\nAnswer: {answer}\n\n"
        "Respond with ONLY valid JSON, no markdown fences, no preamble, in exactly "
        "this structure:\n"
        '{"summary": "one sentence here", "questions": ["question 1", "question 2", "question 3"]}'
    )
    return _call_llm_json(prompt)


def generate_conversation_title(user_message, assistant_message=None):
    """Generates a short title from the first exchange in a conversation."""
    excerpt = user_message
    if assistant_message:
        excerpt += "\n\n" + assistant_message[:500]

    prompt = (
        "Based on this chat exchange, write a short, descriptive conversation "
        "title (3-6 words). Respond with ONLY the title text — no quotes, "
        "no trailing punctuation, no preamble.\n\n"
        f"{excerpt}"
    )

    response = groq_client.chat.completions.create(
        model=INSIGHTS_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )

    title = response.choices[0].message.content.strip().strip('"').strip("'")
    return title[:60] if title else None
