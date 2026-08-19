import logging
import os
import re

from dotenv import load_dotenv
from groq import Groq
from duckduckgo_search import DDGS
from ingest import chroma_client, embedding_model

load_dotenv()

logger = logging.getLogger(__name__)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Kept in one place so a future Groq deprecation only needs one edit here
# instead of a hunt through every chat.completions.create() call.
CHAT_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = (
    "You are DocuMind, an intelligent assistant with access to both uploaded "
    "documents and the internet. Answer primarily from the uploaded documents "
    "when they contain relevant information. When web results are also provided, "
    "use them to supplement the answer.\n\n"
    "For broad questions like 'explain the document' or 'what's the main "
    "topic', synthesize an overview from whatever document excerpts you were "
    "given, even if none of them individually seem like a perfect match — "
    "they were retrieved specifically to help you answer this kind of "
    "question.\n\n"
    "Formatting rules for every answer:\n"
    "- Do NOT repeat phrases like 'from the document', 'according to the "
    "document', or 'from the web' before individual sentences or facts — say "
    "it at most once per source, as a section label, never inline mid-"
    "sentence.\n"
    "- When you have both document and web information, structure the reply "
    "as two short sections in one message: one covering what the document "
    "says, one covering what the web adds — each under its own short bold "
    "label (e.g. '**From your document:**' and '**From the web:**'), each "
    "written once, followed by the relevant content.\n"
    "- When you only have one source (just the document, or just the web), "
    "skip the labels entirely and just answer plainly.\n"
    "- Use bullet points whenever you're listing multiple facts, steps, "
    "options, or comparisons — it's easier to scan than a wall of prose. "
    "Use plain sentences for short, single-fact answers.\n\n"
    "Only use the retrieved document excerpts that are actually relevant to the "
    "question — ignore any that aren't, even if they were handed to you. Never "
    "mention, list, or comment on which documents were or weren't useful, never "
    "compare the uploaded documents to each other, and never explain your own "
    "reasoning process about the sources — the user only wants the answer "
    "itself. If the person just says something conversational (like a greeting), "
    "respond naturally and briefly instead of pulling in document or web content."
)

# Chunks with a similarity distance above this are treated as noise, not
# real matches — without this, ChromaDB always returns its "closest" chunks
# even for a query like "hello" that has nothing to do with any document,
# and those irrelevant chunks would get fed to the model as if they mattered.
MAX_RELEVANT_DISTANCE = 1.1

# Whole-document questions ("explain the document", "what's the main
# topic?", "summarize this") don't resemble any single chunk closely in
# embedding space — the question is broad, the chunks are narrow — so they
# were getting filtered out by MAX_RELEVANT_DISTANCE and wrongly treated as
# "nothing found". These need the top chunks regardless of distance.
OVERVIEW_PATTERN = re.compile(
    r"\b(explain|summarize|summarise|overview|outline|compare|contrast)\b.*\b(document|doc|file|"
    r"this|it)s?\b"
    r"|\bwhat('?s| is)\s+(this|the)\s+(document|doc|file)s?\s+(about|for)\b"
    r"|\bwhat('?s| is)\s+the\s+main\s+(topic|idea|point|subject)\b"
    r"|\bwhat\s+(is|are)\s+(this|the)\s+(document|doc|file)s?\s+(about|cover|discuss)"
    r"|\b(give|provide)\b.*\b(summary|overview|gist|rundown|recap|comparison)\b"
    r"|\b(summary|overview|gist|rundown|recap|comparison|differences?|similarit(y|ies))\b.*\b(document|doc|file)s?\b"
    r"|\b(document|doc|file)s?\b.*\b(summary|overview|gist|rundown|recap|comparison|differences?|similarit(y|ies))\b"
    r"|\bcompare\b.*\b(document|doc|file)s?\b"
    r"|\b(document|doc|file)s?\b.*\bcompare\b"
    r"|^(compare|contrast)\s*(this|it|them)?[\s!.,?]*$"
    r"|^summarize\s*(this|it|them)?[\s!.,?]*$",
    re.IGNORECASE,
)

# Greetings and other content-free openers never need document/web lookup
GREETING_PATTERN = re.compile(
    r"^(hi|hello|hey|hola|yo|sup|good\s?(morning|afternoon|evening)|greetings|"
    r"how are you|what'?s up)[\s!.,?]*$",
    re.IGNORECASE,
)

# Words that force a web search even if document has context
WEB_TRIGGERS = [
    "search", "look up", "find online", "google", "internet", "web",
    "latest", "current", "recent", "today", "news", "2024", "2025", "2026",
    "more", "explain more", "tell me more", "elaborate", "in detail",
    "in depth", "comprehensive", "expand", "further", "additional",
    "what else", "beyond", "outside the document", "in general",
    "worldwide", "globally", "overall", "broadly",
]


def retrieve_chunks(query, collection_name="pdf_docs", top_k=4, ignore_distance=False, allowed_sources=None):
    """Retrieve relevant chunks from the vector database, dropping any that
    are too dissimilar to actually be relevant (unless ignore_distance is
    set, for whole-document questions where no single chunk is expected to
    closely match the question itself).

    allowed_sources: if given, restricts the search to only chunks whose
    "source" filename is in this list. The underlying ChromaDB collection is
    a single shared store — nothing in it ever "expires" on its own, so
    without this filter, a query can surface chunks from documents that were
    uploaded in an earlier session and never explicitly removed, even though
    they're no longer shown as loaded in the UI. Passing the currently
    active filenames closes that gap.

    Note: source filtering is done here in plain Python rather than via
    ChromaDB's `where=` query argument. Passing `where=` kept coming back
    empty in practice even for chunks that demonstrably existed in the
    collection — a version/driver-specific quirk we can't fully diagnose
    without direct access to that environment. Fetching everything and
    filtering ourselves sidesteps that class of bug entirely.
    """
    collection = chroma_client.get_or_create_collection(name=collection_name)

    def _dump_all():
        """Pulls every chunk's text + metadata out of the collection, no
        embedding search involved — the simplest possible ChromaDB call, so
        the least likely to be affected by any where-clause quirk."""
        try:
            got = collection.get(include=["documents", "metadatas"])
        except Exception:
            # Previously silent — a real ChromaDB error here (stale handle,
            # corrupted store, schema mismatch after reset_collection()) was
            # indistinguishable from "the collection is genuinely empty",
            # which is exactly what made whole-document questions (compare/
            # summarize) fail with no trace of why. Log it so a failure here
            # shows up instead of masquerading as "no documents found".
            logger.exception("ChromaDB collection.get() failed in _dump_all()")
            return [], []
        return got.get("documents") or [], got.get("metadatas") or []

    def _matches_allowed(meta):
        if allowed_sources is None:
            return True
        return meta.get("source") in allowed_sources

    # Whole-document / "summarize everything" / "compare the documents"
    # questions: pull a balanced sample per file rather than relying on
    # embedding similarity, since the question itself won't closely match
    # any one narrow chunk. Querying per-source guarantees every loaded
    # document actually contributes, instead of one file's chunks crowding
    # out another's just because they happen to embed closer to the query.
    if ignore_distance and allowed_sources:
        all_docs, all_metas = _dump_all()
        by_source = {}
        for chunk, meta in zip(all_docs, all_metas):
            src = meta.get("source")
            if src in allowed_sources:
                by_source.setdefault(src, []).append((chunk, meta))

        per_source = max(2, top_k // len(allowed_sources))
        combined = []
        for source in allowed_sources:
            combined.extend(by_source.get(source, [])[:per_source])
        return combined

    # Normal similarity search. Over-fetch generously so that, after we
    # filter down to allowed_sources ourselves, there's still a good chance
    # of finding real matches even if the collection has accumulated chunks
    # from other, unrelated documents that happen to embed closer to the
    # query than the ones we actually want.
    query_embedding = embedding_model.encode([query]).tolist()
    fetch_n = max(top_k, 60) if allowed_sources is not None else top_k
    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=fetch_n,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        logger.exception("ChromaDB collection.query() failed in retrieve_chunks()")
        return []
    if not results["documents"] or not results["documents"][0]:
        return []
    matches = [
        (chunk, meta, distance)
        for chunk, meta, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
        if _matches_allowed(meta)
    ]

    if ignore_distance:
        return [(chunk, meta) for chunk, meta, _ in matches[:top_k]]

    filtered = [
        (chunk, meta) for chunk, meta, distance in matches
        if distance <= MAX_RELEVANT_DISTANCE
    ]

    # Last resort: if the allowed files exist in the collection but simply
    # didn't land in the top `fetch_n` nearest neighbors (a large, noisy
    # collection can do this), fall back to the same balanced per-source
    # dump used for overview questions, filtered by plain relevance to
    # returning *something* from the right files rather than nothing.
    if not filtered and allowed_sources is not None:
        all_docs, all_metas = _dump_all()
        for chunk, meta in zip(all_docs, all_metas):
            if meta.get("source") in allowed_sources:
                filtered.append((chunk, meta))
                if len(filtered) >= top_k:
                    break

    return filtered[:top_k] if filtered else filtered


def web_search(query, max_results=4):
    """Search the web using DuckDuckGo — free, no API key needed."""
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception:
        # DuckDuckGo's free endpoint rate-limits aggressively on back-to-back
        # calls — this used to fail silently, which is why a second query
        # right after a first (e.g. "compare" then "combined summary") could
        # come back with nothing from the web *and* nothing from the docs,
        # landing on the "couldn't find anything anywhere" fallback with no
        # indication that the web leg was actually a rate-limit, not a
        # genuine empty result.
        logger.exception("DuckDuckGo web_search() failed for query: %r", query)
        return []


def build_context(retrieved, web_results):
    """Combines document chunks and web results into one context block."""
    parts = []
    if retrieved:
        doc_blocks = [
            f"[📄 Document: {meta['source']} — Page {meta['page']}]\n{chunk}"
            for chunk, meta in retrieved
        ]
        parts.append("=== FROM UPLOADED DOCUMENTS ===\n" + "\n\n".join(doc_blocks))
    if web_results:
        web_blocks = [
            f"[🌐 Web: {r.get('title', 'Result')}]\n{r.get('body', '')}"
            for r in web_results
        ]
        parts.append("=== FROM WEB SEARCH ===\n" + "\n\n".join(web_blocks))
    return "\n\n".join(parts)


def stream_answer(query, chat_history=None, collection_name="pdf_docs", top_k=4, allowed_sources=None):
    """
    Searches the document (and the web, when needed) for every real question.
    - Document gives specific context from uploaded files.
    - Web gives broader, up-to-date information.
    allowed_sources: filenames currently loaded in this session/chat — see
    retrieve_chunks() docstring for why this matters.
    Returns (sources_list, text_generator).
    """
    # A plain greeting doesn't need document/web lookup at all — answering it
    # with retrieved chunks is exactly what caused irrelevant, unrelated
    # documents to get pulled into the reply.
    if GREETING_PATTERN.match(query.strip()):
        def greeting_gen():
            stream = groq_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.4,
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        return [], greeting_gen()

    query_lower = query.lower()
    wants_web = any(trigger in query_lower for trigger in WEB_TRIGGERS)
    is_overview = bool(OVERVIEW_PATTERN.search(query_lower))

    # Step 1: Always check the document. Whole-document questions ("explain
    # the document", "what's the main topic") pull a wider net of chunks and
    # skip the similarity filter, since the question itself is broad and
    # won't closely match any single narrow chunk. Scale the sample size up
    # with the number of loaded files so a "combined summary" request still
    # gets a meaningful slice of each document rather than being dominated
    # by whichever one happens to embed closest to the query.
    overview_top_k = max(top_k, 8, 4 * len(allowed_sources)) if allowed_sources else max(top_k, 8)
    retrieved = retrieve_chunks(
        query,
        collection_name,
        top_k=(overview_top_k if is_overview else top_k),
        ignore_distance=is_overview,
        allowed_sources=allowed_sources,
    )

    # Step 2: Search web only if:
    #   - document has no relevant info, OR
    #   - user explicitly asked for web/more info
    # (previously this always ran, which added a web-search delay before the
    # first streamed token even for plain in-document questions)
    no_doc_context = len(retrieved) == 0
    run_web = no_doc_context or wants_web

    web_results = web_search(query) if run_web else []

    # Step 3: Build sources for display
    sources = []
    if web_results:
        sources += [r.get("href", "Web") for r in web_results[:2]]
    if retrieved:
        sources += [f"{meta['source']} p.{meta['page']}" for _, meta in retrieved]

    # Step 4: Nothing found anywhere. A real assistant doesn't just dead-end
    # here — it still gives the best answer it can from general knowledge
    # and says so, rather than a flat refusal.
    if not retrieved and not web_results:
        def fallback_gen():
            fallback_prompt = (
                f"The user asked: {query}\n\n"
                f"Nothing relevant was found in their uploaded document(s) or "
                f"on the web for this question. Briefly let them know that, "
                f"then answer as helpfully as you can from your own general "
                f"knowledge if the question is answerable that way. If it's "
                f"not a question you can meaningfully answer without their "
                f"document, ask a short clarifying question instead."
            )
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            if chat_history:
                messages.extend(chat_history[-6:])
            messages.append({"role": "user", "content": fallback_prompt})
            stream = groq_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.4,
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        return [], fallback_gen()

    # Step 5: Build combined context and prompt
    context = build_context(retrieved, web_results)

    # Multi-document overview questions ("summarize all documents", "compare
    # the documents") need the answer split per source file, or content from
    # different uploads gets blended into one undifferentiated bullet list —
    # the user has no way to tell which fact came from which file. This is
    # the streaming-answer analog of the per-document separation insights.py
    # already does for topics/mind maps.
    doc_sources = sorted({meta["source"] for _, meta in retrieved}) if retrieved else []
    multi_doc_overview = is_overview and len(doc_sources) > 1

    if multi_doc_overview:
        doc_list = ", ".join(doc_sources)
        source_instruction = (
            f"The document excerpts below come from {len(doc_sources)} separate "
            f"files: {doc_list}. Organize your answer with ONE bold heading per "
            f"file, using its exact filename (e.g. '**{doc_sources[0]}**'), "
            "followed by bullet points covering only what that specific file "
            "contains. Never blend content from two different files under one "
            "heading or bullet. If web results are also provided, add a final "
            "'**From the web:**' section after all the document sections."
        )
    elif retrieved and web_results:
        source_instruction = (
            "You have both document excerpts and web results below. Answer "
            "the question using both, organized as two short sections — "
            "document info first, then web info — each with a single bold "
            "label at the top of that section only (not repeated per line). "
            "Use bullet points within each section for lists of facts, "
            "steps, or comparisons."
        )
    else:
        source_instruction = (
            "Answer the question using the information below. Use bullet "
            "points if you're listing multiple facts, steps, or comparisons; "
            "otherwise write plainly. Don't add a source label since there's "
            "only one source here."
        )

    # The "don't list/compare documents" guardrail exists to stop the model
    # from editorializing about source relevance on ordinary questions — but
    # it directly contradicts what a multi-document overview needs, so skip
    # it in that case.
    no_editorializing = (
        "" if multi_doc_overview else
        " Do not mention which documents were or weren't relevant, and don't "
        "compare or list the uploaded documents — just answer the question."
    )
    user_prompt = (
        f"Available information:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"{source_instruction}{no_editorializing}"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history[-6:])
    messages.append({"role": "user", "content": user_prompt})

    # Step 6: Stream the answer
    stream = groq_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        stream=True,
    )

    def gen():
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    return sources, gen()


def generate_answer(query, chat_history=None, collection_name="pdf_docs", top_k=4):
    """Non-streaming version kept for insights.py compatibility."""
    retrieved = retrieve_chunks(query, collection_name, top_k)
    if not retrieved:
        return "I couldn't find anything relevant in the documents.", []
    doc_blocks = [
        f"[{meta['source']} — Page {meta['page']}]\n{chunk}"
        for chunk, meta in retrieved
    ]
    context = "\n\n".join(doc_blocks)
    user_prompt = (
        f"Context:\n{context}\n\nQuestion: {query}\n\n"
        f"Answer using the context above."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history[-6:])
    messages.append({"role": "user", "content": user_prompt})
    response = groq_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    answer = response.choices[0].message.content
    sources = sorted({meta["page"] for _, meta in retrieved})
    return answer, sources