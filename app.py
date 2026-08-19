import re
import os
import html
import logging
import tempfile
from datetime import datetime

logging.basicConfig(level=logging.WARNING)

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from ingest import process_file, remove_file_chunks, reset_collection
from rag import stream_answer
from insights import (
    extract_topics,
    generate_mindmap_data,
    generate_conversation_title,
    generate_followup,
)
from chat_store import (
    init_db,
    create_conversation,
    list_conversations,
    rename_conversation,
    delete_conversation,
    get_messages,
    get_conversation_name,
    add_message,
    clear_conversation_messages,
)

load_dotenv()
init_db()

SUPPORTED_DOCS = ["pdf", "docx", "txt"]
FILE_LABELS = {"pdf": "PDF", "docx": "DOCX", "txt": "TXT"}
ASSISTANT_AVATAR = "🧠"

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #212121;
        --bg-sidebar: #181818;
        --surface: #2f2f2f;
        --surface-hover: #383838;
        --border: #3a3a3a;
        --text-primary: #ececec;
        --text-muted: #9b9b9b;
        --accent: #5b8cff;
    }

    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
    code, .stCode, [data-testid="stChatMessage"] code { font-family: 'JetBrains Mono', ui-monospace, monospace; }

    .stApp { background-color: var(--bg-main); }

    /* Centered chat column, like Claude/ChatGPT rather than edge-to-edge */
    .block-container {
        padding-top: 2rem;
        max-width: 68rem;
        margin: 0 auto;
    }

    h1 {
        font-weight: 600;
        font-size: 1.6rem;
        letter-spacing: -0.3px;
        color: var(--text-primary);
    }

    .app-subtitle {
        color: var(--text-muted);
        font-size: 0.9rem;
        margin-top: -0.6rem;
        letter-spacing: 0.1px;
    }

    .doc-name { font-weight: 500; color: var(--text-primary); }
    .doc-meta { color: var(--text-muted); font-size: 0.78rem; }

    /* ── Sidebar ─────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] strong {
        color: var(--text-primary);
        font-weight: 600;
    }

    /* ── Buttons: quiet by default, no glow ─────────────────────── */
    .stButton>button {
        border-radius: 8px;
        border: 1px solid var(--border);
        background-color: var(--surface);
        color: var(--text-primary);
        font-weight: 400;
        transition: background-color 0.12s ease, border-color 0.12s ease;
    }
    .stButton>button:hover {
        border-color: #56636e;
        background-color: var(--surface-hover);
        color: var(--text-primary);
    }
    .stButton>button:focus-visible {
        outline: 2px solid var(--accent);
        outline-offset: 1px;
    }

    /* ── Chat bubbles ────────────────────────────────────────────
       Assistant: plain text, no border, avatar on the left (Claude/ChatGPT style)
       User: right-aligned rounded bubble on a subtle surface           */
    [data-testid="stChatMessage"] {
        background-color: transparent;
        border: none;
        padding: 0.35rem 0;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        background-color: var(--surface);
        border-radius: 16px;
        padding: 0.55rem 0.95rem;
        max-width: 75%;
        margin-left: auto;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
        padding: 0.15rem 0;
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: var(--accent) !important;
    }

    /* ── Chat input: rounded pill, subtle shadow ────────────────── */
    [data-testid="stChatInput"] textarea {
        border-radius: 20px !important;
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stChatInput"] {
        box-shadow: 0 2px 14px rgba(0, 0, 0, 0.25);
        border-radius: 20px;
        margin-right: 8px;
    }
    /* Streamlit's own send button, kept small and unobtrusive */
    [data-testid="stChatInputSubmitButton"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        background-color: var(--surface);
    }

    hr, [data-testid="stDivider"] { border-color: var(--border) !important; }

    [data-testid="stMetricValue"], .doc-row { color: var(--text-primary); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session State ─────────────────────────────────────────────────────────────
if "conversation_id" not in st.session_state:
    # Resume "Conversation 1" (the very first conversation ever created) if
    # it still exists, instead of spawning a new empty conversation on every
    # fresh browser session. Falls back to creating one only if none exist.
    existing = list_conversations()
    default_conv = next(
        (c for c in existing if c["name"] == "Conversation 1"),
        existing[-1] if existing else None,  # oldest conversation as fallback
    )
    if default_conv:
        st.session_state.conversation_id = default_conv["id"]
        st.session_state.messages = get_messages(default_conv["id"])
    else:
        st.session_state.conversation_id = create_conversation("Conversation 1")
        st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = {}
if "topics" not in st.session_state:
    st.session_state.topics = []
if "mindmap_data" not in st.session_state:
    st.session_state.mindmap_data = None
if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0
if "last_followup" not in st.session_state:
    st.session_state.last_followup = None


def clear_all_documents():
    reset_collection()
    st.session_state.indexed_files = {}
    st.session_state.messages = []
    clear_conversation_messages(st.session_state.conversation_id)
    st.session_state.topics = []
    st.session_state.mindmap_data = None
    st.session_state.uploader_version += 1


def remove_single_document(fname):
    remove_file_chunks(fname)
    del st.session_state.indexed_files[fname]
    # Clear insights since they were built from the old file set
    st.session_state.topics = []
    st.session_state.mindmap_data = None
    st.session_state.uploader_version += 1


def clear_chat():
    st.session_state.messages = []
    clear_conversation_messages(st.session_state.conversation_id)


def append_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})
    add_message(st.session_state.conversation_id, role, content)


DEFAULT_NAME_PATTERN = re.compile(r"^Conversation \d+$")


def maybe_autoname_conversation(user_message, assistant_message):
    current_name = get_conversation_name(st.session_state.conversation_id)
    if not current_name or not DEFAULT_NAME_PATTERN.match(current_name):
        return
    title = generate_conversation_title(user_message, assistant_message)
    if title:
        rename_conversation(st.session_state.conversation_id, title)


def run_streaming_answer(query, history):
    """
    Runs stream_answer and displays using st.write_stream (typewriter effect).
    Shows a brief "Thinking..." state while retrieval/search happens, so there's
    no dead air before the first token streams in.
    Returns the full answer text including source citations.
    Must be called inside a st.chat_message("assistant") block.
    """
    with st.spinner("Thinking..."):
        sources, gen = stream_answer(
            query,
            chat_history=history,
            allowed_sources=list(st.session_state.indexed_files.keys()),
        )
    answer = st.write_stream(gen)
    if sources:
        source_text = f"\n\n*Source(s): {', '.join(sources)}*"
        st.caption(f"📄 Source(s): {', '.join(sources)}")
        answer += source_text
    return answer


def render_followup(query, answer):
    """
    Generates a one-line summary of the answer plus 3 follow-up questions
    and stores them in session state, tied to the conversation and the
    specific message they belong to. They're actually drawn later, by
    render_persisted_followup(), which runs on every script rerun — not
    just this one. Streamlit throws away any UI that isn't re-created on
    the current rerun, so buttons drawn only inside this "just generated an
    answer" code path would vanish (and their click would go unhandled) the
    moment the page reran for any other reason, e.g. because the button
    itself was clicked. Persisting the data and always redrawing it fixes
    that.
    """
    with st.spinner("Wrapping up..."):
        followup = generate_followup(query, answer)

    if not followup:
        st.session_state.last_followup = None
        return

    st.session_state.last_followup = {
        "conversation_id": st.session_state.conversation_id,
        "message_index": len(st.session_state.messages) - 1,
        "summary": followup.get("summary"),
        "questions": [q for q in (followup.get("questions") or []) if q][:3],
    }


def render_persisted_followup():
    """
    Draws the summary + 'You might also ask' buttons for the most recent
    assistant message, using data stashed by render_followup(). Called
    unconditionally near the bottom of every script run so the buttons
    survive reruns (including the rerun caused by clicking one of them) and
    stay clickable instead of silently disappearing.
    """
    fu = st.session_state.last_followup
    if not fu:
        return
    # Guard against showing stale buttons from a previous conversation or a
    # message that's no longer the last one (e.g. a new question was asked).
    if fu["conversation_id"] != st.session_state.conversation_id:
        return
    if fu["message_index"] != len(st.session_state.messages) - 1:
        return

    if fu.get("summary"):
        st.markdown(f"**In short:** {fu['summary']}")

    questions = fu.get("questions") or []
    if questions:
        st.markdown("**You might also ask:**")
        cols = st.columns(len(questions))
        for i, q in enumerate(questions):
            with cols[i]:
                if st.button(
                    q,
                    key=f"followup_{st.session_state.conversation_id}_{fu['message_index']}_{i}",
                    use_container_width=True,
                ):
                    st.session_state.last_followup = None
                    append_message("user", q)
                    st.rerun()


def render_mindmap_html(data):
    """
    Builds a self-contained HTML/CSS/JS mind map: a central node with
    branches radiating left/right and their children further out, joined by
    curved connectors — like a hand-drawn mind map rather than Mermaid's
    boxy 'neutral' theme. Kept to a single neutral palette (no rainbow
    per-branch colors). Nodes are plain HTML pills that size themselves to
    their text, so labels never get clipped.
    Returns (height_px, html_string).
    """
    central = html.escape(str(data.get("central_topic", "Document")))
    branches = data.get("branches", []) or []

    # Alternate branches between the two sides so both stay roughly balanced
    left, right = [], []
    for i, b in enumerate(branches):
        (left if i % 2 == 0 else right).append(b)

    def branch_row(branch, side):
        name = html.escape(str(branch.get("name", "Branch")))
        children = branch.get("children", []) or []
        child_html = "".join(
            f'<div class="mm-child">{html.escape(str(c))}</div>' for c in children
        )
        children_col = f'<div class="mm-children">{child_html}</div>' if children else ""
        branch_pill = f'<div class="mm-branch">{name}</div>'
        if side == "left":
            return f'<div class="mm-row">{children_col}{branch_pill}</div>'
        return f'<div class="mm-row">{branch_pill}{children_col}</div>'

    left_html = "".join(branch_row(b, "left") for b in left)
    right_html = "".join(branch_row(b, "right") for b in right)

    def side_rows(bs):
        return sum(max(1, len(b.get("children", []) or [])) for b in bs) + max(0, len(bs) - 1)

    est_rows = max(side_rows(left), side_rows(right), 1)
    height = max(420, min(1100, est_rows * 58 + 160))

    page = f"""
    <div class="mm-wrap">
      <style>
        :root {{
          --mm-bg: #212121; --mm-surface: #2b2b2b; --mm-border: #3a3a3a;
          --mm-text: #ececec; --mm-text-muted: #a3a3a3;
          --mm-accent: #5b8cff; --mm-line: #4a4a4a;
        }}
        * {{ box-sizing: border-box; }}
        html, body {{ margin: 0; padding: 0; background: var(--mm-bg); width: 100%; overflow: hidden; }}
        .mm-wrap {{
          font-family: 'Inter', -apple-system, sans-serif;
          background: var(--mm-bg); color: var(--mm-text);
          width: 100%; height: {height}px; overflow: hidden;
          display: flex; align-items: center; justify-content: center;
        }}
        .mm-scale {{
          display: flex; align-items: center; justify-content: center;
        }}
        .mm-grid {{
          position: relative; display: flex; align-items: center;
          justify-content: center; gap: 56px;
        }}
        .mm-side {{ display: flex; flex-direction: column; justify-content: center; gap: 22px; }}
        #mm-left {{ align-items: flex-end; }}
        #mm-right {{ align-items: flex-start; }}
        .mm-row {{ display: flex; align-items: center; gap: 18px; }}
        .mm-children {{ display: flex; flex-direction: column; gap: 8px; }}
        #mm-left .mm-children {{ align-items: flex-end; }}
        #mm-right .mm-children {{ align-items: flex-start; }}
        .mm-branch {{
          background: var(--mm-surface); border: 1px solid var(--mm-border);
          color: var(--mm-text); font-weight: 600; font-size: 13.5px;
          padding: 9px 16px; border-radius: 999px; white-space: nowrap;
          box-shadow: 0 1px 4px rgba(0,0,0,0.25);
        }}
        .mm-child {{
          background: transparent; border: 1px solid var(--mm-border);
          color: var(--mm-text-muted); font-size: 12.5px;
          padding: 6px 14px; border-radius: 999px; white-space: nowrap;
        }}
        .mm-center {{
          background: var(--mm-accent); color: #fff; font-weight: 700;
          font-size: 15px; padding: 16px 26px; border-radius: 999px;
          white-space: nowrap; box-shadow: 0 2px 10px rgba(91,140,255,0.35);
          z-index: 2;
        }}
        svg.mm-lines {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; overflow: visible; }}
        svg.mm-lines path {{ fill: none; stroke: var(--mm-line); stroke-width: 1.6; }}
      </style>
      <div class="mm-scale" id="mm-scale">
        <div class="mm-grid" id="mm-grid">
          <svg class="mm-lines" id="mm-lines"></svg>
          <div class="mm-side" id="mm-left">{left_html}</div>
          <div class="mm-center" id="mm-center">{central}</div>
          <div class="mm-side" id="mm-right">{right_html}</div>
        </div>
      </div>
    </div>
    <script>
      (function() {{
        const wrap = document.querySelector('.mm-wrap');
        const scaleBox = document.getElementById('mm-scale');
        const grid = document.getElementById('mm-grid');
        const svg = document.getElementById('mm-lines');
        const center = document.getElementById('mm-center');

        function anchor(el, side) {{
          const r = el.getBoundingClientRect(), c = grid.getBoundingClientRect();
          const x = side === 'right' ? r.right - c.left : r.left - c.left;
          const y = r.top + r.height / 2 - c.top;
          return {{x: x, y: y}};
        }}

        function curve(p1, p2) {{
          const dx = Math.max(24, Math.abs(p2.x - p1.x) * 0.5);
          const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
          path.setAttribute('d', 'M ' + p1.x + ' ' + p1.y + ' C ' + (p1.x + dx) + ' ' + p1.y + ', ' + (p2.x - dx) + ' ' + p2.y + ', ' + p2.x + ' ' + p2.y);
          path.setAttribute('fill', 'none');
          path.setAttribute('stroke', '#4a4a4a');
          path.setAttribute('stroke-width', '1.6');
          svg.appendChild(path);
        }}

        function link(a, b) {{
          const ac = a.getBoundingClientRect(), bc = b.getBoundingClientRect();
          const aIsLeft = ac.left < bc.left;
          curve(anchor(a, aIsLeft ? 'right' : 'left'), anchor(b, aIsLeft ? 'left' : 'right'));
        }}

        // 1) Draw everything at natural, un-shrunk size first so the pills
        // and connector lines are computed/drawn in one consistent
        // coordinate space (this is what keeps lines lined up with pills).
        document.querySelectorAll('.mm-branch').forEach(function(branchEl) {{
          link(center, branchEl);
          const row = branchEl.closest('.mm-row');
          row.querySelectorAll('.mm-child').forEach(function(childEl) {{
            link(branchEl, childEl);
          }});
        }});

        // 2) Only now measure the natural (unscaled) size of the whole
        // diagram and, if it's wider or taller than the visible area,
        // shrink the ENTIRE diagram (pills + already-drawn lines) as one
        // unit with a CSS transform. Because everything scales together,
        // nothing wraps letter-by-letter and the lines stay perfectly
        // aligned with the pills.
        const naturalWidth = grid.scrollWidth;
        const naturalHeight = grid.scrollHeight;
        const availWidth = wrap.clientWidth - 8;
        const availHeight = wrap.clientHeight - 8;
        const scale = Math.min(1.3, availWidth / naturalWidth, availHeight / naturalHeight);
        if (scale != 1) {{
          scaleBox.style.transform = 'scale(' + scale + ')';
          scaleBox.style.transformOrigin = 'center center';
        }}
      }})();
    </script>
    """
    return height, page


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### DocuMind")
    st.markdown('<div class="app-subtitle">Document Q&A Assistant</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("**Conversations**")
    conversations = list_conversations()
    if not conversations:
        fallback_id = create_conversation()
        st.session_state.conversation_id = fallback_id
        st.session_state.messages = []
        conversations = list_conversations()

    conv_options = {c["id"]: c["name"] for c in conversations}
    conv_ids = list(conv_options.keys())

    current_index = (
        conv_ids.index(st.session_state.conversation_id)
        if st.session_state.conversation_id in conv_ids
        else 0
    )
    selected_id = st.selectbox(
        "Conversation",
        options=conv_ids,
        format_func=lambda cid: conv_options.get(cid, "Untitled"),
        index=current_index,
        label_visibility="collapsed",
    )

    if selected_id != st.session_state.conversation_id:
        st.session_state.conversation_id = selected_id
        st.session_state.messages = get_messages(selected_id)
        st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("New chat", use_container_width=True):
            new_id = create_conversation()
            st.session_state.conversation_id = new_id
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("Delete", use_container_width=True, disabled=len(conversations) <= 1):
            delete_conversation(st.session_state.conversation_id)
            remaining = list_conversations()
            if not remaining:
                new_id = create_conversation()
                st.session_state.conversation_id = new_id
                st.session_state.messages = []
            else:
                st.session_state.conversation_id = remaining[0]["id"]
                st.session_state.messages = get_messages(remaining[0]["id"])
            st.rerun()


    with st.expander("Rename conversation"):
        new_name = st.text_input(
            "Name",
            value=conv_options.get(st.session_state.conversation_id, ""),
            label_visibility="collapsed",
        )
        if st.button("Save name", use_container_width=True):
            rename_conversation(st.session_state.conversation_id, new_name.strip() or "Untitled")
            st.rerun()

    st.divider()
    st.markdown("**Documents**")

    uploaded_file = st.file_uploader(
        "Add a document",
        type=SUPPORTED_DOCS,
        help="Upload PDF, Word, or Text files — you can upload more than one",
        key=f"uploader_{st.session_state.uploader_version}",
        label_visibility="collapsed",
    )

    if uploaded_file and uploaded_file.name not in st.session_state.indexed_files:
        # Adds to the existing document set instead of wiping it —
        # no separate "add another document" step needed.
        ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
        with st.spinner(f"Indexing {uploaded_file.name}..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                num_chunks = process_file(tmp_path, uploaded_file.name)
                os.unlink(tmp_path)
                st.session_state.indexed_files[uploaded_file.name] = num_chunks
                st.session_state.topics = []
                st.session_state.mindmap_data = None
                st.session_state.uploader_version += 1
                if num_chunks == 0:
                    # process_file() returns 0 (no exception) when text
                    # extraction found nothing to embed — most often a
                    # scanned/image-only PDF pdfplumber can't read without
                    # OCR. Previously this looked identical to a successful
                    # upload (same toast), so the file sat in the sidebar as
                    # "loaded" while being completely invisible to every
                    # query — exactly what breaks compare/summarize when it
                    # happens to one of two uploaded files.
                    st.warning(
                        f"'{uploaded_file.name}' was added but no text could "
                        f"be extracted from it, so it won't be searchable. "
                        f"If it's a scanned PDF, try a version with a text "
                        f"layer, or a .docx/.txt export instead."
                    )
                else:
                    st.toast(f"Indexed {uploaded_file.name} ({num_chunks} chunks)")
                st.rerun()
            except Exception as e:
                os.unlink(tmp_path)
                st.error(f"Error: {e}")
    elif uploaded_file:
        st.caption(f"'{uploaded_file.name}' is already indexed.")

    if st.session_state.indexed_files:
        st.caption(f"{len(st.session_state.indexed_files)} file(s) loaded")

        with st.container(border=True):
            for fname, nchunks in list(st.session_state.indexed_files.items()):
                ext = fname.rsplit(".", 1)[-1].lower()
                label = FILE_LABELS.get(ext, ext.upper())
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(
                        f'<div class="doc-name">{html.escape(fname)}</div>'
                        f'<div class="doc-meta">{html.escape(label)}</div>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    if st.button("Remove", key=f"remove_{fname}", use_container_width=True):
                        with st.spinner(f"Removing {fname}..."):
                            remove_single_document(fname)
                        st.rerun()

        if st.button("Clear all documents", use_container_width=True):
            clear_all_documents()
            st.toast("All documents cleared")
            st.rerun()

        st.divider()
        st.markdown("**Insights**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Extract topics", use_container_width=True):
                with st.spinner("Extracting topics..."):
                    st.session_state.topics = extract_topics(
                        list(st.session_state.indexed_files.keys())
                    )
                st.rerun()
        with col2:
            if st.button("Generate mind map", use_container_width=True):
                with st.spinner("Building mind map..."):
                    mindmap_data = generate_mindmap_data(
                        list(st.session_state.indexed_files.keys())
                    )
                    st.session_state.mindmap_data = mindmap_data
                    if not mindmap_data:
                        st.error("Couldn't generate a mind map — try again.")
                st.rerun()

        if st.button("Summarize all documents", use_container_width=True):
            append_message("user", "Give me a combined summary of all the documents.")
            st.rerun()

        st.divider()
        st.markdown("**Suggested questions**")
        for q in [
            "What is the main topic?",
            "List the key points.",
            "What are the conclusions?",
            "Compare the documents.",
        ]:
            if st.button(q, key=q, use_container_width=True):
                append_message("user", q)
                st.rerun()

    st.divider()
    st.markdown("**Chat**")

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.messages:
            chat_export = (
                f"DocuMind Chat\n"
                f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"{'=' * 50}\n\n"
            )
            for msg in st.session_state.messages:
                role = "You" if msg["role"] == "user" else "DocuMind"
                chat_export += f"{role}:\n{msg['content']}\n\n"
            st.download_button(
                "Export",
                data=chat_export,
                file_name=f"docmind_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.button("Export", use_container_width=True, disabled=True)
    with col2:
        if st.button("Clear", use_container_width=True, disabled=not st.session_state.messages):
            clear_chat()
            st.rerun()

    if not os.environ.get("GROQ_API_KEY"):
        st.divider()
        st.warning("Set GROQ_API_KEY in your .env file.")

    st.divider()
    st.caption("Powered by Groq · LLaMA 3.3 · ChromaDB")

# ── Main Area ─────────────────────────────────────────────────────────────────
st.title("DocuMind")

if not st.session_state.indexed_files:
    st.info("Upload one or more documents in the sidebar to get started.")
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("**PDF**")
            st.caption("Textbooks, papers, reports.")
    with col2:
        with st.container(border=True):
            st.markdown("**Word**")
            st.caption("Assignments, essays (.docx).")
    with col3:
        with st.container(border=True):
            st.markdown("**Text**")
            st.caption("Notes, logs (.txt).")
else:
    files_list = ", ".join(st.session_state.indexed_files.keys())
    st.caption(f"Searching across: {files_list}")

# ── Topics & Mind Map ─────────────────────────────────────────────────────────
if st.session_state.topics:
    with st.expander("Extracted topics", expanded=True):
        # Group by document when multiple files are loaded
        by_doc = {}
        for topic in st.session_state.topics:
            doc = topic.get("document", "Document")
            by_doc.setdefault(doc, []).append(topic)

        if len(by_doc) > 1:
            for doc_name, topics in by_doc.items():
                st.markdown(f"**📄 {doc_name}**")
                for topic in topics:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;• **{topic.get('name', 'Untitled')}**")
                    st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{topic.get('summary', '')}")
        else:
            for topic in st.session_state.topics:
                st.markdown(f"**{topic.get('name', 'Untitled')}**")
                st.caption(topic.get("summary", ""))

if st.session_state.mindmap_data:
    with st.expander("Mind map", expanded=True):
        mm_height, mm_html = render_mindmap_html(st.session_state.mindmap_data)
        components.html(mm_html, height=mm_height, scrolling=False)

# ── Chat ──────────────────────────────────────────────────────────────────────
num_messages = len(st.session_state.messages)
for idx, msg in enumerate(st.session_state.messages):
    avatar = ASSISTANT_AVATAR if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        # Redraw follow-up buttons for the most recent assistant message on
        # every rerun (not just the run that generated it) — see
        # render_persisted_followup() docstring for why this matters.
        if msg["role"] == "assistant" and idx == num_messages - 1:
            render_persisted_followup()

# Process pending sidebar-triggered message with streaming
last = st.session_state.messages
if last and last[-1]["role"] == "user" and (
    len(last) == 1 or last[-2]["role"] == "assistant"
):
    st.session_state.last_followup = None  # this question makes any old buttons stale
    pending = last[-1]["content"]
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        was_first = len(st.session_state.messages) == 1
        answer = run_streaming_answer(pending, history)
        append_message("assistant", answer)
        render_followup(pending, answer)
        if was_first:
            maybe_autoname_conversation(pending, answer)
        # Rerun so the just-generated follow-up buttons are picked up by the
        # main message loop above (that's the only place that redraws them
        # on every subsequent rerun too, keeping them clickable).
        st.rerun()

# Chat input with streaming
if user_input := st.chat_input("Ask across all your documents..."):
    st.session_state.last_followup = None  # this question makes any old buttons stale
    was_first = len(st.session_state.messages) == 0
    append_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        answer = run_streaming_answer(user_input, history)
        append_message("assistant", answer)
        render_followup(user_input, answer)
        if was_first:
            maybe_autoname_conversation(user_input, answer)
        st.rerun()