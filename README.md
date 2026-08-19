 🧠 DocuMind

**Chat with your documents. Get real answers — not just search results.**

DocuMind is an AI-powered document assistant built with Streamlit. Upload PDFs, Word docs, or text files and ask questions in plain language. It retrieves the most relevant passages using semantic search, falls back to live web search when your documents don't have the answer, and streams back a clear, well-organized response — powered by Groq's LLM inference.

---

## ✨ Features

- 💬 **Chat with multiple documents at once** — upload several files and ask questions across all of them
- 🔍 **Semantic search, not keyword matching** — powered by sentence-transformer embeddings + ChromaDB
- 🌐 **Web search fallback** — automatically searches the web when your documents don't have the answer, or when you ask for "the latest" / "current" info
- 📑 **Multi-document summaries done right** — "Summarize all documents" separates each file into its own clearly labeled section, instead of blending everything together
- 🗺️ **Auto-generated mind maps** — visualize the key topics in your documents as an interactive mind map
- 🏷️ **Topic extraction** — pulls out the main topics per document automatically
- 💡 **Smart follow-ups** — every answer comes with a one-line summary and 3 suggested next questions
- 🗂️ **Persistent chat history** — conversations are saved locally (SQLite) and can be renamed, revisited, or deleted
- ⚡ **Streamed responses** — answers appear token-by-token, like a real chat experience
- 🎨 **Clean, dark-themed UI** — built to feel like a modern chat app, not a form

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| LLM inference | [Groq API](https://groq.com/) |
| Vector store | [ChromaDB](https://www.trychroma.com/) |
| Embeddings | [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| Document parsing | `pdfplumber`, `python-docx` |
| Web search | `duckduckgo-search` |
| Chat history | SQLite |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/DocuMind.git
cd DocuMind
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```
Get a free API key at [console.groq.com](https://console.groq.com/keys).

### 5. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📂 Project Structure

```
DocuMind/
├── app.py            # Streamlit UI and app state
├── rag.py             # Retrieval + LLM answer generation
├── ingest.py           # Document parsing, chunking, embedding
├── insights.py          # Topic extraction, mind maps, follow-ups
├── chat_store.py         # SQLite conversation history
├── requirements.txt
└── .env                # Your API key (not committed)
```

---

## 📄 Supported File Types

`.pdf`  ·  `.docx`  ·  `.txt`

---

## 🗺️ Roadmap

- [ ] Support for more file types (`.csv`, `.xlsx`)
- [ ] Export conversations
- [ ] Multi-user support
- [ ] Deployment guide (Docker / Streamlit Cloud)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a PR.

---

## 📜 License

This project currently has no license attached. Add one before accepting external contributions.
