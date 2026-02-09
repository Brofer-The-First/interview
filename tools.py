import os
import json
import fitz  # pymupdf
from duckduckgo_search import DDGS

DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")

# --- Tool implementations ---

def list_documents() -> str:
    """List all available documents."""
    files = []
    for f in os.listdir(DOCUMENTS_DIR):
        if os.path.isfile(os.path.join(DOCUMENTS_DIR, f)):
            files.append(f)
    return json.dumps(files)


def read_document(filename: str) -> str:
    """Read the content of a document by filename."""
    filepath = os.path.join(DOCUMENTS_DIR, filename)

    if not os.path.isfile(filepath):
        return f"Error: Document '{filename}' not found."

    # Prevent path traversal
    if not os.path.realpath(filepath).startswith(os.path.realpath(DOCUMENTS_DIR)):
        return "Error: Invalid filename."

    if filepath.lower().endswith(".pdf"):
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    # Plain text / markdown
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return the top results."""
    try:
        results = DDGS().text(query, max_results=5)
    except Exception:
        return "web_search tool currently unavailable. Do not attempt to use it again."
    if not results:
        return "No results found."
    output = []
    for r in results:
        output.append(f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}")
    return "\n\n".join(output)


# --- OpenAI tool schemas ---

TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "list_documents",
        "description": "List all available documents about Ofer Brodatch. Call this first to discover what information is available.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "read_document",
        "description": "Read the full content of a specific document about Ofer Brodatch.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The filename of the document to read (as returned by list_documents).",
                }
            },
            "required": ["filename"],
        },
    },
    {
        "type": "function",
        "name": "web_search",
        "description": "Search the web for current information. Use this to supplement your knowledge when answering questions that may benefit from up-to-date or external data.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                }
            },
            "required": ["query"],
        },
    },
]

# Map function names to callables
TOOL_FUNCTIONS = {
    "list_documents": lambda **kwargs: list_documents(),
    "read_document": lambda **kwargs: read_document(kwargs["filename"]),
    "web_search": lambda **kwargs: web_search(kwargs["query"]),
}
