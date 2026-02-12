import os
import json
import fitz  # pymupdf

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


# --- OpenAI tool schemas ---

TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "read_document",
        "description": "Read the full content of a specific document about Ofer Brodatch. You MUST call this before every response.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The filename of the document to read (from the list in the system prompt).",
                }
            },
            "required": ["filename"],
        },
    },
]

# Map function names to callables
TOOL_FUNCTIONS = {
    "list_documents": lambda **kwargs: list_documents(),
    "read_document": lambda **kwargs: read_document(kwargs["filename"]),
}
