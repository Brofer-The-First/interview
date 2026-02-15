import os
import json
import smtplib
from email.mime.text import MIMEText
import fitz  # pymupdf

DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")
ALERT_EMAIL = "ofer.brodatch@gmail.com"

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


def notify_missing_info(user_question: str, bot_response: str, missing_info: str) -> str:
    """Send an email to Ofer when the chatbot feels the documents lack info."""
    smtp_user = os.environ["SMTP_USER"]
    smtp_password = os.environ["SMTP_PASSWORD"]

    body = (
        f"A user asked a question that the documents couldn't fully cover.\n\n"
        f"--- User's Question ---\n{user_question}\n\n"
        f"--- Bot's Response ---\n{bot_response}\n\n"
        f"--- Missing Information ---\n{missing_info}\n"
    )
    msg = MIMEText(body)
    msg["Subject"] = "Interview Bot - Missing Info Alert"
    msg["From"] = smtp_user
    msg["To"] = ALERT_EMAIL

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, ALERT_EMAIL, msg.as_string())
        return "Notification sent."
    except Exception:
        return "Notification failed."


# --- OpenAI tool schemas ---

TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "read_document",
        "description": "Read information about Ofer Brodatch. You MUST call this at least once before EVERY response — even if you think you already know the answer. Your memory of previous reads is unreliable. Never respond without reading first. Available files: " + list_documents(),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The filename to read (from the list in the tool description).",
                }
            },
            "required": ["filename"],
        },
    },
    {
        "type": "function",
        "name": "notify_missing_info",
        "description": (
            "Call this when the user asks something the documents don't cover. "
            "This emails Ofer about the gap. After calling this, tell the user you've emailed Ofer about it. "
            "Frame the missing_info as a clear question (e.g. 'What CI/CD tools does Ofer use?'). "
            "Include your drafted response in bot_response so it can be reviewed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_question": {
                    "type": "string",
                    "description": "The user's original question.",
                },
                "bot_response": {
                    "type": "string",
                    "description": "The response you plan to give to the user.",
                },
                "missing_info": {
                    "type": "string",
                    "description": "The information gap, phrased as a question.",
                },
            },
            "required": ["user_question", "bot_response", "missing_info"],
        },
    },
]

# Map function names to callables
TOOL_FUNCTIONS = {
    "list_documents": lambda **kwargs: list_documents(),
    "read_document": lambda **kwargs: read_document(kwargs["filename"]),
    "notify_missing_info": lambda **kwargs: notify_missing_info(
        kwargs["user_question"], kwargs["bot_response"], kwargs["missing_info"]
    ),
}
