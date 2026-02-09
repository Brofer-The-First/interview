import json
import os
import smtplib
import traceback
from email.mime.text import MIMEText
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

load_dotenv()

client = OpenAI()

ALERT_EMAIL = "ofer.brodatch@gmail.com"


def send_error_email(error):
    smtp_user = os.environ["SMTP_USER"]
    smtp_password = os.environ["SMTP_PASSWORD"]

    body = f"An error occurred in the Interview app:\n\n{traceback.format_exception(error)}"
    msg = MIMEText(body)
    msg["Subject"] = "Interview App - OpenAI API Error"
    msg["From"] = smtp_user
    msg["To"] = ALERT_EMAIL

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, ALERT_EMAIL, msg.as_string())


SYSTEM_PROMPT = """You are Ofer Brodatch, a full-stack software developer. You are being interviewed by someone who wants to learn about your background, skills, and experience.

You have tools to list and read documents that contain information about you. Use them to ground your answers in facts. On the first message, call list_documents to see what's available, then read relevant documents before answering.

Rules:
- Answer as Ofer, in first person.
- Be professional, personable, and concise.
- Only state things supported by the documents. If asked something not covered, say you'd be happy to discuss it further in a live conversation.
- Do not fabricate experiences, projects, or skills not mentioned in the documents."""


def chat(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": message})

    # Tool use loop
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOL_SCHEMAS,
            )
        except Exception as e:
            try:
                send_error_email(e)
            except Exception:
                pass
            return "An error has occurred while communicating with the OpenAI API. Please try again later."

        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            # Append the assistant message with tool calls
            messages.append(choice.message)

            # Execute each tool call and append results
            for tool_call in choice.message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                result = TOOL_FUNCTIONS[fn_name](**fn_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            # Loop back to get the next response
            continue

        # Regular text response — return it
        return choice.message.content


demo = gr.ChatInterface(
    fn=chat,
    title="Interview Ofer Brodatch",
    description="Ask me anything about my background, skills, and experience.",
    examples=[
        "Tell me about yourself.",
        "What is your professional experience?",
        "What technologies do you work with?",
        "Tell me about your education.",
        "What languages do you speak?",
    ],
)

if __name__ == "__main__":
    demo.launch()
