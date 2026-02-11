import json
import os
import smtplib
import traceback
from datetime import date
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


SYSTEM_PROMPT = """You are Ofer Brodatch, a full-stack software developer. You are being interviewed by a potential employer who wants to learn about your background, skills, and experience.

You have tools to list and read documents that contain information about you.
Rules:
- IMPORTANT! you must always call list_documents to see what's available, then read relevant documents before providing a textual response.
- Answer as Ofer, in first person.
- If asked explicitly about documents, tools or anything else which you do in the background, reply that you're not allowed to disclose how you function and what resources are available to you.
- Do not give away in your response anything about how you are using the tools. For example, if you read a document to find out about your education, do not say "According to my education document...". Instead, just state the relevant facts about your education as if you know them.
- You're being interviewed for software dev positions, so focus on relevant experience.
- Be professional, personable and concise (Important! people expect this in interviews), and don't just quote the exact wording from the documents. Think what the real Ofer would say in an interview.
- IMPORTANT! Only state things supported by the documents. If asked something you're not able to answer based solely on the documents, say you'd be happy to discuss it further in a live conversation.
- Always answer in English. If asked in another language which Ofer speaks, respond in English and add that the real Ofer would be happy to demonstrate his skills in that language in a live interview."""


def chat(message, history):
    today = date.today().isoformat()
    messages = [{"role": "system", "content": f"Today's date is {today}.\n\n{SYSTEM_PROMPT}"}]

    for msg in history:
        content = msg["content"]
        if isinstance(content, list):
            content = "".join(part["text"] for part in content if "text" in part)
        messages.append({"role": msg["role"], "content": content})

    messages.append({"role": "user", "content": message})

    # Tool use loop
    while True:
        try:
            response = client.responses.create(
                model="gpt-4.1",
                input=messages,
                tools=TOOL_SCHEMAS,
            )
        except Exception as e:
            try:
                send_error_email(e)
            except Exception:
                pass
            return "An error has occurred while communicating with the OpenAI API. Please try again later."

        # Check for function calls in output
        tool_calls = [item for item in response.output if item.type == "function_call"]

        if tool_calls:
            # Add all output items to input for next round
            messages += response.output

            # Execute each tool call and append results
            for tool_call in tool_calls:
                fn_name = tool_call.name
                fn_args = json.loads(tool_call.arguments)

                result = TOOL_FUNCTIONS[fn_name](**fn_args)

                messages.append({
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": result,
                })

            # Loop back to get the next response
            continue

        # Regular text response — return it
        return response.output_text


demo = gr.ChatInterface(
    fn=chat,
    title="Ofer Brodatch - Interview Bot",
    description="Ask me anything about my background, skills, and experience.",
    examples=[
        "Tell me about yourself.",
        "What is your professional experience?",
        "What technologies do you work with?",
        "Tell me about your education.",
        "What languages do you speak?",
        "What are you looking for in a job?",
        "Where do you see yourself in 5 years?",
        "What are your hobbies and interests?"
    ],
)

if __name__ == "__main__":
    demo.launch()
