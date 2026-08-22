from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from main import State
import os

from config import llm_with_structured_output,llm_with_tools

SCOPES = [
    "https://www.googleapis.com/auth/gmail.send"
]

def get_creds():
    creds = None

    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file(
           'token.json',
           SCOPES 
        )

    if not creds or not creds.valid:

        if creds and creds.expired and creds.refresh_token:
            creds.refresh_token(Request())

        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json',
                SCOPES
            )

            creds = flow.run_local_server(port = 0)


    with open("token.json", "w") as token:
        token.write(creds.to_json())

    return creds


def make_decision(state:State):
    last_message = state['messages'][-1]

    if last_message.tool_calls:
        return 'tool'

    return 'final'


def agent(state:State):
    messages = [
            SystemMessage(
                content="""
                You are a helpful research assistant.

                Use tools when necessary.

                Never send an email without human approval.
                If an email needs to be sent, prepare the send_email
                tool call, but the application will require approval
                before executing it.

                Do not mention internal tool calls to the user.
                """
            )
        ] + state["messages"]


    response = llm_with_tools.invoke(
        messages
    )

    return {
        'messages' : [response]
    }


    
def finalize(state: State):

    messages = [
        SystemMessage(
            content="""
            Give the user a clear and concise final answer.

            Use the information from the conversation and tool results.

            Do not mention internal tool calls, LangGraph, MCP,
            or implementation details.
            """
        )
    ] + state["messages"]

    response = llm_with_structured_output.invoke(messages)

    return {
        "messages": [
            AIMessage(
                content=response.answer
            )
        ]
    }