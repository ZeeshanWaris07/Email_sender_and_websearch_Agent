from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from config import State
import os

from config import llm_with_structured_output,llm_with_tools


def make_decision(state:State):
    last_message = state['messages'][-1]

    if last_message.tool_calls:
        return 'tool'

    return 'final'


def agent(state: State):

    messages = [
        SystemMessage(
            content="""
            You are a helpful research assistant.

            Use tools when necessary.
            """
        )
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)

    print("\n--- TOOL CALLS ---")

    for call in response.tool_calls:
        print("Tool:", call["name"])
        print("Arguments:", call["args"])
        print("ID:", call["id"])

    return {
        "messages": [response]
    }

    
def finalize(state: State):

    messages = [
        SystemMessage(
            content="""
            You are a helpful research assistant.

            Give the user a clear and concise final answer.

            Use the information from the conversation and tool results.

            Do not mention internal tool calls, LangGraph, MCP,
            or implementation details.
            """
        )
    ] + state["messages"] + [
        HumanMessage(
            content="Now provide the final answer to the user's request."
        )
    ]

    response = llm_with_structured_output.invoke(messages)

    return {
        "messages": [
            AIMessage(
                content=response.answer
            )
        ]
    }