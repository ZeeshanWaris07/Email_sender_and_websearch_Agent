from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from langgraph.runtime import Runtime
from config import State
from langgraph.types import interrupt
import os
import uuid

from config import memory_extraction_llm,llm_with_structured_output,llm_with_tools,user_id,thread_id

MAX_ITERATIONS = 5

def make_decision(state:State):

    iterations = state.get('num_iterations',0)

    if iterations > 5 :
        return 'loop_limit'

    else:
        last_message = state['messages'][-1]

        if not last_message.tool_calls:
            return 'final'

        for call in last_message.tool_calls:
            if call['name'] == "send_mail":
                return 'approval'

        return 'tool'


def agent(state: State,runtime:Runtime):

    iterations = state.get('num_iterations',0)

    iterations += 1

    user = runtime.context.user_id

    memories = runtime.store.search(
        ('users',user_id)
    )

    memory_text = "\n".join(
        str(memory.value)
        for memory in memories
    )

    messages = [
        SystemMessage(
            content=f"""
            You are a helpful research assistant.

            Use tools when necessary.

            User extracted memories from previous conversations : 
            {memory_text}
            """
        )
    ] + state["messages"] 

    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        print("\n--- TOOL CALLS ---")

        for call in response.tool_calls:
            print("Tool:", call["name"])
            print("Arguments:", call["args"])
            print("ID:", call["id"])

    return {
        "messages": [response],
        'num_iterations' : iterations
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
        ],
        'final_response' : response.model_dump()
    }


def human_approval(state:State):

    last_message = state['messages'][-1]

    email_call = None

    for call in last_message.tool_calls:
        if call['name'] == 'send_mail':
            email_call = call
            break

    if email_call == None:
        return {}

    args = email_call['args']

    decision = interrupt({
        'type' : 'email_approval',
        'to' : args['to'],
        'subject' : args['subject'],
        'content' : args['content'],
        'message' : "Do you approve sending this E-mail?"
    })

    return {
        'approval' : decision
    }


def check_approval(state: State):

    if state["approval"] == "yes":
        return "send"

    return "reject"

def memory_extraction(state:State,runtime:Runtime):

    processed = state.get("memory_processed", 0)

    new_messages = state['messages'][processed:]

    user_messages = [
        msg
        for msg in new_messages
        if msg.type == 'human'
    ]

    conversation = "\n".join(
        f"{msg.type}: {msg.content}"
        for msg in user_messages
    )

    prompt = f"""
    You are a memory extraction agent.

    Extract only information about the USER that is worth
    remembering for future conversations.

    Do not store:
    - temporary questions
    - one-time requests
    - tool results
    - weather information
    - casual conversation

    Store things such as:
    - preferences
    - goals
    - career information
    - projects
    - skills being learned
    - important personal preferences

    Conversation:

    {conversation}
    """

    response = memory_extraction_llm.invoke(prompt)

    if response.should_store:

        for memory in response.memories:
            runtime.store.put(
                ('users',runtime.context.user_id),
                str(uuid.uuid4()),
                {
                    'text' : memory
                }
            )

    return {
        'memory_processed' : len(state['messages'])
    }

def loop_limit(state:State):

    return {
        'loop_limit' : True
    }