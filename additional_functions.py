from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from langgraph.runtime import Runtime
from config import State,FinalResponse,ToolError
from langgraph.types import interrupt
import os
import uuid
import asyncio
from config import memory_extraction_llm,llm_with_structured_output,llm_with_tools,user_id,thread_id

MAX_ITERATIONS = 5
MAX_RETRIES = 3

def handle_tool_result(state: State, runtime: Runtime):

    last_message = state["messages"][-1]

    if last_message.type != "tool":
        return "agent"

    content = str(last_message.content)

    history = runtime.context.tool_call_history

    if not history:
        return "agent"

    # Most recent tool call
    latest_call = history[-1]

    try:
        error = ToolError.model_validate_json(content)

    except Exception:

        # No structured error means the tool succeeded

        latest_call["status"] = "success"

        runtime.context.retry_count = 0

        return "agent"


    latest_call["status"] = "failed"

    print(
        f"\nTool failed:"
        f"\nType: {error.error_type}"
        f"\nMessage: {error.message}"
        f"\nRetryable: {error.retryable}"
    )

    # Don't retry non-retryable errors
    if not error.retryable:

        runtime.context.retry_count = 0

        return "agent"

    # Retryable error
    runtime.context.retry_count += 1

    print(
        f"Retry {runtime.context.retry_count}/{MAX_RETRIES}"
    )

    if runtime.context.retry_count >= MAX_RETRIES:

        runtime.context.limit_reached = True

        return "final"

    return "agent"

def make_decision(state:State,runtime:Runtime):

    iterations = runtime.context.num_iterations

    repeated_calls = runtime.context.repeated_tools

    if repeated_calls:
        return 'final'

    if iterations > MAX_ITERATIONS :
        runtime.context.limit_reached = True
        return 'final'

    else:
        last_message = state['messages'][-1]

        if not last_message.tool_calls:
            return 'final'

        for call in last_message.tool_calls:
            if call['name'] == "send_mail":
                return 'approval'

        return 'tool'


async def agent(state: State, runtime: Runtime):

    runtime.context.num_iterations += 1

    user = runtime.context.user_id

    memories = await runtime.store.asearch(
        ("users", user)
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

            User extracted memories from previous conversations:
            {memory_text}
            """
        )
    ] + state["messages"]

    response = await llm_with_tools.ainvoke(messages)

    history = runtime.context.tool_call_history

    if response.tool_calls:

        print("\n--- TOOL CALLS ---")

        for call in response.tool_calls:

            tool_name = call["name"]
            tool_args = call["args"]

            tool_signature = (
                tool_name,
                str(sorted(tool_args.items()))
            )

            # Check previous calls
            for item in history:

                if item["signature"] == tool_signature:

                    if item["status"] == "success":
                        runtime.context.repeated_tool_call = True

                    # Failed calls are allowed to retry
                    break

            else:

                # New tool call
                history.append(
                    {
                        "signature": tool_signature,
                        "status": "pending"
                    }
                )

            print("Tool:", tool_name)
            print("Arguments:", tool_args)
            print("ID:", call["id"])

    return {
        "messages": [response]
    }

    
async def finalize(state: State, runtime: Runtime):

    if runtime.context.limit_reached:

        response = FinalResponse(
            answer=(
                "I wasn't able to complete the request within "
                "the allowed number of steps."
            ),
            tools_used=[]
        )

        return {
            "messages": [
                AIMessage(content=response.answer)
            ],
            "final_response": response.model_dump()
        }


    if runtime.context.repeated_tools:

        response = FinalResponse(
            answer=(
                "I stopped because the agent attempted to "
                "repeat the same tool operation."
            ),
            tools_used=[]
        )

        return {
            "messages": [
                AIMessage(content=response.answer)
            ],
            "final_response": response.model_dump()
        }


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

    response = await llm_with_structured_output.ainvoke(messages)

    return {
        "messages": [
            AIMessage(
                content=response.answer
            )
        ],
        "final_response": response.model_dump()
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

async def memory_extraction(state:State,runtime:Runtime):

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

    response = await memory_extraction_llm.ainvoke(prompt)

    if response.should_store:

        for memory in response.memories:
            await runtime.store.aput(
                ('users',runtime.context.user_id),
                str(uuid.uuid4()),
                {
                    'text' : memory
                }
            )

    return {
        'memory_processed' : len(state['messages'])
    }

def handle_tool_error(error: Exception) -> str:

    if isinstance(error, TimeoutError):

        return ToolError(
            error_type="temporary",
            message=str(error),
            retryable=True
        ).model_dump_json()

    if isinstance(error, ValueError):

        return ToolError(
            error_type="invalid_arguments",
            message=str(error),
            retryable=False
        ).model_dump_json()

    return ToolError(
        error_type="unknown",
        message=str(error),
        retryable=False
    ).model_dump_json()