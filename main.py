from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from langchain.tools import tool
from typing import Annotated,TypedDict
from pydantic import BaseModel
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from langgraph.store.postgres import PostgresStore
from langgraph.errors import GraphRecursionError

from tools import get_weather
from tools import calculator
from tools import search_web
from tools import send_mail

from additional_functions import finalize
from additional_functions import agent
from additional_functions import make_decision
from additional_functions import human_approval
from additional_functions import check_approval
from additional_functions import memory_extraction
from additional_functions import handle_tool_result
from additional_functions import handle_tool_error

from config import State,user_id,thread_id
from dataclasses import dataclass,field

import os

def display_event(event):

    for node_name, node_output in event.items():

        print(f"\n[{node_name.upper()}]")

        if not isinstance(node_output, dict):
            print(node_output)
            continue

        messages = node_output.get("messages", [])

        for message in messages:

            if message.type == "ai":

                if message.tool_calls:

                    for call in message.tool_calls:

                        print(f"Tool: {call['name']}")
                        print(f"Arguments: {call['args']}")

                elif message.content:

                    print(f"AI: {message.content}")

            elif message.type == "tool":

                print(f"Tool Result: {message.content}")

DB_UTL = os.getenv('DB_UTL')

@dataclass
class Context:
    user_id: str
    tool_call_history: list[dict] = field(default_factory=list)
    repeated_tools: bool = False
    num_iterations: int = 0
    limit_reached: bool = False
    retry_count: int = 0

builder = StateGraph(State)

builder.add_node('agent',agent)
tool_node = ToolNode([
    get_weather,
    calculator,
    search_web,
    send_mail
],
handle_tool_errors = handle_tool_error
)
builder.add_node('approval',human_approval)
builder.add_node('tool',tool_node)
builder.add_node('final',finalize)
builder.add_node('memory',memory_extraction)

builder.add_edge(START,'agent')
builder.add_conditional_edges(
    'agent',
    make_decision,
    {
        'final':'final',
        'tool':'tool',
        'approval' : 'approval'
    }
)
builder.add_conditional_edges(
    'approval',
    check_approval,
    {
        'send' : 'tool',
        'reject' : 'final'
    }
)
builder.add_edge('final','memory')
builder.add_edge('memory',END)
builder.add_conditional_edges(
    'tool',
    handle_tool_result,
    {
        'agent' : 'agent',
        'final' : 'final'
    }
)


with PostgresStore.from_conn_string(DB_UTL) as store:


    store.setup()

    memories = store.search(
        ('users',user_id)
    )

    for memory in memories:
        print(memory.value)


    with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    
        graph = builder.compile(
            checkpointer=checkpointer,
            store=store
        )
    
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id" : user_id
            },
            'recursion_limit' : 10
        }
    
        while True:
            
            query = input("\nAsk Anything (type 'exit' to quit): ")

            if query.lower() == "exit":
                break

            context = Context(
                user_id=user_id,
            )

            try:

                result = None

                for event in graph.stream(
                    {
                        'messages' : [
                            ('user',query)
                        ]
                    },
                    config,
                    context=context
                ):
                    
                    display_event(event)

                state = graph.get_state(config)
                result = state.values
                if state.tasks:
                
                    answer = input("Approve this Email? ")

                    for event in graph.stream(
                        Command(resume=answer),
                        config,
                        context=context
                    ):
                        display_event(event)

                response = result["final_response"]

                print(f"AI : {response['answer']}")
                print(f"Tools used : {response['tools_used']}")

            except GraphRecursionError:

                print("\nAI : I wasn't able to complete the request within")
                print("     the allowed number of steps.")