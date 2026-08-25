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
from langgraph.store.memory import InMemoryStore

from tools import get_weather
from tools import calculator
from tools import search_web
from tools import send_mail

from additional_functions import finalize
from additional_functions import agent
from additional_functions import make_decision
from additional_functions import human_approval
from additional_functions import check_approval

from config import State

builder = StateGraph(State)

builder.add_node('agent',agent)
tool_node = ToolNode([
    get_weather,
    calculator,
    search_web,
    send_mail
])
builder.add_node('approval',human_approval)
builder.add_node('tool',tool_node)
builder.add_node('final',finalize)

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
builder.add_edge('final',END)
builder.add_edge('tool','agent')

user_id = "Zeeshan"
thread_id = "thread1"

store = InMemoryStore()
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:

    graph = builder.compile(
        checkpointer=checkpointer
        store=store
    )

    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id" : user_id
        }
    }

    query = input("Ask Anything: ")

    result = graph.invoke(
        {
            "messages": [
                ("user", query)
            ]
        },
        config
    )

    state = graph.get_state(config)

    if state.tasks:

        answer = input("Approve this Email? ")

        result = graph.invoke(
            Command(resume=answer),
            config
        )

    print(result)