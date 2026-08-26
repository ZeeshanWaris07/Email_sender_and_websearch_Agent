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

from config import State,user_id,thread_id
from dataclasses import dataclass

import os


DB_UTL = os.getenv('DB_UTL')

@dataclass
class Context:
    user_id:str

context = Context(
    user_id=user_id
)

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
builder.add_edge('tool','agent')


with PostgresStore.from_conn_string(DB_UTL) as store:

    store.setup()


    with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    
        graph = builder.compile(
            checkpointer=checkpointer,
            store=store
        )
    
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id" : user_id
            }
        }
    
        while True:
        
            query = input("\nAsk Anything (type 'exit' to quit): ")
    
            if query.lower() == "exit":
                break
            
            result = graph.invoke(
                {
                    "messages": [
                        ("user", query)
                    ]
                },
                config,
                context=context
            )
    
            state = graph.get_state(config)
    
            if state.tasks:
            
                answer = input("Approve this Email? ")
    
                result = graph.invoke(
                    Command(resume=answer),
                    config,
                    context=context
                )
    
                state = graph.get_state(config)
    
            print(result)