from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from langchain.tools import tool
from typing import Annotated,TypedDict
from pydantic import BaseModel

from tools import get_weather
from tools import calculator
from tools import search_web
from tools import send_mail

from additional_functions import finalize
from additional_functions import agent
from additional_functions import make_decision

from config import State


builder = StateGraph(State)

builder.add_node('agent',agent)
tool_node = ToolNode([
    get_weather,
    calculator,
    search_web,
    send_mail
])
builder.add_node('tool',tool_node)
builder.add_node('final',finalize)

builder.add_edge(START,'agent')
builder.add_conditional_edges(
    'agent',
    make_decision,
    {
        'final':'final',
        'tool':'tool'
    }
)
builder.add_edge('final',END)
builder.add_edge('tool','agent')

graph = builder.compile()

query = input("Ask Anything : ")

result = graph.invoke(
    {
        'messages' : [
            ('user' , query)
        ]
    }
)

print(result)