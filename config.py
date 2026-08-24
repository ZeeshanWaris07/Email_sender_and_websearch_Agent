from typing import TypedDict,Annotated
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from tools import get_weather
from tools import calculator
from tools import search_web
from tools import send_mail

class State(TypedDict):
    messages : Annotated[list,add_messages]
    approval : str

class FinalResponse(BaseModel):
    answer:str
    tools_used:list[str]


llm = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash"
)

llm_with_tools = llm.bind_tools([
    get_weather,
    calculator,
    search_web,
    send_mail
])

llm_with_structured_output = llm.with_structured_output(FinalResponse)