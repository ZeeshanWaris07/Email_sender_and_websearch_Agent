from typing import TypedDict,Annotated
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore
from tools import get_weather
from tools import calculator
from tools import search_web
from tools import send_mail
from langgraph.store.postgres import PostgresStore

class State(TypedDict):
    messages : Annotated[list,add_messages]
    approval : str
    memory_processed:int

class FinalResponse(BaseModel):
    answer:str
    tools_used:list[str]

class MemoryExtraction(BaseModel):
    should_store:bool
    memories:list[str]

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

memory_extraction_llm = llm.with_structured_output(MemoryExtraction)

user_id = "Zeeshan"
thread_id = "thread2"