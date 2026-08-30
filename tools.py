from langchain.tools import tool
from tavily import AsyncTavilyClient

from googleapiclient.discovery import build

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from email.message import EmailMessage
from dotenv import load_dotenv
import ast
import requests
import operator
import os
import asyncio
import httpx
import base64
load_dotenv()





tavily = AsyncTavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

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
            creds.refresh(Request())

        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json',
                SCOPES
            )

            creds = flow.run_local_server(port = 0)


    with open("token.json", "w") as token:
        token.write(creds.to_json())

    return creds


@tool
async def get_weather(city: str):
    """Get the current weather for a city."""

    async with httpx.AsyncClient() as client:

        # 1. Geocode city
        geo_response = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json"
            }
        )

        geo_data = geo_response.json()

        if "results" not in geo_data:
            return f"Could not find the city {city}"

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        # 2. Get weather
        weather_response = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "wind_speed_10m"
                )
            }
        )

        weather_data = weather_response.json()

        current = weather_data["current"]

        return {
            "city": city,
            "temperature": current["temperature_2m"],
            "humidity": current["relative_humidity_2m"],
            "wind_speed": current["wind_speed_10m"]
        }


@tool
async def search_web(search_statement:str):
    """Use this tool if you want to search anything from web"""

    response = await tavily.search(
        query = search_statement,
        max_results=5
    )

    results = []

    for result in response["results"]:
        results.append({
            "title": result["title"],
            "url": result["url"],
            "content": result["content"]
        })

    return results

@tool
def calculator(expression: str):
    """Calculate a mathematical expression."""

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }

    def calculate(node):

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Invalid number")

        if isinstance(node, ast.BinOp):
            left = calculate(node.left)
            right = calculate(node.right)

            operation = operators.get(type(node.op))

            if operation is None:
                raise ValueError("Unsupported operator")

            return operation(left, right)

        if isinstance(node, ast.UnaryOp):
            value = calculate(node.operand)

            operation = operators.get(type(node.op))

            if operation is None:
                raise ValueError("Unsupported operator")

            return operation(value)

        raise ValueError("Invalid expression")

    tree = ast.parse(expression, mode="eval")

    return calculate(tree.body)




def send_mail(to:str,subject:str,content:str):
    """Sends an Email using Gmail"""

    message = EmailMessage()

    message["To"] = to
    message["Subject"] = subject
    message.set_content(content)

    encoded_message = base64.urlsafe_b64encode(
        message.as_bytes()
    ).decode('utf-8')

    body = {
        'raw' : encoded_message
    }

    creds = get_creds()

    service = build(
        "gmail",
        "v1",
        credentials=creds
    )

    result = service.users().messages().send(
        userId = 'me',
        body = body
    ).execute()

    print("Email sent!")

    return "Email Sent successfully"