from langchain.tools import tool
from tavily import TavilyClient



from dotenv import load_dotenv
import ast
import requests
import operator
import os
load_dotenv()

tavily = TavilyClient(
    api_key = os.getenv('TAVILY_API_KEY'    )
)

gmail_session = None

@tool
def get_weather(city: str):
    """Get the current weather for a city."""

    # 1. Geocode city
    geo_response = requests.get(
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
    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m"
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
def search_web(search_statement:str):
    """Use this tool if you want to search anything from web"""

    response = tavily.search(
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


def send_mail(session,to:str,subject:str,content:str):
    """Sends an Email using Gmail"""