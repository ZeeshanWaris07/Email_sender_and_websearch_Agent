import asyncio
import webbrowser
from urllib.parse import parse_qs, urlparse

import httpx2

from pydantic import AnyUrl

from mcp import ClientSession
from mcp.client.auth import (
    AuthorizationCodeResult,
    OAuthClientProvider,
    TokenStorage,
)
from mcp.client.streamable_http import streamable_http_client

from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)


# --------------------------------------------------
# Token Storage
# --------------------------------------------------

class InMemoryTokenStorage(TokenStorage):

    def __init__(self):
        self.tokens: OAuthToken | None = None
        self.client_info: OAuthClientInformationFull | None = None

    async def get_tokens(self) -> OAuthToken | None:
        return self.tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self.tokens = tokens

    async def get_client_info(
        self,
    ) -> OAuthClientInformationFull | None:
        return self.client_info

    async def set_client_info(
        self,
        client_info: OAuthClientInformationFull,
    ) -> None:
        self.client_info = client_info


# --------------------------------------------------
# OAuth Redirect Handler
# --------------------------------------------------

async def redirect_handler(authorization_url: str):

    print("\nOpening Google authorization page...")
    print(authorization_url)

    webbrowser.open(authorization_url)


# --------------------------------------------------
# OAuth Callback Handler
# --------------------------------------------------

async def callback_handler() -> AuthorizationCodeResult:

    redirect_url = input(
        "\nPaste the callback URL here:\n"
    )

    params = parse_qs(
        urlparse(redirect_url).query
    )

    return AuthorizationCodeResult(
        code=params["code"][0],
        state=params["state"][0],
        iss=params["iss"][0] if "iss" in params else None,
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

async def main():

    storage = InMemoryTokenStorage()

    oauth = OAuthClientProvider(
        server_url="https://gmailmcp.googleapis.com/mcp/v1",

        client_metadata=OAuthClientMetadata(
            client_name="My Gmail MCP Agent",

            redirect_uris=[
                AnyUrl("http://localhost:3030/callback")
            ],

            grant_types=[
                "authorization_code",
                "refresh_token",
            ],

            response_types=[
                "code"
            ],
        ),

        storage=storage,

        redirect_handler=redirect_handler,

        callback_handler=callback_handler,
    )

    async with httpx2.AsyncClient(
        auth=oauth,
        follow_redirects=True,
    ) as http_client:

        async with streamable_http_client(
            "https://gmailmcp.googleapis.com/mcp/v1",
            http_client=http_client,
        ) as (
            read_stream,
            write_stream,
            _
        ):

            async with ClientSession(
                read_stream,
                write_stream,
            ) as session:

                await session.initialize()

                print("\nConnected to Gmail MCP!")

                result = await session.list_tools()

                print("\nAvailable Gmail tools:")

                for tool in result.tools:
                    if tool.name == "create_draft":
                        print(tool.inputSchema)


if __name__ == "__main__":
    asyncio.run(main())