"""
weather_news.py - AutoGen Weather News Agent

An AutoGen-powered assistant that searches for live weather news and forecasts
for a user-specified city using DuckDuckGo (`ddgs`) and returns a concise
3-line report. Reads API keys automatically from a local .env file.
"""

import asyncio
import os
import sys
from pathlib import Path

# Load environment variables from .env file (with override=True to replace any stale shell variables)
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    # Lightweight fallback if python-dotenv is not installed
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

# AutoGen AgentChat and Model Client imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Free keyless search via ddgs (with fallback to duckduckgo_search)
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None


def search_weather_news(query: str) -> str:
    """
    Search the web for current weather conditions, forecasts, and live weather news.

    Use this tool ONLY to look up real-time weather updates, local meteorological news,
    temperature reports, storm warnings, or daily forecasts for a specific city or region.
    Do NOT use this tool for general knowledge, history, historical weather records, or trivia.

    Args:
        query: The targeted weather search query (e.g., 'Tokyo weather forecast today',
               'London current weather news').

    Returns:
        A formatted summary of top search snippets describing current weather and forecasts.
    """
    if DDGS is None:
        return (
            "Error: Search package not installed. Please run `pip install ddgs` "
            "or `pip install duckduckgo-search`."
        )

    try:
        with DDGS() as ddgs:
            # Query the web for live weather news and forecasts
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return f"No current weather news found for query: '{query}'."

        formatted_snippets = []
        for i, item in enumerate(results, start=1):
            title = item.get("title", "").strip()
            body = item.get("body", "").strip()
            formatted_snippets.append(f"[{i}] {title}\n{body}")

        return "\n\n".join(formatted_snippets)
    except Exception as exc:
        return f"Encountered an error while querying weather search: {exc}"


async def main() -> None:
    # 1. Verify API Key loaded from .env or system environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or "your-key" in api_key:
        print(
            "Error: OPENAI_API_KEY is still using the placeholder or is not set.\n"
            "Please open your .env file and ensure it contains your real OpenAI key:\n"
            "OPENAI_API_KEY=sk-proj-YOUR_ACTUAL_KEY_HERE",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. Initialize Model Client (defaulting to gpt-4o-mini for fast, cost-effective inference)
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    model_client = OpenAIChatCompletionClient(
        model=model_name,
        api_key=api_key,
    )

    try:
        # 3. Configure AssistantAgent with tools and reflection enabled
        agent = AssistantAgent(
            name="weather_news_agent",
            model_client=model_client,
            tools=[search_weather_news],
            reflect_on_tool_use=True,  # Crucial: enables second-pass inference to synthesize tool output
            system_message=(
                "You are an expert weather news reporter. "
                "When given a city name, use the `search_weather_news` tool to retrieve today's "
                "live weather news and forecast. "
                "After receiving search results, synthesize the findings and reply in EXACTLY 3 short lines:\n"
                "Line 1: Current condition and temperature\n"
                "Line 2: Today's forecast or short-term outlook\n"
                "Line 3: Notable weather alert, advisory, or key weather news headline\n"
                "Do not include greeting, pleasantries, markdown headers, or extra lines."
            ),
        )

        # 4. User interaction
        city = input("Enter city: ").strip()
        if not city:
            print("No city provided. Exiting.")
            return

        print(f"\nFetching weather news for {city}...\n")

        # 5. Execute task
        task_prompt = f"Provide today's weather news and forecast for {city}."
        result = await agent.run(task=task_prompt)

        # 6. Extract and display final response
        last_message = result.messages[-1]
        if isinstance(last_message, TextMessage):
            print(last_message.content)
        else:
            print(str(last_message.content))

    finally:
        # 7. Properly close model client lifecycle connection
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())