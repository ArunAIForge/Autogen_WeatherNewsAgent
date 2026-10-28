"""
app.py - AutoGen Weather News Agent Streamlit Frontend

Interactive Streamlit UI with dynamic State and City dropdowns and strict
geographic city validation, querying an AutoGen AssistantAgent equipped with
DuckDuckGo keyless search.
"""

import asyncio
import os
import re
from pathlib import Path
import streamlit as st

# Load environment variables from .env file (with override=True)
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
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

# Free keyless search via ddgs
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None


# State and City dataset for interactive cascading dropdowns
LOCATION_DATA = {
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem", "Tirunelveli", "Vellore", "Erode"],
    "Karnataka": ["Bengaluru", "Mysuru", "Mangaluru", "Hubballi", "Belagavi", "Shivamogga"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad", "Solapur"],
    "Delhi NCR": ["New Delhi", "Noida", "Gurugram", "Faridabad", "Ghaziabad"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Karimnagar", "Khammam"],
    "Kerala": ["Kochi", "Thiruvananthapuram", "Kozhikode", "Thrissur", "Kollam"],
    "West Bengal": ["Kolkata", "Howrah", "Siliguri", "Durgapur", "Asansol"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Agra", "Prayagraj", "Meerut"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Bikaner", "Ajmer"],
    "California (USA)": ["Los Angeles", "San Francisco", "San Diego", "San Jose", "Sacramento"],
    "New York (USA)": ["New York City", "Buffalo", "Rochester", "Albany", "Syracuse"],
    "England (UK)": ["London", "Manchester", "Birmingham", "Liverpool", "Leeds"],
    "Tokyo Prefecture (Japan)": ["Tokyo", "Shinjuku", "Shibuya", "Hachioji"],
}


def is_plausible_city_name(name: str) -> bool:
    """Pre-filters obvious non-city inputs like numbers, symbols, or keyboard mashing."""
    name = name.strip()
    if len(name) < 2 or len(name) > 60:
        return False
    # Must contain letters, allows spaces, hyphens, and apostrophes (e.g. "Winston-Salem", "St. John's")
    if not re.search(r"[a-zA-Z]", name):
        return False
    if re.search(r"[0-9!@#$%^&*()_+={}\[\]:;\"<>?/\\|~`]", name):
        return False
    return True


def search_weather_news(query: str) -> str:
    """
    Search the web for current weather conditions, forecasts, and live weather news.

    Use this tool ONLY to look up real-time weather updates, local meteorological news,
    temperature reports, storm warnings, or daily forecasts for a specific city or region.
    Do NOT use this tool for general knowledge, history, historical weather records, or trivia.
    """
    if DDGS is None:
        return "Error: Search package not installed. Run `pip install ddgs`."

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return f"NO_DATA_FOUND: No current weather news found for query: '{query}'."

        formatted_snippets = []
        for i, item in enumerate(results, start=1):
            title = item.get("title", "").strip()
            body = item.get("body", "").strip()
            formatted_snippets.append(f"[{i}] {title}\n{body}")

        return "\n\n".join(formatted_snippets)
    except Exception as exc:
        return f"Encountered an error while querying weather search: {exc}"


async def run_weather_agent(city: str, state: str) -> str:
    """Initializes and runs the AutoGen weather news agent with geographic validation."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or "your-key" in api_key:
        raise ValueError("Valid OPENAI_API_KEY not found in .env file or environment.")

    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    model_client = OpenAIChatCompletionClient(
        model=model_name,
        api_key=api_key,
    )

    try:
        agent = AssistantAgent(
            name="weather_news_agent",
            model_client=model_client,
            tools=[search_weather_news],
            reflect_on_tool_use=True,
            system_message=(
                "You are an expert meteorological reporter with strict geographic verification.\n\n"
                "STEP 1: Validate the location.\n"
                "Check if the requested location is a real, recognized city, town, or municipality.\n"
                "- If the input is fictional (e.g., 'Atlantis', 'Gotham'), nonsensical, gibberish (e.g., 'asdfgh'), "
                "or not a real geographic settlement, do NOT search and do NOT fabricate weather.\n"
                "  Reply strictly with: INVALID_LOCATION: '<input>' is not a recognized real-world city.\n\n"
                "STEP 2: Search and Report.\n"
                "- If the city is valid, use `search_weather_news` to fetch today's weather news.\n"
                "- If the search indicates the location cannot be resolved meteorologically, reply:\n"
                "  INVALID_LOCATION: Unable to locate active weather records for '<input>'.\n"
                "- If valid results are returned, synthesize them into EXACTLY 3 short lines:\n"
                "  Line 1: Current condition and temperature\n"
                "  Line 2: Today's forecast or short-term outlook\n"
                "  Line 3: Notable weather alert, advisory, or key weather news headline\n"
                "Do not include greeting, pleasantries, markdown headers, or extra lines."
            ),
        )

        location_query = f"{city}, {state}" if state and state != "Other / Custom" else city
        task_prompt = f"Verify if '{location_query}' is a valid real-world city. If valid, provide today's weather news and forecast. If invalid, reject it."
        result = await agent.run(task=task_prompt)

        return str(result.messages[-1].content)
    finally:
        await model_client.close()


# Streamlit UI Configuration
st.set_page_config(
    page_title="AutoGen Weather News Agent",
    page_icon="⛅",
    layout="centered",
)

st.title("⛅ AutoGen Weather News Agent")
st.caption("Validates city authenticity before querying AutoGen 0.4 `AssistantAgent` + DuckDuckGo (`ddgs`)")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("Reads API key automatically from your local `.env` file.")
    env_key_found = bool(os.environ.get("OPENAI_API_KEY") and "your-key" not in os.environ.get("OPENAI_API_KEY", ""))
    if env_key_found:
        st.success("API Key detected in environment!")
    else:
        st.warning("No valid API Key detected. Please ensure `.env` contains `OPENAI_API_KEY=...`.")

# Main Interactive Dropdown Interface
st.subheader("📍 Select Location")

col1, col2 = st.columns(2)

with col1:
    state_options = list(LOCATION_DATA.keys()) + ["Other / Custom"]
    selected_state = st.selectbox(
        "State / Region:",
        options=state_options,
        index=0,  # Default to Tamil Nadu
    )

with col2:
    if selected_state != "Other / Custom":
        city_options = LOCATION_DATA[selected_state] + ["Custom City..."]
        selected_city = st.selectbox("City:", options=city_options, index=0)
        if selected_city == "Custom City...":
            target_city = st.text_input("Enter Custom City Name:", placeholder="e.g. Tiruppur")
        else:
            target_city = selected_city
    else:
        target_city = st.text_input("Enter City Name:", placeholder="e.g. Sydney, Australia")

st.divider()

# Action Button
if st.button("🔍 Get Weather News", type="primary", use_container_width=True):
    clean_city = target_city.strip() if target_city else ""
    
    # Tier 1 Validation: Syntactic sanity check on custom input
    if not clean_city:
        st.error("⚠️ Please select or enter a city name.")
    elif selected_state == "Other / Custom" and not is_plausible_city_name(clean_city):
        st.error(f"❌ '{clean_city}' is not a valid city name. Please enter a valid geographic city without numbers or special symbols.")
    else:
        with st.spinner(f"Validating location and researching weather news for {clean_city}..."):
            try:
                report = asyncio.run(run_weather_agent(clean_city, selected_state))

                # Tier 2 Validation: Model/geographic rejection check
                if "INVALID_LOCATION:" in report:
                    error_msg = report.split("INVALID_LOCATION:", 1)[1].strip()
                    st.error(f"❌ {error_msg}")
                else:
                    st.success(f"Weather News for **{clean_city}**:")
                    lines = [line.strip() for line in report.strip().split("\n") if line.strip()]
                    
                    if len(lines) >= 3:
                        st.markdown(f"🌡️ **Condition & Temp:** {lines[0]}")
                        st.markdown(f"📅 **Forecast:** {lines[1]}")
                        st.markdown(f"⚠️ **Alerts & News:** {lines[2]}")
                    else:
                        st.write(report)

            except Exception as e:
                st.error(f"Error running agent: {e}")