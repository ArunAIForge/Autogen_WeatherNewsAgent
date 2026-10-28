# AutoGen Weather News Agent ⛅

An AI-powered agent built with **Microsoft AutoGen 0.4** that searches for real-time weather news and forecasts using keyless **DuckDuckGo Search (`ddgs`)** and synthesizes findings into a concise, 3-line daily brief. Includes both a terminal CLI and an interactive **Streamlit** frontend with cascading State/City dropdowns and geographic validation.

---

## 🚀 Key Features

* **Tool-Using Agent**: Built on AutoGen's `AssistantAgent` where Python functions are automatically converted into LLM-callable tools using type hints and docstrings.
* **Keyless Web Search**: Utilizes `ddgs` (DuckDuckGo search) to fetch live weather reports, storm warnings, and local meteorological news with zero API keys or accounts required.
* **Reflective Synthesis (`reflect_on_tool_use=True`)**: Employs AutoGen's two-pass reasoning to analyze raw search snippets and synthesize an answer instead of dumping unformatted web text.
* **Strict 3-Line Output**:
  * **Line 1**: Current condition and temperature
  * **Line 2**: Today's forecast or short-term outlook
  * **Line 3**: Notable weather advisory, alert, or key news headline
* **Geographic City Validation**: Two-tier validation (instant regex pre-check + agent geographic verification) to reject fictional places (e.g., *Gotham*, *Atlantis*) or gibberish inputs.
* **Dual Interface**:
  * **CLI Mode (`weather_news.py`)**: Lightweight, single-file terminal workflow.
  * **Web UI (`app.py`)**: Modern Streamlit dashboard with cascading State/City dropdowns.

---

## 📁 Project Structure

```text
├── weather_news.py      # Terminal CLI agent (single-file requirement)
├── app.py               # Streamlit frontend with interactive dropdowns & validation
├── .env                 # API keys and environment configuration (do not commit)
├── .gitignore           # Ignores virtual env, secrets, and cache
└── README.md            # Project documentation
```

---

## 🛠️ Quickstart

### 1. Clone the repository & create a virtual environment
```bash
git clone https://github.com/your-username/autogen-weather-news-agent.git
cd autogen-weather-news-agent

# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

### 2. Install dependencies
```bash
python -m pip install ddgs autogen-agentchat "autogen-ext[openai]" python-dotenv streamlit
```

### 3. Configure API key
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=sk-proj-your_actual_key_here
OPENAI_MODEL=gpt-4o-mini
```

---

## 💻 Running the Project

### Option A: Terminal CLI
```bash
python weather_news.py
```
```text
Enter city: Chennai

Fetching weather news for Chennai...

Current: 31°C, Partly Cloudy with 78% humidity and light coastal winds.
Forecast: Isolated afternoon thundershowers expected with daytime highs near 33°C.
Alert: Regional IMD advisory in effect for brief coastal squalls and moderate rain.
```

### Option B: Interactive Streamlit Web UI
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`, pick your State and City from the dropdowns (or enter a custom city), and click **Get Weather News**.

---

## 🧠 Architectural Insights

### Why the Docstring is the Tool's Instruction Manual
In AutoGen, the agent inspects the tool's docstring to determine invocation intent. By providing explicit temporal scoping (*"real-time weather updates, local meteorological news, temperature reports"*) and negative scoping (*"Do NOT use this tool for general knowledge, history, or trivia"*), the model reliably knows when to invoke the tool and avoids triggering it for unrelated queries.

### The Role of `reflect_on_tool_use=True`
* **Without reflection (`False`)**: The agent stops immediately after tool execution and returns a `ToolCallSummaryMessage` containing the raw DuckDuckGo snippet dump.
* **With reflection (`True`)**: AutoGen feeds the tool result back into the model for a second inference pass, allowing the agent to read the search output and format it into the exact 3-line report requested.

---

## 📄 License
MIT
