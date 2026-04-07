import asyncio
import json
import os
from groq import Groq


from dotenv import load_dotenv
from tools import (
    get_earnings,
    get_news_sentiment,
    get_sec_filings,
    get_price_history,
)

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_earnings",
            "description": "Fetch recent earnings data, EPS history, revenue, PE ratio and profit margins for a stock ticker. Use this first for any research task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker e.g. AAPL, MSFT, TSLA"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_news_sentiment",
            "description": "Fetch recent news headlines and score their sentiment for a company. Returns individual articles and an aggregate sentiment score.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "company_name": {"type": "string", "description": "Full company name e.g. Apple Inc"}
                },
                "required": ["ticker", "company_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_sec_filings",
            "description": "Fetch recent 10-K and 10-Q SEC filings metadata from EDGAR for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_history",
            "description": "Fetch 52-week price range, RSI, volume and position in range for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"}
                },
                "required": ["ticker"]
            }
        }
    },
]
TOOL_MAP = {
    "get_earnings": get_earnings,
    "get_news_sentiment": get_news_sentiment,
    "get_sec_filings": get_sec_filings,
    "get_price_history": get_price_history,
}

SYSTEM_PROMPT = """You are a financial research analyst. When given a ticker or company to research:

1. Always call get_earnings first to establish fundamentals
2. Call get_news_sentiment and get_price_history in parallel (you can request both in one response)
3. Call get_sec_filings to check recent regulatory filings
4. Synthesize everything into a structured report with sections:
   - Company snapshot
   - Earnings quality
   - Market sentiment
   - Technical position
   - Key risks
   - One-line verdict

Be concise and data-driven. Always cite specific numbers."""

async def run_agent(query: str) -> str:
    messages = [{"role": "user", "content": query}]
    
    print(f"\n🔍 Starting research: {query}\n")
    
    while True:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=4096,
            tools=TOOLS,
            tool_choice="auto",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        )
        
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        
        # ── Done? ─────────────────────────────────────────────
        if finish_reason == "stop":
            print("\n✅ Research complete\n")
            return message.content
        
        # ── Wants to call tools ───────────────────────────────
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in (message.tool_calls or [])
            ]
        })
        
        # ── Run all tools in parallel ─────────────────────────
        async def execute(tool_call):
            fn = TOOL_MAP[tool_call.function.name]
            args = json.loads(tool_call.function.arguments)  # note: string → dict
            print(f"⚙️  Calling: {tool_call.function.name}({args})")
            result = await fn(**args)
            return tool_call.id, result
        
        results = await asyncio.gather(*[execute(tc) for tc in message.tool_calls])
        
        # ── Feed results back ─────────────────────────────────
        for call_id, result in results:
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(result),
            })