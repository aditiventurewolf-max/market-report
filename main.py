import asyncio
import sys
from agent import run_agent

if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"
    report = asyncio.run(run_agent(f"Research {ticker} for me"))
    print(report)
    