"""Run the whole pipeline in order.  python src/run_all.py [--skip-weather]

Weather is downloaded only if data/raw/weather_daily.csv is missing (or never with --skip-weather).
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = ["etl", "olap", "classify", "cluster", "associate", "forecast", "lockdown", "diwali"]


def run(step):
    print(f"\n######## {step} ########", flush=True)
    subprocess.run([sys.executable, str(SRC / f"{step}.py")], check=True)


if __name__ == "__main__":
    if "--skip-weather" not in sys.argv and not (SRC.parent / "data" / "raw" / "weather_daily.csv").exists():
        run("fetch_weather")
    for step in STEPS:
        run(step)
    print("\nDone. Start the dashboard with:  streamlit run dashboard/app.py")
