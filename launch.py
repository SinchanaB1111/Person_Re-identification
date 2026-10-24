"""Start the web app and open it in the browser. Used by START_HERE_Windows.bat and run_app.sh.

Runs Streamlit headless (so it never stops to ask for an e-mail on first launch), picks a free port,
waits until the server answers, then opens the default browser.
"""
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def free_port(start=8501):
    for port in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def main():
    port = free_port()
    url = f"http://localhost:{port}"
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                             "--server.port", str(port), "--browser.gatherUsageStats", "false"], cwd=ROOT)
    print("Starting Person Search (the first start can take up to a minute) ...", flush=True)
    for _ in range(240):
        if proc.poll() is not None:
            print("The app stopped unexpectedly. See the messages above.")
            return proc.returncode or 1
        try:
            urllib.request.urlopen(f"{url}/_stcore/health", timeout=2)
            break
        except OSError:
            time.sleep(1)
    webbrowser.open(url)
    print(f"\nPerson Search is running at {url}\nKeep this window open while you use the app. "
          f"Close it to stop the app.\n", flush=True)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 0


if __name__ == "__main__":
    sys.exit(main())
