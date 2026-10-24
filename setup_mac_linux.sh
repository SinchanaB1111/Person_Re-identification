#!/bin/sh
# macOS / Linux: one-time setup
cd "$(dirname "$0")"
python3.11 -m venv .venv || python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
