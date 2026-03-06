#!/usr/bin/env bash
# exit on error
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt

# Install Playwright browser during build phase
python -m playwright install chromium
