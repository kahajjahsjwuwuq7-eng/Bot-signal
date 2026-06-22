#!/bin/bash
# Replit-compatible launcher — sets up Nix PATH then runs the bot
export PATH="/home/runner/workspace/.pythonlibs/bin:/nix/store/flbj8bq2vznkcwss7sm0ky8rd0k6kar7-python-wrapped-0.1.0/bin:/nix/store/xwg0ddq9mjf6ibwdvp93jsp0cf51z3xr-pip-wrapper/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONUSERBASE="/home/runner/workspace/.pythonlibs"
export PYTHONPATH="/home/runner/workspace/.pythonlibs/lib/python3.11/site-packages"
export UV_PROJECT_ENVIRONMENT="/home/runner/workspace/.pythonlibs"

cd "$(dirname "$0")"

echo "Python: $(python3 --version 2>&1)"
echo "Pip: $(pip --version 2>&1)"
echo "Installing dependencies..."
pip install -r requirements.txt -q --disable-pip-version-check

echo "Starting bot..."
exec python3 telegram_bot.py
