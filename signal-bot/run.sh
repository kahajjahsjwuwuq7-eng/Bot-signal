#!/usr/bin/env bash
# run.sh — Portable launcher for the Quotex Signal Bot.
# Locates python3/pip regardless of Nix store hash.
set -e

# ─── Find python3 ────────────────────────────────────────────────────────────
PYTHON=""
for candidate in \
    "$(command -v python3.11 2>/dev/null)" \
    "$(command -v python3 2>/dev/null)" \
    "$(command -v python 2>/dev/null)"
do
    if [ -x "$candidate" ]; then
        PYTHON="$candidate"
        break
    fi
done

# Fallback: scan known Nix wrapper locations
if [ -z "$PYTHON" ]; then
    for dir in /nix/store/*python-wrapped*/bin; do
        if [ -x "$dir/python3" ]; then
            PYTHON="$dir/python3"
            break
        fi
    done
fi

if [ -z "$PYTHON" ]; then
    echo "❌ python3 not found. Install Python 3.11 first."
    exit 1
fi

echo "✅ Using Python: $PYTHON ($($PYTHON --version 2>&1))"

# ─── Find pip ────────────────────────────────────────────────────────────────
PIP=""
for candidate in \
    "$(command -v pip3 2>/dev/null)" \
    "$(command -v pip 2>/dev/null)"
do
    if [ -x "$candidate" ]; then
        PIP="$candidate"
        break
    fi
done

# Fallback: use python -m pip
if [ -z "$PIP" ]; then
    PIP="$PYTHON -m pip"
fi

echo "✅ Using pip: $PIP"

# ─── Install dependencies ────────────────────────────────────────────────────
echo "📦 Installing dependencies..."
$PIP install -r requirements.txt -q --disable-pip-version-check

# ─── Launch bot ──────────────────────────────────────────────────────────────
echo "🚀 Starting Quotex Signal Bot..."
exec $PYTHON telegram_bot.py
