#!/usr/bin/env bash
# ============================================================
# Speechos Dev Launcher (Linux / WSL2 / macOS)
# ============================================================
# Starts API + Web dev servers with cleanup on exit.
#
# Usage:
#   ./dev.sh              # Start both API and Web
#   ./dev.sh --api        # API only
#   ./dev.sh --web        # Web only
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

API_ONLY=false
WEB_ONLY=false
PIDS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --api)  API_ONLY=true; shift ;;
        --web)  WEB_ONLY=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--api | --web]"
            echo "  --api   Start only the API server"
            echo "  --web   Start only the Web frontend"
            exit 0
            ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

cleanup() {
    echo ""
    echo -e "${YELLOW}[dev] Shutting down...${NC}"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null
    echo -e "${YELLOW}[dev] All processes stopped.${NC}"
}

trap cleanup EXIT INT TERM

# Kill anything already on our ports
for port in 36300 36301; do
    pid=$(lsof -ti ":$port" 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo -e "${YELLOW}[dev] Killing existing process on port $port (PID $pid)${NC}"
        kill "$pid" 2>/dev/null || true
    fi
done

# Load .env if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo -e "${CYAN}[dev] Loaded .env${NC}"
fi

# Start API
if [ "$WEB_ONLY" = false ]; then
    echo -e "${BLUE}[dev] Starting API on http://localhost:36300${NC}"
    (cd api && uv run python -m src) 2>&1 | sed "s/^/[api] /" &
    PIDS+=($!)
fi

# Start Web
if [ "$API_ONLY" = false ]; then
    echo -e "${GREEN}[dev] Starting Web on http://localhost:36301${NC}"
    (cd web && pnpm dev) 2>&1 | sed "s/^/[web] /" &
    PIDS+=($!)
fi

echo -e "${CYAN}[dev] Press Ctrl+C to stop all.${NC}"
echo ""

# Wait for any child to exit
wait -n 2>/dev/null || true
echo -e "${RED}[dev] A process exited unexpectedly.${NC}"
