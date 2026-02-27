#!/usr/bin/env bash
# ============================================================
# Speechos Launcher
# ============================================================
# Detects hardware and launches the appropriate Docker stack.
#
# Usage:
#   ./start.sh              # Auto-detect (GPU if available)
#   ./start.sh --cpu        # Force CPU-only
#   ./start.sh --gpu        # Force GPU
#   ./start.sh --tier gpu-32gb  # Force specific tier
#   ./start.sh --dev        # Development mode (no Docker)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Speechos Launcher            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════╝${NC}"
echo ""

# Parse arguments
MODE="auto"
TIER=""
DEV=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)    MODE="cpu"; shift ;;
        --gpu)    MODE="gpu"; shift ;;
        --tier)   TIER="$2"; shift 2 ;;
        --dev)    DEV=true; shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cpu          Force CPU-only mode"
            echo "  --gpu          Force GPU mode"
            echo "  --tier TIER    Force specific hardware tier"
            echo "  --dev          Run in development mode (no Docker)"
            echo "  --help         Show this help"
            echo ""
            echo "Tiers: cpu-2gb, cpu-4gb, cpu-8gb, cpu-16gb, cpu-32gb"
            echo "       gpu-4gb, gpu-8gb, gpu-12gb, gpu-24gb, gpu-32gb"
            echo "       hybrid-4gb-gpu, hybrid-8gb-gpu, hybrid-12gb-gpu"
            echo "       hybrid-24gb-gpu, hybrid-32gb-gpu"
            exit 0
            ;;
        *)        echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Detect GPU
detect_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$GPU_NAME" ]; then
            echo -e "${GREEN}GPU detected:${NC} $GPU_NAME (${GPU_VRAM} MB VRAM)"
            return 0
        fi
    fi
    echo -e "${YELLOW}No GPU detected${NC}"
    return 1
}

# Detect RAM
RAM_MB=$(free -m 2>/dev/null | awk '/^Mem:/ {print $2}' || echo "8192")
RAM_GB=$((RAM_MB / 1024))
echo -e "${GREEN}System RAM:${NC} ${RAM_GB} GB"

# Determine compose file
if [ "$MODE" = "auto" ]; then
    if detect_gpu; then
        COMPOSE_FILE="docker/docker-compose.gpu.yml"
        echo -e "${GREEN}Mode:${NC} GPU (auto-detected)"
    else
        COMPOSE_FILE="docker/docker-compose.cpu.yml"
        echo -e "${GREEN}Mode:${NC} CPU-only (auto-detected)"
    fi
elif [ "$MODE" = "gpu" ]; then
    if detect_gpu; then
        COMPOSE_FILE="docker/docker-compose.gpu.yml"
        echo -e "${GREEN}Mode:${NC} GPU (forced)"
    else
        echo -e "${RED}ERROR: GPU mode requested but no GPU found${NC}"
        exit 1
    fi
else
    COMPOSE_FILE="docker/docker-compose.cpu.yml"
    echo -e "${GREEN}Mode:${NC} CPU-only (forced)"
fi

# Set tier override
TIER_ENV=""
if [ -n "$TIER" ]; then
    TIER_ENV="-e SPEECHOS_TIER=$TIER"
    echo -e "${GREEN}Tier:${NC} $TIER (forced)"
else
    echo -e "${GREEN}Tier:${NC} auto-detect at startup"
fi

echo ""

if [ "$DEV" = true ]; then
    echo -e "${BLUE}Starting in development mode...${NC}"
    echo ""

    # Create directories
    mkdir -p models recordings samples

    # Start API in background
    echo -e "${YELLOW}Starting API server...${NC}"
    cd api && uv run python -m src.server &
    API_PID=$!
    cd ..

    # Start web frontend
    echo -e "${YELLOW}Starting web frontend...${NC}"
    cd web && pnpm dev &
    WEB_PID=$!
    cd ..

    echo ""
    echo -e "${GREEN}Speechos running in dev mode:${NC}"
    echo -e "  Web:  http://localhost:36301"
    echo -e "  API:  http://localhost:36300"
    echo -e "  Docs: http://localhost:36300/docs"
    echo ""
    echo -e "Press Ctrl+C to stop"

    trap "kill $API_PID $WEB_PID 2>/dev/null" EXIT
    wait
else
    echo -e "${BLUE}Starting Docker containers...${NC}"
    echo ""

    # Create volumes dirs
    mkdir -p models recordings samples

    docker compose -f "$COMPOSE_FILE" up --build -d

    echo ""
    echo -e "${GREEN}Speechos is running:${NC}"
    echo -e "  App:  http://localhost (via nginx)"
    echo -e "  Web:  http://localhost:36301 (direct)"
    echo -e "  API:  http://localhost:36300 (direct)"
    echo -e "  Docs: http://localhost:36300/docs (Swagger)"
    echo ""
    echo -e "Logs: docker compose -f $COMPOSE_FILE logs -f"
    echo -e "Stop: docker compose -f $COMPOSE_FILE down"
fi
