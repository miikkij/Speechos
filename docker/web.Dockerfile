# ============================================================
# Speechos Web: Node.js frontend
# ============================================================
# Multi-stage: install → build → serve
# ============================================================

FROM node:22-alpine AS deps

RUN corepack enable pnpm

WORKDIR /app
COPY web/package.json web/pnpm-lock.yaml* ./
RUN pnpm install --frozen-lockfile 2>/dev/null || pnpm install

# --- Build ---
FROM deps AS build

COPY web/ ./
RUN pnpm build

# --- Runtime ---
FROM node:22-alpine AS runtime

RUN corepack enable pnpm

WORKDIR /app

COPY --from=build /app/package.json ./
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/.next ./.next
COPY --from=build /app/public ./public

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD wget -q --spider http://localhost:36301/ || exit 1

EXPOSE 36301

CMD ["pnpm", "start", "-p", "36301"]
