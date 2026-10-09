# ============================================================
# Speechos Web — Node.js frontend
# ============================================================
# Multi-stage: install → build → serve (port 36301)
# ============================================================

FROM node:22-alpine AS deps

RUN corepack enable pnpm

WORKDIR /app
COPY web/package.json web/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

# --- Build ---
FROM deps AS build

# NEXT_PUBLIC_* is inlined into the browser bundle at build time. The browser
# calls the API directly, so this must be an address the *browser* can reach
# (default http://localhost:36300), not the compose service name.
ARG NEXT_PUBLIC_API_URL=http://localhost:36300
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL \
    NEXT_TELEMETRY_DISABLED=1

COPY web/ ./
RUN pnpm build

# --- Runtime ---
FROM node:22-alpine AS runtime

RUN corepack enable pnpm

WORKDIR /app

COPY --from=build /app/package.json ./
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/.next ./.next

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD wget -q --spider http://localhost:36301/ || exit 1

EXPOSE 36301

CMD ["pnpm", "start"]
