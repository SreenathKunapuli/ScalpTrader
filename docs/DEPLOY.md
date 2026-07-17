# Production Deploy Guide

This document covers the exact steps to deploy the ScalpTrader stack:

- **Engine** — runs on the owner Mac, writes to hosted Postgres via `DATABASE_URL`
- **API** (`api/`) — deploys to Fly.io, reads the same `DATABASE_URL`
- **Dashboard** (`web/`) — deploys to Vercel, talks to the Fly API

---

## Architecture

```
Owner Mac
  └─ scalpctl run --tier medium   ──writes──► Hosted Postgres (Neon / Supabase)
                                                       │
                                               Fly.io (scalptrader-api)
                                                  FastAPI  :8080
                                                       │
                                               Vercel (web/)
                                                  Next.js dashboard
                                                       │
                                               Browser (owner + guests)
```

---

## 1. Provision Postgres

### Option A — Neon (recommended, generous free tier)

```bash
# 1. Create a project at https://neon.tech
# 2. Copy the connection string from the dashboard:
#    postgresql://user:password@ep-xxx.us-east-2.aws.neon.tech/neondb?sslmode=require
```

### Option B — Supabase

```bash
# 1. Create a project at https://supabase.com
# 2. Go to Project Settings → Database → Connection string (URI mode)
#    postgresql://postgres:password@db.xxxx.supabase.co:5432/postgres
```

Store the connection string — you'll use it in every section below.

---

## 2. Configure the Engine (Owner Mac)

```bash
# In the repo root:
cp .env.example .env
# Edit .env and set at minimum:
#   DATABASE_URL=postgresql://...
#   ALPACA_API_KEY=...
#   ALPACA_SECRET_KEY=...
#   APP_PASSWORD=<strong password>
#   JWT_SECRET=$(openssl rand -hex 32)
#   CORS_ORIGINS=https://<your-app>.vercel.app
```

The engine reads `.env` via pydantic-settings on startup. The API (locally or on Fly) also
reads from environment variables with the same names.

---

## 3. Deploy API to Fly.io

### 3a. Prerequisites

```bash
# Install Fly CLI if needed:
curl -L https://fly.io/install.sh | sh
fly auth login
```

### 3b. Launch (first time only)

Run from the **repo root** (build context must be root so `pyproject.toml` and `engine/`
source are present — the Dockerfile at `api/Dockerfile` relies on this):

```bash
# This creates the app on Fly but does NOT deploy yet.
fly launch --no-deploy --config fly.toml
# When prompted: use the existing fly.toml (do not overwrite).
# Choose a region near your Postgres host (e.g. iad for US-East).
```

### 3c. Set secrets

Secrets are injected as environment variables at runtime; they are never stored in fly.toml:

```bash
fly secrets set \
  DATABASE_URL="postgresql://user:pass@host/db?sslmode=require" \
  APP_PASSWORD="<strong password matching .env>" \
  JWT_SECRET="$(openssl rand -hex 32)" \
  CORS_ORIGINS="https://<your-app>.vercel.app"
```

Optional (only if you add Redis pub-sub):

```bash
fly secrets set REDIS_URL="redis://..."
```

### 3d. Deploy

```bash
# From repo root — Fly uses fly.toml which points to api/Dockerfile
fly deploy
```

Fly will build the image, push it, and start the VM. Watch the logs:

```bash
fly logs
```

The app URL will be `https://scalptrader-api.fly.dev` (or your custom name).

---

## 4. Deploy Dashboard to Vercel

### 4a. Prerequisites

```bash
npm i -g vercel   # if not already installed
vercel login
```

### 4b. Deploy

```bash
cd web/
vercel deploy --prod
```

When Vercel asks for environment variables (or set them in the Vercel dashboard under
Project → Settings → Environment Variables):

| Variable | Value |
|---|---|
| `API_INTERNAL_BASE` | `https://scalptrader-api.fly.dev` |
| `NEXT_PUBLIC_API_BASE` | `https://scalptrader-api.fly.dev` |
| `NEXT_PUBLIC_WS_BASE` | `wss://scalptrader-api.fly.dev` |

After the first deploy you get a `.vercel.app` URL. Update `CORS_ORIGINS` on Fly:

```bash
fly secrets set CORS_ORIGINS="https://<your-project>.vercel.app"
```

---

## 5. Post-Deploy Smoke Checklist

Run these checks after every deploy:

- [ ] **API health** — `curl https://scalptrader-api.fly.dev/healthz` should return
  `{"db":"ok","engine_heartbeat_fresh":false}` (heartbeat only goes true once the engine runs).

- [ ] **Owner login** — open `https://<your-project>.vercel.app`, log in with `APP_PASSWORD`,
  verify the dashboard loads and shows "engine status: IDLE" or similar.

- [ ] **Guest spectate** — open the URL in an incognito window, click "Guest", confirm you can
  see the equity curve (read-only) without a password.

- [ ] **WebSocket** — on the dashboard, confirm the live equity ticker updates within ~2s once
  the engine is running on the Mac.

- [ ] **Engine write** — on the Mac run `scalpctl run --tier low` for 30s, then check that the
  `/healthz` endpoint returns `"engine_heartbeat_fresh": true`.

---

## 6. Local Development

The engine and API both fall back to an **SQLite** file (`scalpengine.db` in the repo root)
when `DATABASE_URL` is blank. This means you can develop and run tests without a Postgres
instance.

### 6a. Minimal local .env

```bash
cp .env.example .env
# Only these lines are required for local dev:
#   APP_PASSWORD=dev
#   JWT_SECRET=dev-secret-do-not-use-in-prod
#   CORS_ORIGINS=http://localhost:3000
# Leave DATABASE_URL blank (SQLite fallback kicks in automatically).
```

### 6b. Run two dev servers

Terminal 1 — FastAPI (port 8000):

```bash
uvicorn api.app.main:app --reload --port 8000
```

Terminal 2 — Next.js dashboard (port 3000):

```bash
cd web/
cp .env.example .env.local
# .env.local already has http://localhost:8000 URLs — no changes needed.
npm run dev
```

### 6c. Run tests

```bash
# Full test suite (engine + api):
.venv/bin/pytest -q

# API tests only:
.venv/bin/pytest -q api/tests/

# Engine tests only:
.venv/bin/pytest -q engine/tests/
```

### 6d. Engine CLI

```bash
# Paper trading (only mode available):
scalpctl run --tier medium

# Inspect DB state:
scalpctl status
```

---

## Env-var to Settings field mapping

The table below confirms that every key in `.env.example` matches the exact field name in
`engine/scalpengine/config/settings.py` (pydantic-settings lower-cases env vars):

| `.env.example` key | `Settings` field | Notes |
|---|---|---|
| `TRADING_MODE` | `trading_mode` | |
| `ALPACA_API_KEY` | `alpaca_api_key` | |
| `ALPACA_SECRET_KEY` | `alpaca_secret_key` | |
| `ALPACA_PAPER_BASE_URL` | `alpaca_paper_base_url` | |
| `ALPACA_LIVE_API_KEY` | `alpaca_live_api_key` | blank → live impossible |
| `ALPACA_LIVE_SECRET_KEY` | `alpaca_live_secret_key` | |
| `MAX_LIVE_EQUITY_USD` | `max_live_equity_usd` | float or blank |
| `DATABASE_URL` | `database_url` | blank → SQLite fallback |
| `DB_BACKEND` | `db_backend` | |
| `SQLITE_PATH` | `sqlite_path` | |
| `REDIS_URL` | `redis_url` | optional |
| `APP_PASSWORD` | `app_password` | |
| `JWT_SECRET` | `jwt_secret` | |
| `JWT_EXPIRY_HOURS` | `jwt_expiry_hours` | |
| `CORS_ORIGINS` | `cors_origins` | comma-separated |
| `SCALP_PROFILE` | `scalp_profile` | off \| small \| large |
| `RISK_TIER` | `risk_tier` | |
| `BAR_INTERVAL_S` | `bar_interval_s` | |
| `SIGNAL_BAR_INTERVAL_S` | `signal_bar_interval_s` | |
| `HISTORY_WARMUP_DAYS` | `history_warmup_days` | |
| `STALENESS_PAUSE_S` | `staleness_pause_s` | |
| `STALENESS_KILL_S` | `staleness_kill_s` | |
| `BROKER_ERROR_KILL_COUNT` | `broker_error_kill_count` | |
| `BROKER_ERROR_KILL_WINDOW_S` | `broker_error_kill_window_s` | |
| `HEARTBEAT_INTERVAL_S` | `heartbeat_interval_s` | |
| `MODELS_DIR` | `models_dir` | |
