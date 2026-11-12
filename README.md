# VoxASR — Commercial Multilingual Voice ASR & CS/Math AI Assistant

## Project Overview
**VoxASR** (`webapp`) is a commercial-grade Voice & Text AI Chatbot built for **Computer Science**, **Mathematical Computing**, and **Automatic Speech Recognition (ASR)** across **35 certified international languages**, engineered for **Render** deployment (`Node.js` + `Hono` + `PostgreSQL` / `SQLite`).

### Core Architecture & Features
- **Hybrid AI Inference (`src/inference.js` + `src/knowledge.js`)**:
  - **Sub-millisecond Softmax Neural Classifier**: Sublinear TF-IDF (Word 1-2g + Char 3-4g + CJK) weights trained in Python (`training/train_chatbot.py`) and exported as portable JSON (`src/model/chatbot_model.json`).
  - **Okapi BM25 Extractive Knowledge Pool**: Indexes sharded Wikipedia & ArXiv articles (`public/static/kb/`) with question analysis, WH-type hints, synonym matching, and side-by-side comparisons (`Compare A and B`).
  - **Self-Learning Live Web Scraper (`src/web_learner.js`)**: Automatically fetches unknown topics from Wikipedia MediaWiki API & ArXiv Atom API on demand, indexes them live into memory, and persists them to PostgreSQL / SQLite across restarts.
  - **Injection-Safe Mathematical Evaluator**: Evaluates arithmetic, powers, trig, and logarithmic expressions directly without `eval()`.
  - **Multi-Turn Context Resolution**: Resolves follow-up pronoun queries using the session's active topic.
- **35 Certified World Languages (`src/languages.json`)**:
  - Full Web Speech API (STT) & SpeechSynthesis (TTS) support across European, Asian, Indic, Middle Eastern, and African locales, with automatic RTL layout for Arabic, Persian, Urdu, and Hebrew.
- **Commercial Persistence & Security (`src/db.js`, `src/index.js`)**:
  - **Render Managed PostgreSQL (`DATABASE_URL`)** in production with automatic **SQLite (`node:sqlite`)** fallback for zero-config local development.
  - Full session history (create, search, rename, delete, Markdown/JSON export), per-message thumbs-up/down feedback, WER/CER benchmarking history, live usage analytics (`/api/analytics`), security headers, sliding-window rate limiting, and `/health` readiness checks.

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | React 18 SPA (Voice Chat, Knowledge Pool, WER Lab, Analytics & Model) |
| `GET` | `/health` | Render health & database/KB readiness check |
| `GET` | `/api/languages` | Lists all 35 certified world languages (with RTL metadata) |
| `GET` | `/api/model` | Model architecture, CV metrics, and KB stats |
| `GET` | `/api/analytics` | Usage telemetry (sessions, voice vs text, feedback, latency) |
| `GET` | `/api/knowledge/status` | Live BM25 knowledge pool & upgrade status |
| `GET` | `/api/knowledge/search` | Instant search over indexed CS/Math articles |
| `POST` | `/api/knowledge/upgrade` | Triggers live ArXiv & Wikipedia knowledge harvest |
| `POST` | `/api/knowledge/scrape` | On-demand scraping & persistence for any topic |
| `POST` | `/api/chat` | Hybrid multilingual chat inference & DB persistence |
| `POST` | `/api/chat/feedback` | Records thumbs-up/down message rating |
| `GET` | `/api/sessions` | Lists recent chat sessions |
| `GET` | `/api/sessions/:id` | Retrieves session messages |
| `PATCH` | `/api/sessions/:id` | Renames a session |
| `DELETE` | `/api/sessions/:id` | Deletes a session |
| `GET` | `/api/sessions/:id/export` | Exports conversation as Markdown or JSON |
| `POST` | `/api/wer` | Computes WER, CER, and word-level Levenshtein alignment |
| `GET` / `DELETE` | `/api/wer/history` | Lists or clears WER benchmark history |

## Local Development
```bash
# Install Node dependencies
npm install

# Run automated tests
npm test

# Start server (binds 0.0.0.0:10000 by default)
npm start

# Optional: Refresh Wikipedia KB shards & retrain classifier (requires Python 3.12+)
npm run harvest -- --max 600 --per-domain 45 --no-vital
npm run train
```

## Deploying to Render

### Method 1: Render Blueprint (`render.yaml`) — Recommended
1. Push the repository (`webapp` directory as repo root, or configure `rootDir` if using a monorepo) to GitHub, GitLab, or Bitbucket.
2. Validate the Blueprint locally:
   ```bash
   render blueprints validate render.yaml
   ```
3. In the Render Dashboard, click **New + -> Blueprint** and select your repository (or open `https://dashboard.render.com/blueprint/new?repo=<YOUR_HTTPS_REPO_URL>`).
4. Review the `voxasr-chatbot` web service and `voxasr-db` PostgreSQL database, then click **Apply**.
   - Note: `render.yaml` uses `plan: free` for quick evaluation (Render free Postgres expires after 30 days; switch `plan` to `0.1c-256mb` for persistent production storage and `0.5c-512mb` for always-on web compute).

### Method 2: Docker on Render (Node.js + Python ML)
If you want to run Python `scikit-learn` retraining (`training/train_chatbot.py`) inside the container as well, switch `runtime: docker` in `render.yaml` using the included `Dockerfile` and `.dockerignore`.
