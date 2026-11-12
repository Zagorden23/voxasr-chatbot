# VoxASR — Automatic Speech Recognition Chatbot

## Project Overview
- **Name**: VoxASR (`webapp`)
- **Goal**: Interactive Automatic Speech Recognition (ASR) Voice & Text Chatbot where **Python** trains the NLP intent model offline and **pure JavaScript** (zero TypeScript) powers the backend API, edge Softmax inference, React 18 SPA, live microphone Speech-to-Text, and Word Error Rate (WER) evaluator.
- **Tech Stack**:
  - **Training Pipeline (Python 3)**: `scikit-learn` + `NumPy` (`training/train_chatbot.py`, `training/intents.json`)
  - **Backend & Inference (Pure JavaScript)**: Hono (`src/index.js`, `src/inference.js`) + exported Softmax weights (`src/model/chatbot_model.json`)
  - **Frontend (Pure JavaScript + React 18)**: `public/static/app.js` + TailwindCSS + Web Speech API (`SpeechRecognition` & `speechSynthesis`) + Web Audio API (`AnalyserNode` live spectrum)
  - **Database**: Cloudflare D1 (SQLite) for chat session history, ASR confidence logs, and WER evaluations

## Live Preview URL
- **Sandbox Preview**: https://3000-i9g6zueb7rbfsm3ns19zz-5185f4aa.sandbox.novita.ai

## Currently Completed Features
1. **Python NLP Training Pipeline (`npm run train`)**:
   - Trains a 12-class ASR domain Softmax Logistic Classifier over Sublinear TF-IDF Word (1–2) n-grams + per-word Character (3–4) n-grams (robust to ASR phonetic transcription variations).
   - Achieves **99.83% training accuracy** across 589 augmented samples (1,600 n-gram features) and exports portable JSON weights (`src/model/chatbot_model.json`).
2. **Pure JavaScript Edge Inference (`src/inference.js`)**:
   - Exact replication of Python tokenization, Sublinear TF-IDF, L2 normalization, and Softmax probabilities in `<1 ms` without TypeScript.
3. **Live Browser Speech-to-Text & Voice Output**:
   - Real-time microphone dictation (`Web Speech API`) with interim live transcripts, Web Audio frequency visualizer bars, locale selector (`en-US`, `en-GB`, `en-IN`), and toggleable Text-to-Speech (`speechSynthesis`) replies.
4. **Word Error Rate (WER) & Character Error Rate (CER) Lab**:
   - Levenshtein dynamic programming word-level alignment showing color-coded Matches, Substitutions, Deletions, and Insertions—plus live microphone dictation to test your own spoken hypothesis against ground-truth text.
5. **Persistent Session History (Cloudflare D1)**:
   - Stores sessions, voice/text messages, ASR confidence scores, and Softmax intent tags in D1 SQLite.

## Functional Entry URIs
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | React 18 Single-Page Application (Voice Chat, WER Lab, Python Model Inspector) |
| `GET` | `/api/model` | Returns Python training metrics, architecture details, and intent summaries |
| `POST` | `/api/chat` | Classifies message (`{ sessionId, message, inputMode, asrConfidence }`) & saves to D1 |
| `GET` | `/api/sessions` | Lists recent chat sessions stored in D1 |
| `GET` | `/api/sessions/:id` | Retrieves full message & ASR transcript history for a session |
| `DELETE` | `/api/sessions/:id` | Deletes a saved chat session |
| `POST` | `/api/wer` | Computes WER, CER, and word-level Levenshtein alignment (`{ reference, hypothesis }`) |
| `GET` | `/api/wer/history` | Lists recent WER benchmark runs |

## Data Architecture (Cloudflare D1)
- `chat_sessions`: `id`, `title`, `created_at`, `updated_at`
- `chat_messages`: `id`, `session_id`, `role`, `content`, `input_mode`, `asr_confidence`, `intent_tag`, `intent_confidence`, `created_at`
- `wer_evaluations`: `id`, `reference_text`, `hypothesis_text`, `wer`, `cer`, `substitutions`, `deletions`, `insertions`, `created_at`

## User Guide
```bash
# 1. Retrain the chatbot NLP model in Python (optional after editing training/intents.json)
npm run train

# 2. Apply local D1 database migrations
npm run db:migrate:local

# 3. Build pure JavaScript bundle
npm run build

# 4. Run locally with PM2
pm2 start ecosystem.config.cjs
```

## Features Not Yet Implemented & Next Steps
- Server-side audio file upload transcription via external Whisper REST API (can be added with an OpenAI/Groq API secret).
- Multilingual intent training corpus (Spanish, Mandarin, Hindi).

## Deployment Status
- **Environment**: Sandbox Live & Verified (`port 3000`)
- **Last Updated**: 2026-11-12
