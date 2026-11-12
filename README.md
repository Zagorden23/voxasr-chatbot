# VoxASR — Multilingual Computer Science & Mathematical AI Assistant

## Project Overview
- **Name**: VoxASR (`webapp`)
- **Mission**: An intelligent Voice & Text AI Chatbot trained across **35 certified international languages** and all core pillars of **Computer Science** and **Mathematical Computing**, featuring a **continuous automated web-scraping harvester** (Wikipedia & ArXiv) and **ML-DL retraining pipeline** with sub-millisecond pure JavaScript edge inference.
- **Tech Stack**:
  - **Machine Learning & Deep Learning (Python 3.12)**: `scikit-learn` + `NumPy` + `SciPy` (`training/train_chatbot.py`, `training/auto_train_pipeline.py`)
  - **Automated Knowledge Harvester**: `requests` + `BeautifulSoup` + `xml.etree` (`training/knowledge_harvester.py`, Wikipedia REST API, ArXiv Atom API)
  - **Continuous Upgrade Daemon (Port 3001)**: `training/upgrade_server.py`
  - **Backend & Pure JS Edge Inference (Zero TypeScript)**: Hono (`src/index.js`, `src/inference.js`) + exported Softmax weights (`src/model/chatbot_model.json`)
  - **Frontend (Pure JavaScript + React 18)**: `public/static/app.js` + TailwindCSS + Web Speech API (35 SpeechRecognition & SpeechSynthesis locales) + Web Audio API (`AnalyserNode` live spectrum)
  - **Database**: Cloudflare D1 (SQLite) for multilingual chat sessions, ASR confidence logs, and WER evaluations

## Key Upgrades & Features

### 1. 35 Certified World Languages Support
- Comprehensive support for UN official and ISO-639 / BCP-47 certified world languages:
  - **European**: English (`en-US`, `en-GB`, `en-IN`), Spanish (`es-ES`, `es-MX`), French (`fr-FR`), German (`de-DE`), Italian (`it-IT`), Portuguese (`pt-BR`), Dutch (`nl-NL`), Polish (`pl-PL`), Czech (`cs-CZ`), Swedish (`sv-SE`), Ukrainian (`uk-UA`), Greek (`el-GR`).
  - **Asian & Indic**: Mandarin Chinese (`zh-CN`, `zh-TW`), Hindi (`hi-IN`), Japanese (`ja-JP`), Korean (`ko-KR`), Bengali (`bn-IN`), Vietnamese (`vi-VN`), Indonesian (`id-ID`), Tamil (`ta-IN`), Telugu (`te-IN`), Marathi (`mr-IN`), Thai (`th-TH`).
  - **Middle Eastern & African**: Arabic (`ar-SA`), Russian (`ru-RU`), Turkish (`tr-TR`), Persian/Farsi (`fa-IR`), Urdu (`ur-PK`), Hebrew (`he-IL`), Swahili (`sw-KE`).
- **Unicode-Preserving Multi-Script Tokenization**: Unified NFKC normalization in Python and JavaScript retaining Latin, Devanagari, Hanzi, Cyrillic, Arabic, Greek, Kana, and Hangul scripts.
- **Dynamic SpeechRecognition Locale**: Switch speech recognition and Text-to-Speech voices instantly in the UI.

### 2. Vast Knowledge Pool in Computer Science & Mathematical Computing
Expert knowledge coverage across 12 pillars:
1. **Algorithms & Complexity**: Dijkstra, A*, Dynamic Programming, NP-completeness, Cook-Levin, Big-O asymptotic analysis, Master Theorem.
2. **Advanced Data Structures**: B-Trees, B+ Trees, Red-Black Trees, Bloom Filters, Segment Trees, Disjoint-Set Union (Union-Find), HNSW Vector Indexing.
3. **Numerical Analysis & Mathematical Computing**: Singular Value Decomposition (SVD), QR/LU decompositions, Runge-Kutta RK4, Cooley-Tukey FFT, Newton-Raphson, IEEE 754 precision, Monte Carlo.
4. **Machine Learning & Deep Learning (ML/DL)**: Backpropagation, Scaled Dot-Product Attention, Transformers, RoPE, AdamW, Diffusion Models, Loss Functions, Regularization, RLHF.
5. **Speech AI & Automatic Speech Recognition (ASR)**: OpenAI Whisper, Conformer, CTC Loss, RNN-T, Wav2Vec 2.0, Mel-scale filterbanks, Log-Mel Spectrograms, VAD.
6. **Theoretical CS, Automata & Discrete Math**: Turing Machines, Chomsky Hierarchy, Halting Problem, Graph Theory, Modular Arithmetic, Fermat's Little Theorem.
7. **Operating Systems & Distributed Systems**: CAP Theorem, PACELC, Raft Consensus, Virtual Memory & Paging, TLB, Mutex vs Semaphore, Consistent Hashing.
8. **Compilers & Programming Language Theory**: Lexing, Parsing (LL/LR/Pratt), Abstract Syntax Tree (AST), LLVM IR, SSA Form, Type Inference, Generational GC.
9. **Modern Cryptography & Cyber Security**: RSA, Elliptic Curve Cryptography (ECC), AES-GCM, Zero-Knowledge Proofs (ZK-SNARKs), Post-Quantum Cryptography (ML-KEM / Kyber).
10. **Linear Algebra, Calculus & Probability for Computing**: Jacobian & Hessian matrices, Eigenvalues, Bayes' Theorem, Markov Chains, Central Limit Theorem.
11. **GPU Computing & High-Performance Computing (HPC)**: CUDA architecture, Warps, Thread Blocks, OpenMP, MPI, BLAS GEMM, Tensor Cores.
12. **Database Internals & Vector Search**: B+ Trees vs LSM Trees, ACID, MVCC, In-memory MemTable, SSTables, Vector Cosine/L2 similarity search.

### 3. Continuous Automated Web Scraping & ML-DL Retraining
- **Automated Harvester (`training/knowledge_harvester.py`)**: Crawls Wikipedia REST API and ArXiv Atom API across computer science categories (`cs.AI`, `cs.LG`, `cs.DS`, `math.NA`, `math.OC`), extracts summaries, mathematical formulas, and auto-generates multilingual query patterns.
- **Dynamic Knowledge Pool (`training/dynamic_knowledge_pool.json`)**: Persists ingested articles and expanding training patterns.
- **ML-DL Auto-Train Pipeline (`training/auto_train_pipeline.py`)**: Ingests new articles, extracts 5,000+ Sublinear TF-IDF features, retrains the Softmax classifier with 5-fold cross validation, and exports portable JSON weights (`src/model/chatbot_model.json`).
- **Continuous Upgrade Daemon (`training/upgrade_server.py`)**: Runs on `127.0.0.1:3001` to accept one-click upgrade triggers and on-demand topic queries.
- **Interactive UI Dashboard**: Dedicated "Knowledge Base & Auto-Upgrade" tab with live counters, one-click upgrade button, and on-demand topic input.

## Functional Entry URIs
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | React 18 SPA (Voice Chat, Knowledge Base, WER Lab, Model Inspector) |
| `GET` | `/api/model` | Model training metrics, 5-fold CV score, and architecture details |
| `GET` | `/api/languages` | Lists all 35 certified world languages with BCP-47 codes & flags |
| `GET` | `/api/knowledge/status` | Real-time dynamic knowledge pool statistics and upgrade history |
| `POST` | `/api/knowledge/upgrade` | Triggers background web scraping & ML-DL model retraining |
| `POST` | `/api/knowledge/scrape` | On-demand web scraping for any specific CS or mathematical topic |
| `POST` | `/api/chat` | Multilingual intent classification & technical explanations |
| `GET` | `/api/sessions` | Lists recent chat sessions stored in Cloudflare D1 |
| `GET` | `/api/sessions/:id` | Retrieves full session history |
| `DELETE` | `/api/sessions/:id` | Deletes a chat session |
| `POST` | `/api/wer` | Computes WER, CER, and word-level Levenshtein alignment |
| `GET` | `/api/wer/history` | Lists recent WER benchmark runs |

## User Guide
```bash
# 1. Harvest latest CS/Math knowledge and retrain the model
npm run upgrade

# 2. Or retrain directly from current knowledge pool
npm run train

# 3. Start the continuous upgrade background daemon
npm run daemon

# 4. Build pure JavaScript distribution
npm run build

# 5. Start web server locally with PM2 (or wrangler)
pm2 start ecosystem.config.cjs
```
