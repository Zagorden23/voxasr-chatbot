-- VoxASR Database Schema (PostgreSQL / SQLite compatible)

CREATE TABLE IF NOT EXISTS chat_sessions (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL DEFAULT 'Voice ASR Session',
  language TEXT DEFAULT 'en-US',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL, -- 'user' or 'bot'
  content TEXT NOT NULL,
  input_mode TEXT DEFAULT 'text', -- 'voice', 'text', or 'system'
  asr_confidence REAL,
  intent_tag TEXT,
  intent_confidence REAL,
  detected_language TEXT,
  source_title TEXT,
  source_url TEXT,
  latency_ms REAL,
  feedback TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS wer_evaluations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  reference_text TEXT NOT NULL,
  hypothesis_text TEXT NOT NULL,
  wer REAL NOT NULL,
  cer REAL NOT NULL,
  word_accuracy REAL NOT NULL DEFAULT 0,
  substitutions INTEGER NOT NULL DEFAULT 0,
  deletions INTEGER NOT NULL DEFAULT 0,
  insertions INTEGER NOT NULL DEFAULT 0,
  language TEXT DEFAULT 'en-US',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS learned_articles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  lang TEXT NOT NULL DEFAULT 'en',
  domain TEXT,
  text TEXT NOT NULL,
  aliases_json TEXT DEFAULT '[]',
  source TEXT DEFAULT 'Wikipedia',
  url TEXT DEFAULT '',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (title, lang)
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id);
