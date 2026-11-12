-- VoxASR Database Schema (D1 / SQLite)

CREATE TABLE IF NOT EXISTS chat_sessions (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL DEFAULT 'Voice ASR Session',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL, -- 'user' or 'bot'
  content TEXT NOT NULL,
  input_mode TEXT DEFAULT 'text', -- 'voice' or 'text'
  asr_confidence REAL,
  intent_tag TEXT,
  intent_confidence REAL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS wer_evaluations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  reference_text TEXT NOT NULL,
  hypothesis_text TEXT NOT NULL,
  wer REAL NOT NULL,
  cer REAL NOT NULL,
  substitutions INTEGER NOT NULL DEFAULT 0,
  deletions INTEGER NOT NULL DEFAULT 0,
  insertions INTEGER NOT NULL DEFAULT 0,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id);
