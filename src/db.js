/**
 * Commercial Database Layer for VoxASR
 * ------------------------------------
 * Automatically selects:
 *   1. PostgreSQL (`pg`) when `DATABASE_URL` is configured (Render Managed PostgreSQL)
 *   2. Node built-in SQLite (`node:sqlite`) for zero-config local/single-instance persistence
 *   3. Atomic JSON file store fallback if neither is available
 */

import fs from 'node:fs'
import path from 'node:path'

let pgPool = null
let sqliteDb = null
let backendType = 'memory'
let jsonStorePath = null
let jsonStore = { sessions: [], messages: [], wer: [], learned: [] }

function isRenderInternalPostgres(url) {
  return /^postgres(?:ql)?:\/\/[^/]*@dpg-[^./]+(?::\d+)?\//i.test(url)
}

function resolveDataDir() {
  const configured = process.env.DATA_DIR
  if (configured) return path.resolve(configured)
  return path.resolve(process.cwd(), 'data')
}

function flushJsonStore() {
  if (!jsonStorePath) return
  try {
    fs.mkdirSync(path.dirname(jsonStorePath), { recursive: true })
    const tmp = `${jsonStorePath}.tmp`
    fs.writeFileSync(tmp, JSON.stringify(jsonStore), 'utf8')
    fs.renameSync(tmp, jsonStorePath)
  } catch (err) {
    console.warn('[DB] JSON store flush failed:', err.message)
  }
}

export async function initDatabase() {
  const dbUrl = (process.env.DATABASE_URL || '').trim()

  if (dbUrl) {
    try {
      const { Pool } = await import('pg')
      const useSsl =
        process.env.PGSSLMODE === 'require' ||
        (!isRenderInternalPostgres(dbUrl) && !/localhost|127\.0\.0\.1/i.test(dbUrl))

      pgPool = new Pool({
        connectionString: dbUrl,
        max: Number(process.env.PG_POOL_MAX || 10),
        idleTimeoutMillis: 30000,
        connectionTimeoutMillis: 8000,
        ssl: useSsl ? { rejectUnauthorized: false } : false
      })

      await pgPool.query(`
        CREATE TABLE IF NOT EXISTS chat_sessions (
          id TEXT PRIMARY KEY,
          title TEXT NOT NULL DEFAULT 'Voice ASR Session',
          language TEXT DEFAULT 'en-US',
          created_at TIMESTAMPTZ DEFAULT NOW(),
          updated_at TIMESTAMPTZ DEFAULT NOW()
        );
      `)
      await pgPool.query(`
        CREATE TABLE IF NOT EXISTS chat_messages (
          id SERIAL PRIMARY KEY,
          session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          input_mode TEXT DEFAULT 'text',
          asr_confidence DOUBLE PRECISION,
          intent_tag TEXT,
          intent_confidence DOUBLE PRECISION,
          detected_language TEXT,
          source_title TEXT,
          source_url TEXT,
          latency_ms DOUBLE PRECISION,
          feedback TEXT,
          created_at TIMESTAMPTZ DEFAULT NOW()
        );
      `)
      await pgPool.query(`
        CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id);
      `)
      await pgPool.query(`
        CREATE TABLE IF NOT EXISTS wer_evaluations (
          id SERIAL PRIMARY KEY,
          reference_text TEXT NOT NULL,
          hypothesis_text TEXT NOT NULL,
          wer DOUBLE PRECISION NOT NULL,
          cer DOUBLE PRECISION NOT NULL,
          word_accuracy DOUBLE PRECISION NOT NULL DEFAULT 0,
          substitutions INTEGER NOT NULL DEFAULT 0,
          deletions INTEGER NOT NULL DEFAULT 0,
          insertions INTEGER NOT NULL DEFAULT 0,
          language TEXT DEFAULT 'en-US',
          created_at TIMESTAMPTZ DEFAULT NOW()
        );
      `)
      await pgPool.query(`
        CREATE TABLE IF NOT EXISTS learned_articles (
          id SERIAL PRIMARY KEY,
          title TEXT NOT NULL,
          lang TEXT NOT NULL DEFAULT 'en',
          domain TEXT,
          text TEXT NOT NULL,
          aliases_json TEXT DEFAULT '[]',
          source TEXT DEFAULT 'Wikipedia',
          url TEXT DEFAULT '',
          created_at TIMESTAMPTZ DEFAULT NOW(),
          updated_at TIMESTAMPTZ DEFAULT NOW(),
          UNIQUE (title, lang)
        );
      `)

      backendType = 'postgres'
      console.log('[DB] Connected to PostgreSQL')
      return backendType
    } catch (err) {
      console.error('[DB] PostgreSQL initialization failed, falling back to local store:', err.message)
      if (pgPool) {
        await pgPool.end().catch(() => {})
        pgPool = null
      }
    }
  }

  // Try built-in node:sqlite (Node >= 22.5)
  try {
    const { DatabaseSync } = await import('node:sqlite')
    const dataDir = resolveDataDir()
    fs.mkdirSync(dataDir, { recursive: true })
    const dbPath = path.join(dataDir, 'voxasr.sqlite')
    sqliteDb = new DatabaseSync(dbPath)
    sqliteDb.exec('PRAGMA journal_mode = WAL;')
    sqliteDb.exec('PRAGMA foreign_keys = ON;')
    sqliteDb.exec(`
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
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        input_mode TEXT DEFAULT 'text',
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
      CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id);
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
    `)
    backendType = 'sqlite'
    console.log(`[DB] Connected to local SQLite (${dbPath})`)
    return backendType
  } catch (err) {
    console.warn('[DB] node:sqlite unavailable, using JSON file store:', err.message)
  }

  // Fallback atomic JSON store
  const dataDir = resolveDataDir()
  jsonStorePath = path.join(dataDir, 'voxasr_db.json')
  if (fs.existsSync(jsonStorePath)) {
    try {
      const parsed = JSON.parse(fs.readFileSync(jsonStorePath, 'utf8'))
      jsonStore = {
        sessions: Array.isArray(parsed.sessions) ? parsed.sessions : [],
        messages: Array.isArray(parsed.messages) ? parsed.messages : [],
        wer: Array.isArray(parsed.wer) ? parsed.wer : [],
        learned: Array.isArray(parsed.learned) ? parsed.learned : []
      }
    } catch (_) {}
  }
  backendType = 'json'
  console.log(`[DB] Using JSON file store (${jsonStorePath})`)
  return backendType
}

export function getDbBackend() {
  return backendType
}

export async function closeDatabase() {
  if (pgPool) {
    await pgPool.end().catch(() => {})
    pgPool = null
  }
  if (sqliteDb && typeof sqliteDb.close === 'function') {
    try {
      sqliteDb.close()
    } catch (_) {}
    sqliteDb = null
  }
}

export async function listSessions(limit = 30) {
  if (backendType === 'postgres') {
    const { rows } = await pgPool.query(
      `SELECT s.id, s.title, s.language, s.created_at, s.updated_at,
              COUNT(m.id)::int AS message_count
       FROM chat_sessions s
       LEFT JOIN chat_messages m ON m.session_id = s.id
       GROUP BY s.id
       ORDER BY s.updated_at DESC
       LIMIT $1`,
      [limit]
    )
    return rows
  }
  if (backendType === 'sqlite') {
    return sqliteDb
      .prepare(
        `SELECT s.id, s.title, s.language, s.created_at, s.updated_at,
                COUNT(m.id) AS message_count
         FROM chat_sessions s
         LEFT JOIN chat_messages m ON m.session_id = s.id
         GROUP BY s.id
         ORDER BY s.updated_at DESC
         LIMIT ?`
      )
      .all(limit)
  }
  return jsonStore.sessions
    .map((s) => ({
      ...s,
      message_count: jsonStore.messages.filter((m) => m.session_id === s.id).length
    }))
    .sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at))
    .slice(0, limit)
}

export async function getSessionWithMessages(sessionId) {
  if (backendType === 'postgres') {
    const { rows: sRows } = await pgPool.query(`SELECT * FROM chat_sessions WHERE id = $1`, [sessionId])
    const { rows: mRows } = await pgPool.query(
      `SELECT * FROM chat_messages WHERE session_id = $1 ORDER BY id ASC`,
      [sessionId]
    )
    return {
      session: sRows[0] || { id: sessionId, title: 'Voice ASR Session', language: 'en-US' },
      messages: mRows
    }
  }
  if (backendType === 'sqlite') {
    const session = sqliteDb.prepare(`SELECT * FROM chat_sessions WHERE id = ?`).get(sessionId)
    const messages = sqliteDb
      .prepare(`SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id ASC`)
      .all(sessionId)
    return {
      session: session || { id: sessionId, title: 'Voice ASR Session', language: 'en-US' },
      messages
    }
  }
  const session = jsonStore.sessions.find((s) => s.id === sessionId)
  const messages = jsonStore.messages.filter((m) => m.session_id === sessionId)
  return {
    session: session || { id: sessionId, title: 'Voice ASR Session', language: 'en-US' },
    messages
  }
}

export async function renameSession(sessionId, title) {
  const cleanTitle = String(title || '').trim().slice(0, 120)
  if (!cleanTitle) return false
  if (backendType === 'postgres') {
    const res = await pgPool.query(
      `UPDATE chat_sessions SET title = $1, updated_at = NOW() WHERE id = $2`,
      [cleanTitle, sessionId]
    )
    return res.rowCount > 0
  }
  if (backendType === 'sqlite') {
    const res = sqliteDb
      .prepare(`UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?`)
      .run(cleanTitle, sessionId)
    return res.changes > 0
  }
  const s = jsonStore.sessions.find((x) => x.id === sessionId)
  if (!s) return false
  s.title = cleanTitle
  s.updated_at = new Date().toISOString()
  flushJsonStore()
  return true
}

export async function deleteSession(sessionId) {
  if (backendType === 'postgres') {
    await pgPool.query(`DELETE FROM chat_sessions WHERE id = $1`, [sessionId])
    return true
  }
  if (backendType === 'sqlite') {
    sqliteDb.prepare(`DELETE FROM chat_messages WHERE session_id = ?`).run(sessionId)
    sqliteDb.prepare(`DELETE FROM chat_sessions WHERE id = ?`).run(sessionId)
    return true
  }
  jsonStore.messages = jsonStore.messages.filter((m) => m.session_id !== sessionId)
  jsonStore.sessions = jsonStore.sessions.filter((s) => s.id !== sessionId)
  flushJsonStore()
  return true
}

export async function recordChatTurn({
  sessionId,
  sessionTitle,
  language,
  userMessage,
  botMessage
}) {
  if (backendType === 'postgres') {
    const client = await pgPool.connect()
    try {
      await client.query('BEGIN')
      await client.query(
        `INSERT INTO chat_sessions (id, title, language)
         VALUES ($1, $2, $3)
         ON CONFLICT (id) DO UPDATE SET updated_at = NOW(), language = EXCLUDED.language`,
        [sessionId, sessionTitle, language]
      )
      await client.query(
        `INSERT INTO chat_messages (session_id, role, content, input_mode, asr_confidence, intent_tag, intent_confidence, detected_language)
         VALUES ($1, 'user', $2, $3, $4, $5, $6, $7)`,
        [
          sessionId,
          userMessage.content,
          userMessage.inputMode,
          userMessage.asrConfidence,
          botMessage.intentTag,
          botMessage.confidence,
          language
        ]
      )
      const botRes = await client.query(
        `INSERT INTO chat_messages (session_id, role, content, input_mode, asr_confidence, intent_tag, intent_confidence, detected_language, source_title, source_url, latency_ms)
         VALUES ($1, 'bot', $2, 'system', NULL, $3, $4, $5, $6, $7, $8)
         RETURNING id`,
        [
          sessionId,
          botMessage.reply,
          botMessage.intentTag,
          botMessage.confidence,
          language,
          botMessage.sourceTitle || null,
          botMessage.sourceUrl || null,
          botMessage.latencyMs || null
        ]
      )
      await client.query('COMMIT')
      return botRes.rows[0]?.id || null
    } catch (err) {
      await client.query('ROLLBACK').catch(() => {})
      throw err
    } finally {
      client.release()
    }
  }

  if (backendType === 'sqlite') {
    const existing = sqliteDb.prepare(`SELECT id FROM chat_sessions WHERE id = ?`).get(sessionId)
    if (!existing) {
      sqliteDb
        .prepare(`INSERT INTO chat_sessions (id, title, language) VALUES (?, ?, ?)`)
        .run(sessionId, sessionTitle, language)
    } else {
      sqliteDb
        .prepare(`UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP, language = ? WHERE id = ?`)
        .run(language, sessionId)
    }
    sqliteDb
      .prepare(
        `INSERT INTO chat_messages (session_id, role, content, input_mode, asr_confidence, intent_tag, intent_confidence, detected_language)
         VALUES (?, 'user', ?, ?, ?, ?, ?, ?)`
      )
      .run(
        sessionId,
        userMessage.content,
        userMessage.inputMode,
        userMessage.asrConfidence,
        botMessage.intentTag,
        botMessage.confidence,
        language
      )
    const botRes = sqliteDb
      .prepare(
        `INSERT INTO chat_messages (session_id, role, content, input_mode, asr_confidence, intent_tag, intent_confidence, detected_language, source_title, source_url, latency_ms)
         VALUES (?, 'bot', ?, 'system', NULL, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        sessionId,
        botMessage.reply,
        botMessage.intentTag,
        botMessage.confidence,
        language,
        botMessage.sourceTitle || null,
        botMessage.sourceUrl || null,
        botMessage.latencyMs || null
      )
    return Number(botRes.lastInsertRowid)
  }

  const now = new Date().toISOString()
  let sess = jsonStore.sessions.find((s) => s.id === sessionId)
  if (!sess) {
    sess = { id: sessionId, title: sessionTitle, language, created_at: now, updated_at: now }
    jsonStore.sessions.push(sess)
  } else {
    sess.updated_at = now
    sess.language = language
  }
  const nextId = (jsonStore.messages[jsonStore.messages.length - 1]?.id || 0) + 1
  jsonStore.messages.push({
    id: nextId,
    session_id: sessionId,
    role: 'user',
    content: userMessage.content,
    input_mode: userMessage.inputMode,
    asr_confidence: userMessage.asrConfidence,
    intent_tag: botMessage.intentTag,
    intent_confidence: botMessage.confidence,
    detected_language: language,
    created_at: now
  })
  const botId = nextId + 1
  jsonStore.messages.push({
    id: botId,
    session_id: sessionId,
    role: 'bot',
    content: botMessage.reply,
    input_mode: 'system',
    asr_confidence: null,
    intent_tag: botMessage.intentTag,
    intent_confidence: botMessage.confidence,
    detected_language: language,
    source_title: botMessage.sourceTitle || null,
    source_url: botMessage.sourceUrl || null,
    latency_ms: botMessage.latencyMs || null,
    feedback: null,
    created_at: now
  })
  flushJsonStore()
  return botId
}

export async function getLastBotContextTopic(sessionId) {
  if (!sessionId) return null
  if (backendType === 'postgres') {
    const { rows } = await pgPool.query(
      `SELECT source_title FROM chat_messages WHERE session_id = $1 AND role = 'bot' AND source_title IS NOT NULL ORDER BY id DESC LIMIT 1`,
      [sessionId]
    )
    return rows[0]?.source_title || null
  }
  if (backendType === 'sqlite') {
    const row = sqliteDb
      .prepare(
        `SELECT source_title FROM chat_messages WHERE session_id = ? AND role = 'bot' AND source_title IS NOT NULL ORDER BY id DESC LIMIT 1`
      )
      .get(sessionId)
    return row?.source_title || null
  }
  const msgs = jsonStore.messages.filter((m) => m.session_id === sessionId && m.role === 'bot' && m.source_title)
  return msgs.length ? msgs[msgs.length - 1].source_title : null
}

export async function saveMessageFeedback(messageId, feedback) {
  const valid = feedback === 'up' || feedback === 'down' ? feedback : null
  if (!messageId || !valid) return false
  if (backendType === 'postgres') {
    const res = await pgPool.query(`UPDATE chat_messages SET feedback = $1 WHERE id = $2`, [valid, Number(messageId)])
    return res.rowCount > 0
  }
  if (backendType === 'sqlite') {
    const res = sqliteDb.prepare(`UPDATE chat_messages SET feedback = ? WHERE id = ?`).run(valid, Number(messageId))
    return res.changes > 0
  }
  const msg = jsonStore.messages.find((m) => m.id === Number(messageId))
  if (!msg) return false
  msg.feedback = valid
  flushJsonStore()
  return true
}

export async function recordWerEvaluation(referenceText, hypothesisText, metrics, language = 'en-US') {
  if (backendType === 'postgres') {
    await pgPool.query(
      `INSERT INTO wer_evaluations (reference_text, hypothesis_text, wer, cer, word_accuracy, substitutions, deletions, insertions, language)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)`,
      [
        referenceText,
        hypothesisText,
        metrics.wer,
        metrics.cer,
        metrics.wordAccuracy,
        metrics.substitutions,
        metrics.deletions,
        metrics.insertions,
        language
      ]
    )
    return
  }
  if (backendType === 'sqlite') {
    sqliteDb
      .prepare(
        `INSERT INTO wer_evaluations (reference_text, hypothesis_text, wer, cer, word_accuracy, substitutions, deletions, insertions, language)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        referenceText,
        hypothesisText,
        metrics.wer,
        metrics.cer,
        metrics.wordAccuracy,
        metrics.substitutions,
        metrics.deletions,
        metrics.insertions,
        language
      )
    return
  }
  const nextId = (jsonStore.wer[jsonStore.wer.length - 1]?.id || 0) + 1
  jsonStore.wer.push({
    id: nextId,
    reference_text: referenceText,
    hypothesis_text: hypothesisText,
    wer: metrics.wer,
    cer: metrics.cer,
    word_accuracy: metrics.wordAccuracy,
    substitutions: metrics.substitutions,
    deletions: metrics.deletions,
    insertions: metrics.insertions,
    language,
    created_at: new Date().toISOString()
  })
  jsonStore.wer = jsonStore.wer.slice(-200)
  flushJsonStore()
}

export async function listWerHistory(limit = 15) {
  if (backendType === 'postgres') {
    const { rows } = await pgPool.query(`SELECT * FROM wer_evaluations ORDER BY id DESC LIMIT $1`, [limit])
    return rows
  }
  if (backendType === 'sqlite') {
    return sqliteDb.prepare(`SELECT * FROM wer_evaluations ORDER BY id DESC LIMIT ?`).all(limit)
  }
  return [...jsonStore.wer].reverse().slice(0, limit)
}

export async function clearWerHistory() {
  if (backendType === 'postgres') {
    await pgPool.query(`DELETE FROM wer_evaluations`)
    return
  }
  if (backendType === 'sqlite') {
    sqliteDb.prepare(`DELETE FROM wer_evaluations`).run()
    return
  }
  jsonStore.wer = []
  flushJsonStore()
}

export async function saveLearnedArticle(doc) {
  const title = String(doc.title || '').trim().slice(0, 200)
  const lang = String(doc.lang || 'en').split('-')[0].toLowerCase().slice(0, 12)
  const text = String(doc.text || '').trim().slice(0, 6000)
  if (!title || text.length < 40) return
  const aliasesJson = JSON.stringify((doc.aliases || []).slice(0, 12))
  const source = String(doc.source || 'Wikipedia').slice(0, 60)
  const url = String(doc.url || '').slice(0, 500)
  const domain = doc.domain ? String(doc.domain).slice(0, 80) : null

  if (backendType === 'postgres') {
    await pgPool.query(
      `INSERT INTO learned_articles (title, lang, domain, text, aliases_json, source, url)
       VALUES ($1, $2, $3, $4, $5, $6, $7)
       ON CONFLICT (title, lang) DO UPDATE SET
         text = CASE WHEN LENGTH(EXCLUDED.text) > LENGTH(learned_articles.text) THEN EXCLUDED.text ELSE learned_articles.text END,
         aliases_json = EXCLUDED.aliases_json,
         updated_at = NOW()`,
      [title, lang, domain, text, aliasesJson, source, url]
    )
    return
  }
  if (backendType === 'sqlite') {
    sqliteDb
      .prepare(
        `INSERT INTO learned_articles (title, lang, domain, text, aliases_json, source, url)
         VALUES (?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT (title, lang) DO UPDATE SET
           text = CASE WHEN LENGTH(excluded.text) > LENGTH(learned_articles.text) THEN excluded.text ELSE learned_articles.text END,
           aliases_json = excluded.aliases_json,
           updated_at = CURRENT_TIMESTAMP`
      )
      .run(title, lang, domain, text, aliasesJson, source, url)
    return
  }
  const existing = jsonStore.learned.find((d) => d.title === title && d.lang === lang)
  if (existing) {
    if (text.length > existing.text.length) existing.text = text
    existing.aliases = doc.aliases || existing.aliases
    existing.updated_at = new Date().toISOString()
  } else {
    jsonStore.learned.push({
      title,
      lang,
      domain,
      text,
      aliases: doc.aliases || [],
      source,
      url,
      created_at: new Date().toISOString()
    })
  }
  flushJsonStore()
}

export async function loadLearnedArticlesFromDb() {
  if (backendType === 'postgres') {
    const { rows } = await pgPool.query(`SELECT * FROM learned_articles ORDER BY id ASC LIMIT 5000`)
    return rows.map((r) => ({
      title: r.title,
      lang: r.lang,
      domain: r.domain,
      text: r.text,
      aliases: safeParseJson(r.aliases_json, []),
      source: r.source,
      url: r.url,
      learned: true
    }))
  }
  if (backendType === 'sqlite') {
    const rows = sqliteDb.prepare(`SELECT * FROM learned_articles ORDER BY id ASC LIMIT 5000`).all()
    return rows.map((r) => ({
      title: r.title,
      lang: r.lang,
      domain: r.domain,
      text: r.text,
      aliases: safeParseJson(r.aliases_json, []),
      source: r.source,
      url: r.url,
      learned: true
    }))
  }
  return jsonStore.learned.map((r) => ({ ...r, learned: true }))
}

export async function getAnalyticsSummary() {
  if (backendType === 'postgres') {
    const [sessionsRes, msgStatsRes, intentRes, langRes, werRes, feedbackRes] = await Promise.all([
      pgPool.query(`SELECT COUNT(*)::int AS total_sessions FROM chat_sessions`),
      pgPool.query(`
        SELECT
          COUNT(*) FILTER (WHERE role = 'user')::int AS total_queries,
          COUNT(*) FILTER (WHERE role = 'user' AND input_mode = 'voice')::int AS voice_queries,
          COUNT(*) FILTER (WHERE role = 'user' AND input_mode = 'text')::int AS text_queries,
          ROUND(AVG(asr_confidence)::numeric, 4) AS avg_asr_confidence,
          ROUND(AVG(latency_ms)::numeric, 2) AS avg_latency_ms
        FROM chat_messages
      `),
      pgPool.query(`
        SELECT intent_tag, COUNT(*)::int AS count
        FROM chat_messages
        WHERE role = 'bot' AND intent_tag IS NOT NULL
        GROUP BY intent_tag
        ORDER BY count DESC
        LIMIT 8
      `),
      pgPool.query(`
        SELECT detected_language, COUNT(*)::int AS count
        FROM chat_messages
        WHERE role = 'user' AND detected_language IS NOT NULL
        GROUP BY detected_language
        ORDER BY count DESC
        LIMIT 8
      `),
      pgPool.query(`
        SELECT COUNT(*)::int AS total_wer_runs,
               ROUND(AVG(wer)::numeric, 2) AS avg_wer,
               ROUND(AVG(word_accuracy)::numeric, 2) AS avg_word_accuracy
        FROM wer_evaluations
      `),
      pgPool.query(`
        SELECT
          COUNT(*) FILTER (WHERE feedback = 'up')::int AS positive,
          COUNT(*) FILTER (WHERE feedback = 'down')::int AS negative
        FROM chat_messages
      `)
    ])
    return {
      totalSessions: sessionsRes.rows[0]?.total_sessions || 0,
      ...(msgStatsRes.rows[0] || {}),
      topIntents: intentRes.rows,
      topLanguages: langRes.rows,
      ...(werRes.rows[0] || {}),
      feedback: feedbackRes.rows[0] || { positive: 0, negative: 0 }
    }
  }

  if (backendType === 'sqlite') {
    const totalSessions = sqliteDb.prepare(`SELECT COUNT(*) AS c FROM chat_sessions`).get().c || 0
    const msgStats = sqliteDb
      .prepare(
        `SELECT
           SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) AS total_queries,
           SUM(CASE WHEN role = 'user' AND input_mode = 'voice' THEN 1 ELSE 0 END) AS voice_queries,
           SUM(CASE WHEN role = 'user' AND input_mode = 'text' THEN 1 ELSE 0 END) AS text_queries,
           ROUND(AVG(CASE WHEN asr_confidence IS NOT NULL THEN asr_confidence END), 4) AS avg_asr_confidence,
           ROUND(AVG(CASE WHEN latency_ms IS NOT NULL THEN latency_ms END), 2) AS avg_latency_ms
         FROM chat_messages`
      )
      .get()
    const topIntents = sqliteDb
      .prepare(
        `SELECT intent_tag, COUNT(*) AS count
         FROM chat_messages WHERE role = 'bot' AND intent_tag IS NOT NULL
         GROUP BY intent_tag ORDER BY count DESC LIMIT 8`
      )
      .all()
    const topLanguages = sqliteDb
      .prepare(
        `SELECT detected_language, COUNT(*) AS count
         FROM chat_messages WHERE role = 'user' AND detected_language IS NOT NULL
         GROUP BY detected_language ORDER BY count DESC LIMIT 8`
      )
      .all()
    const werStats = sqliteDb
      .prepare(
        `SELECT COUNT(*) AS total_wer_runs,
                ROUND(AVG(wer), 2) AS avg_wer,
                ROUND(AVG(word_accuracy), 2) AS avg_word_accuracy
         FROM wer_evaluations`
      )
      .get()
    const feedback = sqliteDb
      .prepare(
        `SELECT
           SUM(CASE WHEN feedback = 'up' THEN 1 ELSE 0 END) AS positive,
           SUM(CASE WHEN feedback = 'down' THEN 1 ELSE 0 END) AS negative
         FROM chat_messages`
      )
      .get()
    return {
      totalSessions,
      total_queries: msgStats.total_queries || 0,
      voice_queries: msgStats.voice_queries || 0,
      text_queries: msgStats.text_queries || 0,
      avg_asr_confidence: msgStats.avg_asr_confidence || null,
      avg_latency_ms: msgStats.avg_latency_ms || null,
      topIntents,
      topLanguages,
      total_wer_runs: werStats.total_wer_runs || 0,
      avg_wer: werStats.avg_wer || null,
      avg_word_accuracy: werStats.avg_word_accuracy || null,
      feedback: { positive: feedback.positive || 0, negative: feedback.negative || 0 }
    }
  }

  const userMsgs = jsonStore.messages.filter((m) => m.role === 'user')
  const botMsgs = jsonStore.messages.filter((m) => m.role === 'bot')
  const confs = userMsgs.map((m) => m.asr_confidence).filter((v) => typeof v === 'number')
  const lats = botMsgs.map((m) => m.latency_ms).filter((v) => typeof v === 'number')
  return {
    totalSessions: jsonStore.sessions.length,
    total_queries: userMsgs.length,
    voice_queries: userMsgs.filter((m) => m.input_mode === 'voice').length,
    text_queries: userMsgs.filter((m) => m.input_mode === 'text').length,
    avg_asr_confidence: confs.length ? Number((confs.reduce((a, b) => a + b, 0) / confs.length).toFixed(4)) : null,
    avg_latency_ms: lats.length ? Number((lats.reduce((a, b) => a + b, 0) / lats.length).toFixed(2)) : null,
    topIntents: [],
    topLanguages: [],
    total_wer_runs: jsonStore.wer.length,
    avg_wer: jsonStore.wer.length
      ? Number((jsonStore.wer.reduce((a, b) => a + b.wer, 0) / jsonStore.wer.length).toFixed(2))
      : null,
    avg_word_accuracy: jsonStore.wer.length
      ? Number((jsonStore.wer.reduce((a, b) => a + b.word_accuracy, 0) / jsonStore.wer.length).toFixed(2))
      : null,
    feedback: {
      positive: botMsgs.filter((m) => m.feedback === 'up').length,
      negative: botMsgs.filter((m) => m.feedback === 'down').length
    }
  }
}

function safeParseJson(raw, fallback) {
  try {
    return JSON.parse(raw)
  } catch (_) {
    return fallback
  }
}
