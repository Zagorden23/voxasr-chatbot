/**
 * VoxASR Backend Server (Pure JavaScript - Hono on Cloudflare Workers/Pages)
 * Serves REST API endpoints for:
 *  - Multilingual CS & Mathematical Computing Intent Chat (/api/chat)
 *  - Certified World Languages Registry (/api/languages)
 *  - Automated Knowledge Pool & Scraped Data Status (/api/knowledge/status)
 *  - Real-Time Automated Web Scraping & Training Trigger (/api/knowledge/upgrade)
 *  - On-Demand Topic Ingestion (/api/knowledge/scrape)
 *  - Session & Transcript History via Cloudflare D1 (/api/sessions)
 *  - Word Error Rate (WER) & Character Error Rate (CER) Alignment (/api/wer)
 *  - Model Architecture & Training Metrics (/api/model)
 */

import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { serveStatic } from 'hono/cloudflare-workers'
import { predictIntent, computeWERAndAlignment, getModelSummary } from './inference.js'
import certifiedLanguages from './languages.json'

const app = new Hono()

app.use('/api/*', cors())
app.use('/static/*', serveStatic({ root: './' }))

async function ensureSchema(db) {
  if (!db) return
  await db
    .prepare(`
      CREATE TABLE IF NOT EXISTS chat_sessions (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL DEFAULT 'Voice ASR Session',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `)
    .run()

  await db
    .prepare(`
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
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `)
    .run()

  await db
    .prepare(`
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
      )
    `)
    .run()
}

// Health & Model Info
app.get('/api/model', (c) => {
  return c.json({
    ok: true,
    model: getModelSummary()
  })
})

// Certified World Languages List
app.get('/api/languages', (c) => {
  return c.json({
    ok: true,
    total: certifiedLanguages.length,
    languages: certifiedLanguages
  })
})

// Knowledge Pool & Auto-Upgrade Status
app.get('/api/knowledge/status', async (c) => {
  // First attempt to query the local upgrade daemon if running
  try {
    const daemonResp = await fetch('http://127.0.0.1:3001/status', {
      headers: { 'Accept': 'application/json' }
    })
    if (daemonResp.ok) {
      const data = await daemonResp.json()
      return c.json(data)
    }
  } catch (_) {
    // Daemon not running, fall back to in-bundle model stats
  }

  const summary = getModelSummary()
  return c.json({
    ok: true,
    is_upgrading: false,
    last_progress: 'Idle (Daemon offline)',
    model_version: summary.version || '2.0.0',
    trained_at: summary.trained_at || 'N/A',
    train_accuracy: summary.train_accuracy || 0.868,
    macro_f1: summary.macro_f1 || 0.889,
    vocab_size: summary.vocab_size || 5000,
    num_classes: summary.num_classes || 18,
    base_patterns: summary.base_patterns || 1721,
    total_scraped_articles: summary.scraped_articles_included || 36,
    certified_languages_count: certifiedLanguages.length,
    domains_summary: (summary.intents || []).map((it) => ({
      key: it.tag,
      title: it.title,
      article_count: it.pattern_count
    }))
  })
})

// Trigger Automated Web Scraping & ML-DL Continuous Retraining
app.post('/api/knowledge/upgrade', async (c) => {
  try {
    const daemonResp = await fetch('http://127.0.0.1:3001/upgrade', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    })
    const data = await daemonResp.json()
    return c.json(data)
  } catch (err) {
    return c.json({
      ok: false,
      error:
        'Continuous upgrade daemon is not reachable on port 3001. You can run it via `python webapp/training/upgrade_server.py` or `npm run upgrade`!'
    }, 503)
  }
})

// On-Demand Topic Web Scraping & Ingestion
app.post('/api/knowledge/scrape', async (c) => {
  try {
    const body = await c.req.json()
    const topic = String(body.topic || '').trim()
    const lang = String(body.lang || 'en')
    if (!topic) {
      return c.json({ ok: false, error: 'Topic string is required' }, 400)
    }

    const daemonResp = await fetch('http://127.0.0.1:3001/scrape', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic, lang })
    })
    const data = await daemonResp.json()
    return c.json(data)
  } catch (err) {
    return c.json({
      ok: false,
      error:
        'Upgrade daemon is not running on port 3001. Run `python webapp/training/upgrade_server.py` to enable real-time scraping!'
    }, 503)
  }
})

// List recent sessions
app.get('/api/sessions', async (c) => {
  const db = c.env?.DB
  if (!db) {
    return c.json({ ok: true, sessions: [] })
  }
  try {
    await ensureSchema(db)
    const { results } = await db
      .prepare(
        `SELECT s.id, s.title, s.created_at, s.updated_at,
                COUNT(m.id) AS message_count
         FROM chat_sessions s
         LEFT JOIN chat_messages m ON m.session_id = s.id
         GROUP BY s.id
         ORDER BY s.updated_at DESC
         LIMIT 25`
      )
      .all()
    return c.json({ ok: true, sessions: results || [] })
  } catch (err) {
    return c.json({ ok: false, error: err.message }, 500)
  }
})

// Fetch messages for a session
app.get('/api/sessions/:id', async (c) => {
  const db = c.env?.DB
  const sessionId = c.req.param('id')
  if (!db) {
    return c.json({ ok: true, session: { id: sessionId }, messages: [] })
  }
  try {
    await ensureSchema(db)
    const session = await db
      .prepare(`SELECT * FROM chat_sessions WHERE id = ?`)
      .bind(sessionId)
      .first()
    const { results } = await db
      .prepare(
        `SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id ASC`
      )
      .bind(sessionId)
      .all()
    return c.json({
      ok: true,
      session: session || { id: sessionId, title: 'Voice ASR Session' },
      messages: results || []
    })
  } catch (err) {
    return c.json({ ok: false, error: err.message }, 500)
  }
})

// Delete a session
app.delete('/api/sessions/:id', async (c) => {
  const db = c.env?.DB
  const sessionId = c.req.param('id')
  if (db) {
    await ensureSchema(db)
    await db.prepare(`DELETE FROM chat_messages WHERE session_id = ?`).bind(sessionId).run()
    await db.prepare(`DELETE FROM chat_sessions WHERE id = ?`).bind(sessionId).run()
  }
  return c.json({ ok: true })
})

// Main Chatbot Endpoint
app.post('/api/chat', async (c) => {
  try {
    const body = await c.req.json()
    const message = String(body.message || '').trim()
    const inputMode = body.inputMode === 'voice' ? 'voice' : 'text'
    const asrConfidence =
      typeof body.asrConfidence === 'number' ? Number(body.asrConfidence.toFixed(4)) : null
    const sessionId = body.sessionId || `sess_${Date.now().toString(36)}`

    if (!message) {
      return c.json({ ok: false, error: 'Message cannot be empty' }, 400)
    }

    const prediction = predictIntent(message)
    const detectedLangCode = prediction.detectedLanguage?.lang || 'en-US'

    // Persist in Cloudflare D1 if bound
    const db = c.env?.DB
    if (db) {
      try {
        await ensureSchema(db)
        const existing = await db
          .prepare(`SELECT id FROM chat_sessions WHERE id = ?`)
          .bind(sessionId)
          .first()

        const shortTitle =
          message.length > 38 ? `${message.slice(0, 38)}...` : message

        if (!existing) {
          await db
            .prepare(`INSERT INTO chat_sessions (id, title) VALUES (?, ?)`)
            .bind(sessionId, shortTitle)
            .run()
        } else {
          await db
            .prepare(`UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?`)
            .bind(sessionId)
            .run()
        }

        await db
          .prepare(
            `INSERT INTO chat_messages (session_id, role, content, input_mode, asr_confidence, intent_tag, intent_confidence, detected_language)
             VALUES (?, 'user', ?, ?, ?, ?, ?, ?)`
          )
          .bind(
            sessionId,
            message,
            inputMode,
            asrConfidence,
            prediction.intentTag,
            prediction.confidence,
            detectedLangCode
          )
          .run()

        await db
          .prepare(
            `INSERT INTO chat_messages (session_id, role, content, input_mode, asr_confidence, intent_tag, intent_confidence, detected_language)
             VALUES (?, 'bot', ?, 'system', NULL, ?, ?, ?)`
          )
          .bind(sessionId, prediction.reply, prediction.intentTag, prediction.confidence, detectedLangCode)
          .run()
      } catch (_) {
        // Non-fatal DB fallback
      }
    }

    return c.json({
      ok: true,
      sessionId,
      userMessage: {
        content: message,
        inputMode,
        asrConfidence
      },
      botResponse: prediction
    })
  } catch (err) {
    return c.json({ ok: false, error: err.message }, 500)
  }
})

// Word Error Rate (WER) & Character Error Rate (CER) Evaluator Endpoint
app.post('/api/wer', async (c) => {
  try {
    const body = await c.req.json()
    const reference = String(body.reference || '')
    const hypothesis = String(body.hypothesis || '')

    if (!reference.trim()) {
      return c.json({ ok: false, error: 'Reference ground-truth text is required' }, 400)
    }

    const metrics = computeWERAndAlignment(reference, hypothesis)

    const db = c.env?.DB
    if (db) {
      try {
        await ensureSchema(db)
        await db
          .prepare(
            `INSERT INTO wer_evaluations
             (reference_text, hypothesis_text, wer, cer, substitutions, deletions, insertions)
             VALUES (?, ?, ?, ?, ?, ?, ?)`
          )
          .bind(
            reference.trim(),
            hypothesis.trim(),
            metrics.wer,
            metrics.cer,
            metrics.substitutions,
            metrics.deletions,
            metrics.insertions
          )
          .run()
      } catch (_) {}
    }

    return c.json({ ok: true, metrics })
  } catch (err) {
    return c.json({ ok: false, error: err.message }, 500)
  }
})

app.get('/api/wer/history', async (c) => {
  const db = c.env?.DB
  if (!db) return c.json({ ok: true, history: [] })
  try {
    await ensureSchema(db)
    const { results } = await db
      .prepare(`SELECT * FROM wer_evaluations ORDER BY id DESC LIMIT 10`)
      .all()
    return c.json({ ok: true, history: results || [] })
  } catch (err) {
    return c.json({ ok: false, error: err.message }, 500)
  }
})

// Main Single-Page React Application Shell
app.get('*', (c) => {
  return c.html(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>VoxASR — Multilingual CS & Mathematical AI Assistant</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.5.2/css/all.min.css" />
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body class="bg-slate-950 text-slate-100 antialiased selection:bg-indigo-500 selection:text-white">
  <div id="root"></div>

  <!-- React 18 (Pure JavaScript SPA without TypeScript) -->
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="/static/app.js"></script>
</body>
</html>`)
})

export default app
