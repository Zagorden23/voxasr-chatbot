/**
 * VoxASR Commercial Backend Server (Pure JavaScript - Hono on Node.js / Render)
 * Serves REST API endpoints + SPA frontend for:
 *  - Multilingual CS & Mathematical Computing AI Assistant (/api/chat)
 *  - Message Feedback & Session Management / Export (/api/sessions, /api/chat/feedback)
 *  - Certified 35 World Languages Registry (/api/languages)
 *  - Live BM25 Knowledge Search, Web Scraping & Auto-Upgrade (/api/knowledge/*)
 *  - Word Error Rate (WER) & CER Benchmarking Lab (/api/wer)
 *  - Commercial Usage Analytics (/api/analytics) & Render Health Checks (/health)
 */

import fs from 'node:fs'
import path from 'node:path'
import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { serveStatic } from '@hono/node-server/serve-static'
import {
  clearWerHistory,
  deleteSession,
  getAnalyticsSummary,
  getDbBackend,
  getLastBotContextTopic,
  getSessionWithMessages,
  listSessions,
  listWerHistory,
  pingDatabase,
  recordChatTurn,
  recordWerEvaluation,
  renameSession,
  saveMessageFeedback
} from './db.js'
import {
  computeWERAndAlignment,
  generateIntelligentReply,
  getModelSummary,
  reloadModel
} from './inference.js'
import {
  checkHotReload,
  getKnowledgeIndex,
  browseAndLearnTopic,
  getManifestMeta,
  getUpgradeState,
  runNativeKnowledgeUpgrade
} from './web_learner.js'

const LANGUAGES_PATH = path.resolve(process.cwd(), 'src', 'languages.json')
const certifiedLanguages = JSON.parse(fs.readFileSync(LANGUAGES_PATH, 'utf8'))

const STARTED_AT = Date.now()
const RATE_LIMIT_WINDOW_MS = 60_000
const RATE_LIMIT_MAX = Number(process.env.RATE_LIMIT_MAX || 180)
const rateBuckets = new Map()

// Periodic cleanup of expired rate-limit buckets
setInterval(() => {
  const now = Date.now()
  for (const [ip, bucket] of rateBuckets.entries()) {
    if (now > bucket.resetAt) rateBuckets.delete(ip)
  }
}, 60_000).unref?.()

const app = new Hono()

// Security Headers
app.use('*', async (c, next) => {
  await next()
  c.header('X-Content-Type-Options', 'nosniff')
  c.header('X-Frame-Options', 'SAMEORIGIN')
  c.header('Referrer-Policy', 'strict-origin-when-cross-origin')
  c.header('Permissions-Policy', 'microphone=(self)')
})

// CORS for API routes
const allowedOrigin = process.env.CORS_ORIGIN || '*'
app.use('/api/*', cors({ origin: allowedOrigin }))

// Sliding-Window API Rate Limiter
app.use('/api/*', async (c, next) => {
  const pathName = c.req.path
  if (pathName === '/api/health') return next()

  const forwarded = c.req.header('x-forwarded-for') || ''
  const clientIp = forwarded.split(',')[0].trim() || c.req.header('x-real-ip') || 'local'
  const now = Date.now()
  let bucket = rateBuckets.get(clientIp)
  if (!bucket || now > bucket.resetAt) {
    bucket = { count: 0, resetAt: now + RATE_LIMIT_WINDOW_MS }
    rateBuckets.set(clientIp, bucket)
  }
  bucket.count++

  const remaining = Math.max(0, RATE_LIMIT_MAX - bucket.count)
  c.header('X-RateLimit-Limit', String(RATE_LIMIT_MAX))
  c.header('X-RateLimit-Remaining', String(remaining))
  c.header('X-RateLimit-Reset', String(Math.ceil(bucket.resetAt / 1000)))

  if (bucket.count > RATE_LIMIT_MAX) {
    const retryAfter = Math.max(1, Math.ceil((bucket.resetAt - now) / 1000))
    c.header('Retry-After', String(retryAfter))
    return c.json({ ok: false, error: 'Rate limit exceeded. Please wait a moment and try again.' }, 429)
  }

  return next()
})

// Static assets under ./public/static/*
app.use('/static/*', serveStatic({ root: './public' }))

// Render Health Check Endpoints
app.get('/health', async (c) => {
  const kbStats = getKnowledgeIndex().stats()
  const model = getModelSummary()
  const dbOk = await pingDatabase()
  return c.json(
    {
      ok: dbOk,
      status: dbOk ? 'healthy' : 'degraded',
      service: 'voxasr-chatbot',
      version: model.version || '3.0.0',
      uptimeSeconds: Math.round((Date.now() - STARTED_AT) / 1000),
      database: getDbBackend(),
      kbArticles: kbStats.articles,
      kbLearned: kbStats.learned,
      vocabSize: model.vocab_size
    },
    dbOk ? 200 : 503
  )
})

app.get('/api/health', (c) => c.redirect('/health'))

// Model Architecture & Metrics
app.get('/api/model', async (c) => {
  await checkHotReload(reloadModel)
  const kbStats = getKnowledgeIndex().stats()
  return c.json({
    ok: true,
    model: {
      ...getModelSummary(),
      kb_articles: kbStats.articles,
      kb_learned: kbStats.learned,
      kb_terms: kbStats.terms
    }
  })
})

// Certified World Languages List (35 Languages)
app.get('/api/languages', (c) => {
  return c.json({
    ok: true,
    total: certifiedLanguages.length,
    languages: certifiedLanguages
  })
})

// Commercial Usage & Performance Analytics
app.get('/api/analytics', async (c) => {
  try {
    const summary = await getAnalyticsSummary()
    const kbStats = getKnowledgeIndex().stats()
    return c.json({
      ok: true,
      analytics: {
        ...summary,
        kbArticles: kbStats.articles,
        kbLearned: kbStats.learned,
        kbTerms: kbStats.terms,
        databaseBackend: getDbBackend(),
        uptimeSeconds: Math.round((Date.now() - STARTED_AT) / 1000)
      }
    })
  } catch (err) {
    return c.json({ ok: false, error: 'Failed to load analytics' }, 500)
  }
})

// Knowledge Pool Search (Typeahead & Knowledge Explorer)
app.get('/api/knowledge/search', (c) => {
  const q = String(c.req.query('q') || '').trim()
  const lang = String(c.req.query('lang') || 'en').split('-')[0].toLowerCase()
  const limit = Math.min(24, Math.max(1, Number(c.req.query('limit') || 12)))
  const kbIndex = getKnowledgeIndex()

  if (!q) {
    // Return sample featured articles across domains
    const sample = kbIndex.docs.slice(0, limit).map((d) => ({
      title: d.title,
      domain: d.domain || 'general',
      source: d.source,
      lang: d.lang,
      url: d.url,
      aliases: d.aliases,
      learned: d.learned,
      score: 1,
      summary: d.text.slice(0, 300)
    }))
    return c.json({ ok: true, total: kbIndex.size, results: sample })
  }

  const results = kbIndex.searchArticles(q, lang, limit)
  return c.json({ ok: true, total: kbIndex.size, query: q, results })
})

// Knowledge Pool & Auto-Upgrade Status
app.get('/api/knowledge/status', async (c) => {
  await checkHotReload(reloadModel)

  const daemonUrl = (process.env.UPGRADE_DAEMON_URL || '').trim()
  if (daemonUrl) {
    try {
      const daemonResp = await fetch(`${daemonUrl.replace(/\/$/, '')}/status`, {
        headers: { Accept: 'application/json' },
        signal: AbortSignal.timeout(1500)
      })
      if (daemonResp.ok) {
        const data = await daemonResp.json()
        return c.json(data)
      }
    } catch (_) {}
  }

  const summary = getModelSummary()
  const kbStats = getKnowledgeIndex().stats()
  const manifest = getManifestMeta()
  const upgradeState = getUpgradeState()

  const domainCounts = {}
  for (const doc of getKnowledgeIndex().docs) {
    const key = doc.domain || 'general'
    domainCounts[key] = (domainCounts[key] || 0) + 1
  }

  const domainsSummary = (summary.intents || [])
    .filter((it) => domainCounts[it.tag] || !['greeting', 'thanks_goodbye', 'language_support', 'wer_evaluation', 'knowledge_auto_upgrade'].includes(it.tag))
    .map((it) => ({
      key: it.tag,
      title: it.title,
      article_count: domainCounts[it.tag] || it.pattern_count
    }))

  return c.json({
    ok: true,
    is_upgrading: upgradeState.is_upgrading,
    last_progress: upgradeState.last_progress,
    last_report: upgradeState.last_report,
    model_version: summary.version || '2.0.0',
    trained_at: summary.trained_at || manifest.generated_at || 'N/A',
    train_accuracy: summary.train_accuracy || 0,
    macro_f1: summary.macro_f1 || 0,
    cv_accuracy_mean: summary.cv_accuracy_mean || 0,
    vocab_size: summary.vocab_size || 5000,
    num_classes: summary.num_classes || 18,
    base_patterns: summary.base_patterns || 0,
    total_scraped_articles: kbStats.articles,
    learned_articles: kbStats.learned,
    kb_terms: kbStats.terms,
    certified_languages_count: certifiedLanguages.length,
    domains_summary: domainsSummary
  })
})

// Trigger Automated Web Scraping & Knowledge Pool Upgrade
app.post('/api/knowledge/upgrade', async (c) => {
  const daemonUrl = (process.env.UPGRADE_DAEMON_URL || '').trim()
  if (daemonUrl) {
    try {
      const daemonResp = await fetch(`${daemonUrl.replace(/\/$/, '')}/upgrade`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
        signal: AbortSignal.timeout(4000)
      })
      const data = await daemonResp.json()
      return c.json(data, daemonResp.status)
    } catch (_) {}
  }

  const result = await runNativeKnowledgeUpgrade()
  return c.json(result, result.ok ? 200 : 409)
})

// On-Demand Topic Web Scraping & Ingestion
app.post('/api/knowledge/scrape', async (c) => {
  let body = {}
  try {
    body = await c.req.json()
  } catch (_) {
    return c.json({ ok: false, error: 'Invalid JSON request body' }, 400)
  }

  const topic = String(body.topic || '').trim().slice(0, 180)
  const lang = String(body.lang || 'en').split('-')[0].toLowerCase()
  if (!topic || topic.length < 2) {
    return c.json({ ok: false, error: 'A valid topic string (at least 2 characters) is required' }, 400)
  }

  const daemonUrl = (process.env.UPGRADE_DAEMON_URL || '').trim()
  if (daemonUrl) {
    try {
      const daemonResp = await fetch(`${daemonUrl.replace(/\/$/, '')}/scrape`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, lang }),
        signal: AbortSignal.timeout(5000)
      })
      const data = await daemonResp.json()
      if (daemonResp.ok) return c.json(data, daemonResp.status)
    } catch (_) {}
  }

  const directUrl = /^https?:\/\//i.test(topic) ? [topic] : body.url && /^https?:\/\//i.test(String(body.url)) ? [String(body.url).trim()] : []
  const depthMode = String(body.depth || 'detailed')
  const resBrowse = await browseAndLearnTopic(topic, lang, body.domain || null, {
    depthMode,
    directUrls: directUrl
  })
  const doc = resBrowse.primary
  if (!doc) {
    return c.json(
      {
        ok: false,
        error: `Could not extract readable public web content matching "${topic}".`,
        blockedUrls: resBrowse.blockedUrls || []
      },
      404
    )
  }

  const stats = getKnowledgeIndex().stats()
  return c.json({
    ok: true,
    message: `Browsed ${resBrowse.docs.length} live page(s) and ingested "${doc.title}" (${stats.articles} total articles).`,
    article: {
      title: doc.title,
      source: doc.source,
      url: doc.url,
      summary: String(doc.extract || doc.text || '').slice(0, 320)
    },
    docs: resBrowse.docs.map((d) => ({
      title: d.title,
      source: d.source,
      url: d.url,
      summary: String(d.extract || d.text || '').slice(0, 260)
    })),
    scrapedHosts: resBrowse.scrapedHosts,
    blockedUrls: resBrowse.blockedUrls,
    total_articles: stats.articles,
    learned_articles: stats.learned
  })
})

app.post('/api/web/browse', async (c) => {
  let body = {}
  try {
    body = await c.req.json()
  } catch (_) {
    return c.json({ ok: false, error: 'Invalid JSON request body' }, 400)
  }
  const query = String(body.query || body.topic || body.url || '').trim().slice(0, 240)
  if (!query || query.length < 2) {
    return c.json({ ok: false, error: 'A valid query or URL is required' }, 400)
  }
  const lang = String(body.lang || 'en').split('-')[0].toLowerCase()
  const directUrls = /^https?:\/\//i.test(query) ? [query] : []
  const resBrowse = await browseAndLearnTopic(query, lang, body.domain || null, {
    depthMode: String(body.depth || 'standard'),
    directUrls
  })
  return c.json({
    ok: Boolean(resBrowse.docs?.length),
    query,
    docs: resBrowse.docs || [],
    scrapedHosts: resBrowse.scrapedHosts || [],
    blockedUrls: resBrowse.blockedUrls || []
  })
})

// Session Management
app.get('/api/sessions', async (c) => {
  try {
    const sessions = await listSessions(30)
    return c.json({ ok: true, sessions })
  } catch (err) {
    return c.json({ ok: false, error: 'Failed to list sessions' }, 500)
  }
})

app.get('/api/sessions/:id', async (c) => {
  const sessionId = String(c.req.param('id') || '').slice(0, 100)
  if (!sessionId) return c.json({ ok: false, error: 'Invalid session id' }, 400)
  try {
    const data = await getSessionWithMessages(sessionId)
    return c.json({ ok: true, ...data })
  } catch (err) {
    return c.json({ ok: false, error: 'Failed to load session' }, 500)
  }
})

app.patch('/api/sessions/:id', async (c) => {
  const sessionId = String(c.req.param('id') || '').slice(0, 100)
  let body = {}
  try {
    body = await c.req.json()
  } catch (_) {
    return c.json({ ok: false, error: 'Invalid JSON body' }, 400)
  }
  const title = String(body.title || '').trim()
  if (!title) return c.json({ ok: false, error: 'Title is required' }, 400)
  try {
    const updated = await renameSession(sessionId, title)
    return c.json({ ok: updated })
  } catch (err) {
    return c.json({ ok: false, error: 'Failed to rename session' }, 500)
  }
})

app.delete('/api/sessions/:id', async (c) => {
  const sessionId = String(c.req.param('id') || '').slice(0, 100)
  try {
    await deleteSession(sessionId)
    return c.json({ ok: true })
  } catch (err) {
    return c.json({ ok: false, error: 'Failed to delete session' }, 500)
  }
})

app.get('/api/sessions/:id/export', async (c) => {
  const sessionId = String(c.req.param('id') || '').slice(0, 100)
  const format = String(c.req.query('format') || 'markdown').toLowerCase()
  try {
    const { session, messages } = await getSessionWithMessages(sessionId)
    if (format === 'json') {
      c.header('Content-Disposition', `attachment; filename="voxasr-${sessionId}.json"`)
      return c.json({ session, messages })
    }
    const lines = [
      `# ${session.title || 'VoxASR Session'}`,
      `- Session ID: \`${sessionId}\``,
      `- Exported: ${new Date().toISOString()}`,
      ''
    ]
    for (const m of messages) {
      const speaker = m.role === 'user' ? `**User** (${m.input_mode || 'text'})` : `**VoxASR** (${m.intent_tag || 'assistant'})`
      lines.push(`### ${speaker}`)
      lines.push(m.content)
      lines.push('')
    }
    c.header('Content-Type', 'text/markdown; charset=utf-8')
    c.header('Content-Disposition', `attachment; filename="voxasr-${sessionId}.md"`)
    return c.body(lines.join('\n'))
  } catch (err) {
    return c.json({ ok: false, error: 'Failed to export session' }, 500)
  }
})

// Main Chatbot Endpoint
app.post('/api/chat', async (c) => {
  let body = {}
  try {
    body = await c.req.json()
  } catch (_) {
    return c.json({ ok: false, error: 'Invalid JSON request body' }, 400)
  }

  const message = String(body.message || '').trim()
  if (!message) {
    return c.json({ ok: false, error: 'Message cannot be empty' }, 400)
  }
  if (message.length > 4000) {
    return c.json({ ok: false, error: 'Message exceeds maximum length of 4000 characters' }, 400)
  }

  const inputMode = body.inputMode === 'voice' ? 'voice' : 'text'
  const asrConfidence =
    typeof body.asrConfidence === 'number' && Number.isFinite(body.asrConfidence)
      ? Number(Math.max(0, Math.min(1, body.asrConfidence)).toFixed(4))
      : null
  const asrLang = String(body.asrLang || '').trim().slice(0, 20)
  const sessionId = String(body.sessionId || `sess_${Date.now().toString(36)}`).slice(0, 100)
  const autoLearn = body.autoLearn !== false
  const skipLiveWeb = body.skipLiveWeb === true

  try {
    await checkHotReload(reloadModel)
    const contextTopic = await getLastBotContextTopic(sessionId).catch(() => null)
    const prediction = await generateIntelligentReply(message, {
      langHint: asrLang,
      contextTopic,
      autoLearn,
      skipLiveWeb
    })

    const detectedLangCode = prediction.detectedLanguage?.lang || asrLang || 'en-US'
    const shortTitle = message.length > 42 ? `${message.slice(0, 42)}...` : message

    let botMessageId = null
    try {
      botMessageId = await recordChatTurn({
        sessionId,
        sessionTitle: shortTitle,
        language: detectedLangCode,
        userMessage: { content: message, inputMode, asrConfidence },
        botMessage: prediction
      })
    } catch (dbErr) {
      console.warn('[Chat] DB persistence failed:', dbErr.message)
    }

    return c.json({
      ok: true,
      sessionId,
      botMessageId,
      userMessage: {
        content: message,
        inputMode,
        asrConfidence
      },
      botResponse: prediction
    })
  } catch (err) {
    console.error('[Chat] Error:', err)
    return c.json({ ok: false, error: 'Internal inference error' }, 500)
  }
})

// Message Feedback (Commercial Quality Loop)
app.post('/api/chat/feedback', async (c) => {
  let body = {}
  try {
    body = await c.req.json()
  } catch (_) {
    return c.json({ ok: false, error: 'Invalid JSON body' }, 400)
  }
  const messageId = Number(body.messageId)
  const feedback = body.feedback === 'up' || body.feedback === 'down' ? body.feedback : null
  if (!Number.isFinite(messageId) || !feedback) {
    return c.json({ ok: false, error: 'Valid messageId and feedback ("up" or "down") required' }, 400)
  }
  try {
    const ok = await saveMessageFeedback(messageId, feedback)
    return c.json({ ok })
  } catch (err) {
    return c.json({ ok: false, error: 'Failed to save feedback' }, 500)
  }
})

// Word Error Rate (WER) & CER Evaluator Endpoint
app.post('/api/wer', async (c) => {
  let body = {}
  try {
    body = await c.req.json()
  } catch (_) {
    return c.json({ ok: false, error: 'Invalid JSON body' }, 400)
  }

  const reference = String(body.reference || '').slice(0, 4000)
  const hypothesis = String(body.hypothesis || '').slice(0, 4000)
  const language = String(body.language || 'en-US').slice(0, 20)
  const persist = body.persist !== false

  if (!reference.trim()) {
    return c.json({ ok: false, error: 'Reference ground-truth text is required' }, 400)
  }

  const metrics = computeWERAndAlignment(reference, hypothesis)

  if (persist) {
    try {
      await recordWerEvaluation(reference.trim(), hypothesis.trim(), metrics, language)
    } catch (err) {
      console.warn('[WER] DB persist failed:', err.message)
    }
  }

  return c.json({ ok: true, metrics })
})

app.get('/api/wer/history', async (c) => {
  try {
    const history = await listWerHistory(15)
    return c.json({ ok: true, history })
  } catch (err) {
    return c.json({ ok: false, error: 'Failed to load WER history' }, 500)
  }
})

app.delete('/api/wer/history', async (c) => {
  try {
    await clearWerHistory()
    return c.json({ ok: true })
  } catch (err) {
    return c.json({ ok: false, error: 'Failed to clear WER history' }, 500)
  }
})

// Inline SVG Favicon (avoids 404 in browsers)
app.get('/favicon.ico', (c) => {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#4f46e5"/><stop offset="100%" stop-color="#10b981"/></linearGradient></defs><rect width="64" height="64" rx="16" fill="url(#g)"/><path d="M32 14a7 7 0 0 0-7 7v12a7 7 0 0 0 14 0V21a7 7 0 0 0-7-7zm-12 19a12 12 0 0 0 24 0m-12 12v7m-7 0h14" stroke="#fff" stroke-width="4" stroke-linecap="round" fill="none"/></svg>`
  c.header('Content-Type', 'image/svg+xml')
  c.header('Cache-Control', 'public, max-age=86400')
  return c.body(svg)
})

// Main Single-Page React Application Shell
app.get('*', (c) => {
  return c.html(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="theme-color" content="#020617" />
  <meta name="description" content="VoxASR — Commercial Multilingual Voice ASR & Computer Science / Mathematical Computing AI Assistant across 35 Certified Languages." />
  <title>VoxASR — Multilingual CS & Mathematical AI Assistant</title>
  <link rel="icon" href="/favicon.ico" />
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.5.2/css/all.min.css" />
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body class="bg-slate-950 text-slate-100 antialiased selection:bg-indigo-500 selection:text-white">
  <div id="root"></div>

  <!-- React 18 (Pure JavaScript SPA) -->
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="/static/app.js"></script>
</body>
</html>`)
})

export default app
