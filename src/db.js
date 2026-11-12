/**
 * Commercial NoSQL Database Layer for VoxASR (MongoDB Only)
 * --------------------------------------------------------
 * Connects exclusively to MongoDB via the official `mongodb` Node.js driver.
 * Reads `VOXASR_MONGODB_URI` (local `.env` override) or `MONGODB_URI` (Render/Cloud),
 * plus optional `MONGODB_DB` (defaults to the URI database path or `voxasr`).
 *
 * Collections:
 *   - `counters`          Atomic sequence generator for numeric message/WER IDs
 *   - `chat_sessions`     Session metadata (title, language, timestamps)
 *   - `chat_messages`     Multilingual user & bot turns, ASR confidence, latency, feedback
 *   - `wer_evaluations`   ASR Word Error Rate & CER benchmark history
 *   - `learned_articles`  Dynamically harvested Wikipedia & ArXiv articles
 */

import fs from 'node:fs'
import path from 'node:path'
import { MongoClient } from 'mongodb'

let mongoClient = null
let mongoDb = null
let backendType = 'disconnected'

function loadDotEnvIfPresent() {
  const envPath = path.resolve(process.cwd(), '.env')
  if (!fs.existsSync(envPath)) return
  try {
    const raw = fs.readFileSync(envPath, 'utf8')
    for (const line of raw.split(/\r?\n/)) {
      const trimmed = line.trim()
      if (!trimmed || trimmed.startsWith('#')) continue
      const eq = trimmed.indexOf('=')
      if (eq <= 0) continue
      const key = trimmed.slice(0, eq).trim()
      let val = trimmed.slice(eq + 1).trim()
      if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
        val = val.slice(1, -1)
      }
      if (!process.env[key]) {
        process.env[key] = val
      }
    }
  } catch (err) {
    console.warn('[DB] Failed reading local .env:', err.message)
  }
}

function resolveDbName(uri, explicitDbName) {
  if (explicitDbName && String(explicitDbName).trim()) {
    return String(explicitDbName).trim()
  }
  const envDb = (process.env.MONGODB_DB || '').trim()
  if (envDb) return envDb
  try {
    const parsed = new URL(uri)
    const cleanPath = parsed.pathname.replace(/^\/+/, '').trim()
    if (cleanPath) return cleanPath
  } catch (_) {}
  return 'voxasr'
}

function collections() {
  if (!mongoDb) {
    throw new Error('MongoDB is not initialized. Call initDatabase() first.')
  }
  return {
    counters: mongoDb.collection('counters'),
    sessions: mongoDb.collection('chat_sessions'),
    messages: mongoDb.collection('chat_messages'),
    wer: mongoDb.collection('wer_evaluations'),
    learned: mongoDb.collection('learned_articles')
  }
}

async function ensureIndexes() {
  const { sessions, messages, wer, learned } = collections()
  await Promise.all([
    sessions.createIndex({ id: 1 }, { unique: true }),
    sessions.createIndex({ updated_at: -1 }),
    messages.createIndex({ id: 1 }, { unique: true }),
    messages.createIndex({ session_id: 1, id: 1 }),
    messages.createIndex({ role: 1, intent_tag: 1 }),
    messages.createIndex({ role: 1, detected_language: 1 }),
    wer.createIndex({ id: -1 }),
    learned.createIndex({ title: 1, lang: 1 }, { unique: true })
  ])
}

async function nextSequence(counterName, increment = 1) {
  const { counters } = collections()
  const doc = await counters.findOneAndUpdate(
    { _id: counterName },
    { $inc: { seq: increment } },
    { upsert: true, returnDocument: 'after' }
  )
  const seq = Number(doc?.seq ?? doc?.value?.seq ?? increment)
  return seq - increment + 1
}

export async function initDatabase(options = {}) {
  loadDotEnvIfPresent()

  if (options.db) {
    mongoDb = options.db
    mongoClient = options.client || null
    backendType = 'mongodb'
    await ensureIndexes()
    return backendType
  }

  const uri = String(options.uri || process.env.VOXASR_MONGODB_URI || process.env.MONGODB_URI || '').trim()
  if (!uri) {
    throw new Error('MongoDB connection URI missing: set MONGODB_URI (or VOXASR_MONGODB_URI for local dev).')
  }

  const dbName = resolveDbName(uri, options.dbName)
  const client = new MongoClient(uri, {
    maxPoolSize: Number(process.env.MONGO_POOL_MAX || 20),
    minPoolSize: 1,
    serverSelectionTimeoutMS: Number(process.env.MONGO_TIMEOUT_MS || 10000),
    retryWrites: true
  })

  try {
    await client.connect()
    const db = client.db(dbName)
    await db.command({ ping: 1 })
    mongoClient = client
    mongoDb = db
    backendType = 'mongodb'
    await ensureIndexes()
    console.log(`[DB] Connected to MongoDB database "${dbName}"`)
    return backendType
  } catch (err) {
    await client.close().catch(() => {})
    mongoClient = null
    mongoDb = null
    backendType = 'disconnected'
    throw err
  }
}

export function getDbBackend() {
  return backendType
}

export async function pingDatabase() {
  if (!mongoDb) return false
  try {
    const res = await mongoDb.command({ ping: 1 })
    return res?.ok === 1
  } catch (_) {
    return false
  }
}

export async function closeDatabase() {
  if (mongoClient && typeof mongoClient.close === 'function') {
    await mongoClient.close().catch(() => {})
  }
  mongoClient = null
  mongoDb = null
  backendType = 'disconnected'
}

export async function listSessions(limit = 30) {
  const { sessions } = collections()
  return sessions
    .aggregate([
      { $sort: { updated_at: -1 } },
      { $limit: limit },
      {
        $lookup: {
          from: 'chat_messages',
          localField: 'id',
          foreignField: 'session_id',
          as: '_msgs'
        }
      },
      {
        $project: {
          _id: 0,
          id: 1,
          title: 1,
          language: 1,
          created_at: 1,
          updated_at: 1,
          message_count: { $size: '$_msgs' }
        }
      }
    ])
    .toArray()
}

export async function getSessionWithMessages(sessionId) {
  const { sessions, messages } = collections()
  const [session, msgList] = await Promise.all([
    sessions.findOne({ id: sessionId }, { projection: { _id: 0 } }),
    messages.find({ session_id: sessionId }, { projection: { _id: 0 } }).sort({ id: 1 }).toArray()
  ])
  return {
    session: session || { id: sessionId, title: 'Voice ASR Session', language: 'en-US' },
    messages: msgList
  }
}

export async function renameSession(sessionId, title) {
  const cleanTitle = String(title || '').trim().slice(0, 120)
  if (!cleanTitle) return false
  const { sessions } = collections()
  const res = await sessions.updateOne(
    { id: sessionId },
    { $set: { title: cleanTitle, updated_at: new Date() } }
  )
  return res.matchedCount > 0
}

export async function deleteSession(sessionId) {
  const { sessions, messages } = collections()
  await Promise.all([
    messages.deleteMany({ session_id: sessionId }),
    sessions.deleteOne({ id: sessionId })
  ])
  return true
}

export async function recordChatTurn({ sessionId, sessionTitle, language, userMessage, botMessage }) {
  const { sessions, messages } = collections()
  const now = new Date()

  await sessions.updateOne(
    { id: sessionId },
    {
      $set: { language, updated_at: now },
      $setOnInsert: { id: sessionId, title: sessionTitle, created_at: now }
    },
    { upsert: true }
  )

  const startId = await nextSequence('chat_messages', 2)
  const userId = startId
  const botId = startId + 1

  await messages.insertMany([
    {
      id: userId,
      session_id: sessionId,
      role: 'user',
      content: userMessage.content,
      input_mode: userMessage.inputMode,
      asr_confidence: userMessage.asrConfidence ?? null,
      intent_tag: botMessage.intentTag,
      intent_confidence: botMessage.confidence,
      detected_language: language,
      created_at: now
    },
    {
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
    }
  ])

  return botId
}

export async function getLastBotContextTopic(sessionId) {
  if (!sessionId) return null
  const { messages } = collections()
  const row = await messages.findOne(
    { session_id: sessionId, role: 'bot', source_title: { $type: 'string' } },
    { sort: { id: -1 }, projection: { _id: 0, source_title: 1 } }
  )
  return row?.source_title || null
}

export async function saveMessageFeedback(messageId, feedback) {
  const valid = feedback === 'up' || feedback === 'down' ? feedback : null
  if (!messageId || !valid) return false
  const { messages } = collections()
  const res = await messages.updateOne(
    { id: Number(messageId) },
    { $set: { feedback: valid } }
  )
  return res.matchedCount > 0
}

export async function recordWerEvaluation(referenceText, hypothesisText, metrics, language = 'en-US') {
  const { wer } = collections()
  const id = await nextSequence('wer_evaluations', 1)
  await wer.insertOne({
    id,
    reference_text: referenceText,
    hypothesis_text: hypothesisText,
    wer: metrics.wer,
    cer: metrics.cer,
    word_accuracy: metrics.wordAccuracy,
    substitutions: metrics.substitutions,
    deletions: metrics.deletions,
    insertions: metrics.insertions,
    language,
    created_at: new Date()
  })
}

export async function listWerHistory(limit = 15) {
  const { wer } = collections()
  return wer.find({}, { projection: { _id: 0 } }).sort({ id: -1 }).limit(limit).toArray()
}

export async function clearWerHistory() {
  const { wer } = collections()
  await wer.deleteMany({})
}

export async function saveLearnedArticle(doc) {
  const title = String(doc.title || '').trim().slice(0, 200)
  const lang = String(doc.lang || 'en').split('-')[0].toLowerCase().slice(0, 12)
  const text = String(doc.text || '').trim().slice(0, 6000)
  if (!title || text.length < 40) return

  const aliases = Array.isArray(doc.aliases) ? doc.aliases.slice(0, 12) : []
  const source = String(doc.source || 'Wikipedia').slice(0, 60)
  const url = String(doc.url || '').slice(0, 500)
  const domain = doc.domain ? String(doc.domain).slice(0, 80) : null
  const now = new Date()

  const { learned } = collections()
  const existing = await learned.findOne({ title, lang }, { projection: { _id: 0, text: 1 } })
  const nextText = existing && existing.text && existing.text.length > text.length ? existing.text : text

  await learned.updateOne(
    { title, lang },
    {
      $set: {
        domain,
        text: nextText,
        aliases,
        source,
        url,
        updated_at: now
      },
      $setOnInsert: {
        title,
        lang,
        created_at: now
      }
    },
    { upsert: true }
  )
}

export async function loadLearnedArticlesFromDb() {
  const { learned } = collections()
  const rows = await learned.find({}, { projection: { _id: 0 } }).sort({ created_at: 1 }).limit(5000).toArray()
  return rows.map((r) => ({
    title: r.title,
    lang: r.lang,
    domain: r.domain,
    text: r.text,
    aliases: Array.isArray(r.aliases) ? r.aliases : [],
    source: r.source,
    url: r.url,
    learned: true
  }))
}

export async function getAnalyticsSummary() {
  const { sessions, messages, wer } = collections()

  const [totalSessions, msgSummaryArr, topIntents, topLanguages, werSummaryArr] = await Promise.all([
    sessions.countDocuments(),
    messages
      .aggregate([
        {
          $group: {
            _id: null,
            total_queries: { $sum: { $cond: [{ $eq: ['$role', 'user'] }, 1, 0] } },
            voice_queries: {
              $sum: {
                $cond: [{ $and: [{ $eq: ['$role', 'user'] }, { $eq: ['$input_mode', 'voice'] }] }, 1, 0]
              }
            },
            text_queries: {
              $sum: {
                $cond: [{ $and: [{ $eq: ['$role', 'user'] }, { $eq: ['$input_mode', 'text'] }] }, 1, 0]
              }
            },
            avg_asr_confidence: {
              $avg: { $cond: [{ $ne: ['$asr_confidence', null] }, '$asr_confidence', null] }
            },
            avg_latency_ms: {
              $avg: { $cond: [{ $ne: ['$latency_ms', null] }, '$latency_ms', null] }
            },
            positive: { $sum: { $cond: [{ $eq: ['$feedback', 'up'] }, 1, 0] } },
            negative: { $sum: { $cond: [{ $eq: ['$feedback', 'down'] }, 1, 0] } }
          }
        }
      ])
      .toArray(),
    messages
      .aggregate([
        { $match: { role: 'bot', intent_tag: { $type: 'string' } } },
        { $group: { _id: '$intent_tag', count: { $sum: 1 } } },
        { $sort: { count: -1 } },
        { $limit: 8 },
        { $project: { _id: 0, intent_tag: '$_id', count: 1 } }
      ])
      .toArray(),
    messages
      .aggregate([
        { $match: { role: 'user', detected_language: { $type: 'string' } } },
        { $group: { _id: '$detected_language', count: { $sum: 1 } } },
        { $sort: { count: -1 } },
        { $limit: 8 },
        { $project: { _id: 0, detected_language: '$_id', count: 1 } }
      ])
      .toArray(),
    wer
      .aggregate([
        {
          $group: {
            _id: null,
            total_wer_runs: { $sum: 1 },
            avg_wer: { $avg: '$wer' },
            avg_word_accuracy: { $avg: '$word_accuracy' }
          }
        }
      ])
      .toArray()
  ])

  const m = msgSummaryArr[0] || {}
  const w = werSummaryArr[0] || {}

  return {
    totalSessions,
    total_queries: m.total_queries || 0,
    voice_queries: m.voice_queries || 0,
    text_queries: m.text_queries || 0,
    avg_asr_confidence:
      typeof m.avg_asr_confidence === 'number' ? Number(m.avg_asr_confidence.toFixed(4)) : null,
    avg_latency_ms: typeof m.avg_latency_ms === 'number' ? Number(m.avg_latency_ms.toFixed(2)) : null,
    topIntents,
    topLanguages,
    total_wer_runs: w.total_wer_runs || 0,
    avg_wer: typeof w.avg_wer === 'number' ? Number(w.avg_wer.toFixed(2)) : null,
    avg_word_accuracy: typeof w.avg_word_accuracy === 'number' ? Number(w.avg_word_accuracy.toFixed(2)) : null,
    feedback: {
      positive: m.positive || 0,
      negative: m.negative || 0
    }
  }
}
