/**
 * Live Knowledge Pool Loader, Hot-Reloader & Self-Learning Web Scraper (Pure Node.js)
 * ----------------------------------------------------------------------------------
 * - Loads sharded KB (`public/static/kb/manifest.json` + `kb-*.json`) into `KnowledgeIndex`
 * - Also ingests `training/dynamic_knowledge_pool.json` & `training/learned_knowledge.json`
 * - Hydrates previously learned articles from MongoDB on startup
 * - Scrapes Wikipedia MediaWiki API & ArXiv Atom API on-demand or during auto-upgrade
 * - Hot-reloads `chatbot_model.json` & KB shards automatically when `manifest.json` updates
 */

import fs from 'node:fs'
import path from 'node:path'
import { KnowledgeIndex } from './knowledge.js'
import { cleanExtract, decodeEntities, truncateAtSentence } from './text.js'
import { loadLearnedArticlesFromDb, saveLearnedArticle } from './db.js'

const USER_AGENT = 'VoxASR-KnowledgeBot/3.0 (commercial multilingual CS/Math AI assistant)'

const ROOT_DIR = path.resolve(process.cwd())
const KB_DIR = path.join(ROOT_DIR, 'public', 'static', 'kb')
const MANIFEST_PATH = path.join(KB_DIR, 'manifest.json')
const DYNAMIC_POOL_PATH = path.join(ROOT_DIR, 'training', 'dynamic_knowledge_pool.json')
const LEARNED_JSON_PATH = path.join(ROOT_DIR, 'training', 'learned_knowledge.json')
const MODEL_PATH = path.join(ROOT_DIR, 'src', 'model', 'chatbot_model.json')

const index = new KnowledgeIndex()
let loadedManifestVersion = -1
let lastUpgradeStatus = {
  is_upgrading: false,
  last_progress: 'Idle (Native Node.js + Live Web Learning Active)',
  last_report: null
}

const ARXIV_CATEGORIES = [
  { domain: 'algorithms_complexity', cat: 'cs.DS' },
  { domain: 'machine_learning_dl', cat: 'cs.LG' },
  { domain: 'nlp_large_language_models', cat: 'cs.CL' },
  { domain: 'numerical_analysis', cat: 'math.NA' },
  { domain: 'cryptography_security', cat: 'cs.CR' },
  { domain: 'speech_ai_asr', cat: 'eess.AS' }
]

export function getKnowledgeIndex() {
  return index
}

export function getManifestMeta() {
  try {
    if (fs.existsSync(MANIFEST_PATH)) {
      return JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf8'))
    }
  } catch (_) {}
  return { version: 0, total_articles: index.size, domains: {}, sources: {} }
}

export function getUpgradeState() {
  return { ...lastUpgradeStatus }
}

function readJsonSafe(filePath) {
  try {
    if (fs.existsSync(filePath)) {
      return JSON.parse(fs.readFileSync(filePath, 'utf8'))
    }
  } catch (_) {}
  return null
}

function appendLearnedJsonFile(doc) {
  try {
    const list = readJsonSafe(LEARNED_JSON_PATH)
    const arr = Array.isArray(list) ? list.filter((d) => d.title !== doc.title) : []
    arr.push({
      title: String(doc.title).slice(0, 200),
      text: String(doc.text).slice(0, 6000),
      url: String(doc.url || '').slice(0, 500),
      source: String(doc.source || 'Web').slice(0, 60),
      lang: String(doc.lang || 'en').slice(0, 12),
      domain: doc.domain || null,
      aliases: Array.isArray(doc.aliases) ? doc.aliases.slice(0, 12) : [],
      learned_at: new Date().toISOString()
    })
    fs.mkdirSync(path.dirname(LEARNED_JSON_PATH), { recursive: true })
    const tmp = `${LEARNED_JSON_PATH}.tmp`
    fs.writeFileSync(tmp, JSON.stringify(arr.slice(-5000), null, 1), 'utf8')
    fs.renameSync(tmp, LEARNED_JSON_PATH)
  } catch (_) {}
}

export async function initKnowledgeEngine(reloadModelFn) {
  const manifest = getManifestMeta()
  if (manifest && Array.isArray(manifest.shards)) {
    for (const shard of manifest.shards) {
      const shardPath = path.join(KB_DIR, shard.file)
      const records = readJsonSafe(shardPath)
      if (Array.isArray(records)) {
        for (const rec of records) index.add(rec)
      }
    }
    loadedManifestVersion = Number(manifest.version || 0)
    index.version = loadedManifestVersion
  }

  // Also load ArXiv/seed entries from dynamic_knowledge_pool.json
  const dynPool = readJsonSafe(DYNAMIC_POOL_PATH)
  if (dynPool && dynPool.domains) {
    for (const [domainKey, dom] of Object.entries(dynPool.domains)) {
      for (const art of dom.articles || []) {
        if (!art.title || !art.summary) continue
        index.add({
          title: art.title,
          text: art.summary,
          domain: domainKey,
          source: art.source || 'Wikipedia',
          url: art.url || '',
          lang: 'en',
          learned: false
        })
      }
    }
  }

  // Load learned_knowledge.json
  const learnedFile = readJsonSafe(LEARNED_JSON_PATH)
  if (Array.isArray(learnedFile)) {
    for (const doc of learnedFile) {
      index.add({ ...doc, learned: true })
    }
  }

  // Load learned articles persisted in MongoDB
  try {
    const dbDocs = await loadLearnedArticlesFromDb()
    for (const doc of dbDocs) {
      index.add({ ...doc, learned: true })
    }
  } catch (err) {
    console.warn('[KB] Could not hydrate learned articles from DB:', err.message)
  }

  // Hot-load published model if newer
  if (typeof reloadModelFn === 'function') {
    const kbModel = readJsonSafe(path.join(KB_DIR, 'chatbot_model.json')) || readJsonSafe(MODEL_PATH)
    if (kbModel) reloadModelFn(kbModel)
  }

  console.log(`[KB] KnowledgeIndex ready: ${index.size} articles (${index.learnedCount} learned), ${index.postings.size} terms`)
  return index.stats()
}

export async function checkHotReload(reloadModelFn) {
  const manifest = getManifestMeta()
  const ver = Number(manifest.version || 0)
  if (ver > loadedManifestVersion && Array.isArray(manifest.shards) && manifest.shards.length > 0) {
    for (const shard of manifest.shards) {
      const records = readJsonSafe(path.join(KB_DIR, shard.file))
      if (Array.isArray(records)) {
        for (const rec of records) index.add(rec)
      }
    }
    loadedManifestVersion = ver
    index.version = ver
    if (typeof reloadModelFn === 'function') {
      const freshModel = readJsonSafe(path.join(KB_DIR, 'chatbot_model.json')) || readJsonSafe(MODEL_PATH)
      if (freshModel) reloadModelFn(freshModel)
    }
    return true
  }
  return false
}

async function fetchWikipediaExtract(topic, lang = 'en') {
  const baseUrl = `https://${lang}.wikipedia.org/w/api.php`
  const params = new URLSearchParams({
    action: 'query',
    format: 'json',
    formatversion: '2',
    prop: 'extracts|pageprops|redirects',
    exintro: '1',
    explaintext: '1',
    ppprop: 'disambiguation',
    rdlimit: '20',
    rdnamespace: '0',
    redirects: '1',
    titles: topic
  })

  const resp = await fetch(`${baseUrl}?${params.toString()}`, {
    headers: { 'User-Agent': USER_AGENT, Accept: 'application/json' },
    signal: AbortSignal.timeout(6500)
  })
  if (!resp.ok) return null
  const data = await resp.json()
  const pages = data?.query?.pages || []
  const redirects = (data?.query?.redirects || []).map((r) => r.from).filter(Boolean)

  for (const page of pages) {
    if (page.missing || page.pageprops?.disambiguation !== undefined) continue
    const text = truncateAtSentence(cleanExtract(page.extract || ''), 1400)
    if (text.length < 60) continue
    const aliases = [
      ...new Set([
        ...redirects,
        ...(page.redirects || []).map((r) => r.title).filter(Boolean)
      ])
    ].slice(0, 8)
    return {
      title: page.title,
      text,
      lang,
      source: 'Wikipedia (Live Web)',
      url: `https://${lang}.wikipedia.org/wiki/${encodeURIComponent(page.title.replace(/ /g, '_'))}`,
      aliases,
      learned: true
    }
  }
  return null
}

async function searchWikipediaTitles(query, lang = 'en', limit = 3) {
  const baseUrl = `https://${lang}.wikipedia.org/w/api.php`
  const params = new URLSearchParams({
    action: 'query',
    format: 'json',
    formatversion: '2',
    list: 'search',
    srsearch: query,
    srlimit: String(limit)
  })
  const resp = await fetch(`${baseUrl}?${params.toString()}`, {
    headers: { 'User-Agent': USER_AGENT, Accept: 'application/json' },
    signal: AbortSignal.timeout(6500)
  })
  if (!resp.ok) return []
  const data = await resp.json()
  return (data?.query?.search || []).map((h) => h.title).filter(Boolean)
}

export async function fetchArxivArticles(query, domain = null, maxResults = 3) {
  const url = `https://export.arxiv.org/api/query?search_query=${encodeURIComponent(query)}&start=0&max_results=${maxResults}&sortBy=submittedDate&sortOrder=descending`
  const resp = await fetch(url, {
    headers: { 'User-Agent': USER_AGENT, Accept: 'application/atom+xml' },
    signal: AbortSignal.timeout(7000)
  })
  if (!resp.ok) return []
  const xml = await resp.text()
  const entries = []
  const entryRegex = /<entry>([\s\S]*?)<\/entry>/g
  let m
  while ((m = entryRegex.exec(xml)) !== null) {
    const block = m[1]
    const titleMatch = block.match(/<title>([\s\S]*?)<\/title>/)
    const sumMatch = block.match(/<summary>([\s\S]*?)<\/summary>/)
    const idMatch = block.match(/<id>([\s\S]*?)<\/id>/)
    if (!titleMatch || !sumMatch) continue
    const title = decodeEntities(titleMatch[1]).replace(/\s+/g, ' ').trim()
    const summary = truncateAtSentence(decodeEntities(sumMatch[1]).replace(/\s+/g, ' ').trim(), 1200)
    const link = idMatch ? idMatch[1].trim() : ''
    if (title && summary.length >= 60) {
      entries.push({
        title,
        text: summary,
        lang: 'en',
        domain,
        source: 'ArXiv (Live Preprint)',
        url: link,
        aliases: [],
        learned: true
      })
    }
  }
  return entries
}

async function persistLearnedDoc(doc) {
  index.add(doc)
  appendLearnedJsonFile(doc)
  try {
    await saveLearnedArticle(doc)
  } catch (err) {
    console.warn('[KB] DB persist learned article failed:', err.message)
  }
}

export async function learnTopicFromWeb(topic, lang = 'en', domain = null) {
  const cleanTopic = String(topic || '').trim()
  if (!cleanTopic || cleanTopic.length < 2) return null

  const shortLang = String(lang || 'en').split('-')[0].toLowerCase()
  const tryLangs = shortLang !== 'en' ? [shortLang, 'en'] : ['en']

  for (const l of tryLangs) {
    try {
      let doc = await fetchWikipediaExtract(cleanTopic, l)
      if (!doc) {
        const hits = await searchWikipediaTitles(cleanTopic, l, 3)
        for (const hit of hits) {
          doc = await fetchWikipediaExtract(hit, l)
          if (doc) break
        }
      }
      if (doc) {
        if (domain) doc.domain = domain
        if (cleanTopic.toLowerCase() !== doc.title.toLowerCase()) {
          doc.aliases = [...new Set([...(doc.aliases || []), cleanTopic])].slice(0, 10)
        }
        await persistLearnedDoc(doc)
        return doc
      }
    } catch (_) {}
  }

  // Fallback to ArXiv search
  try {
    const arxivDocs = await fetchArxivArticles(`all:${cleanTopic}`, domain, 2)
    if (arxivDocs.length > 0) {
      for (const doc of arxivDocs) await persistLearnedDoc(doc)
      return arxivDocs[0]
    }
  } catch (_) {}

  return null
}

export async function runNativeKnowledgeUpgrade() {
  if (lastUpgradeStatus.is_upgrading) {
    return { ok: false, error: 'An upgrade job is already running.' }
  }

  lastUpgradeStatus = {
    is_upgrading: true,
    last_progress: 'Harvesting latest ArXiv CS/Math preprints & Wikipedia concepts...',
    last_report: null
  }

  const startedAt = Date.now()
  let addedCount = 0

  ;(async () => {
    try {
      for (let i = 0; i < ARXIV_CATEGORIES.length; i++) {
        const { domain, cat } = ARXIV_CATEGORIES[i]
        lastUpgradeStatus.last_progress = `Scraping ArXiv ${cat} (${i + 1}/${ARXIV_CATEGORIES.length})...`
        try {
          const papers = await fetchArxivArticles(`cat:${cat}`, domain, 3)
          for (const p of papers) {
            const before = index.size
            await persistLearnedDoc(p)
            if (index.size > before) addedCount++
          }
        } catch (_) {}
      }

      const bonusTopics = [
        { topic: 'FlashAttention', domain: 'nlp_large_language_models' },
        { topic: 'Rotary Position Embedding', domain: 'machine_learning_dl' },
        { topic: 'Low-rank adaptation', domain: 'nlp_large_language_models' },
        { topic: 'ML-KEM', domain: 'cryptography_security' },
        { topic: 'HNSW', domain: 'database_vector_search' },
        { topic: 'Connectionist temporal classification', domain: 'speech_ai_asr' }
      ]
      lastUpgradeStatus.last_progress = 'Checking Wikipedia for emerging CS/Math topics...'
      for (const item of bonusTopics) {
        const before = index.size
        await learnTopicFromWeb(item.topic, 'en', item.domain).catch(() => null)
        if (index.size > before) addedCount++
      }

      const duration = Number(((Date.now() - startedAt) / 1000).toFixed(2))
      lastUpgradeStatus = {
        is_upgrading: false,
        last_progress: `Upgrade complete (+${addedCount} new articles in ${duration}s)`,
        last_report: {
          ok: true,
          added_articles: addedCount,
          total_articles: index.size,
          learned_articles: index.learnedCount,
          duration_seconds: duration,
          completed_at: new Date().toISOString()
        }
      }
    } catch (err) {
      lastUpgradeStatus = {
        is_upgrading: false,
        last_progress: `Failed: ${err.message}`,
        last_report: { ok: false, error: err.message }
      }
    }
  })()

  return {
    ok: true,
    message: 'Live ArXiv & Wikipedia knowledge harvest initiated in background.',
    status: 'started'
  }
}
