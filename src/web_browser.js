/**
 * VoxASR Multi-Website Browser, Discoverer & Readability Scraper
 *
 * Acts as an embedded multi-site browser engine:
 *  - Discovers external websites via DuckDuckGo HTML/Lite + DuckDuckGo Instant Answer
 *  - Validates every target URL & resolved DNS IP against SSRF / private ranges
 *  - Scrapes real external HTML pages with clean readability extraction & boilerplate removal
 *  - Immediately aborts cleanly on CAPTCHA / Cloudflare / anti-bot gates (never bypasses)
 *  - Queries domain-aware public APIs across all fields of knowledge:
 *    Wikipedia, Wikidata, OpenAlex, CrossRef, ArXiv, OpenLibrary, PubChem,
 *    TheSportsDB, iTunes/Media, and StackExchange.
 */

import dns from 'node:dns/promises'
import net from 'node:net'
import { cleanExtract, decodeEntities, splitSentences, stem, tokenize, truncateAtSentence } from './text.js'
import { isBoilerplateSentence } from './query_analyzer.js'

const BROWSER_UA =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'

const BLOCKED_EXT_HOSTS = /(?:web\.archive\.org|doi\.org|dx\.doi\.org|worldcat\.org|isbn|viaf\.org|isni\.org|jstor\.org|semanticscholar\.org|facebook\.com|twitter\.com|x\.com|instagram\.com|youtube\.com|linkedin\.com|tiktok\.com|pinterest\.com|reddit\.com)$/i

const PRIVATE_HOST_PATTERNS = [
  /^localhost$/i,
  /\.local$/i,
  /\.internal$/i,
  /\.arpa$/i,
  /^metadata\.google\.internal$/i
]

const CHALLENGE_REGEX =
  /(?:cf-turnstile|g-recaptcha|hcaptcha|verify\s+you\s+are\s+human|just\s+a\s+moment\.\.\.|attention\s+required!\s*\|\s*cloudflare|access\s+to\s+this\s+page\s+has\s+been\s+denied|please\s+enable\s+cookies\s+and\s+reload)/i

export function isPrivateIpAddress(ip = '') {
  const normalized = String(ip || '').trim().toLowerCase()
  if (!normalized) return true
  const v4 = normalized.startsWith('::ffff:') ? normalized.slice(7) : normalized

  if (net.isIPv4(v4)) {
    const parts = v4.split('.').map(Number)
    if (parts.length !== 4 || parts.some((n) => Number.isNaN(n))) return true
    const [a, b] = parts
    if (a === 0 || a === 10 || a === 127) return true
    if (a === 169 && b === 254) return true
    if (a === 172 && b >= 16 && b <= 31) return true
    if (a === 192 && b === 168) return true
    if (a === 100 && b >= 64 && b <= 127) return true
    if (a >= 224) return true
    return false
  }

  if (net.isIPv6(normalized)) {
    if (
      normalized === '::1' ||
      normalized === '::' ||
      normalized.startsWith('fe80:') ||
      normalized.startsWith('fc') ||
      normalized.startsWith('fd')
    ) {
      return true
    }
    return false
  }

  return true
}

export function isSafePublicUrl(rawUrl = '') {
  try {
    const u = new URL(String(rawUrl || '').trim())
    if (u.protocol !== 'http:' && u.protocol !== 'https:') return { ok: false, reason: 'invalid-scheme' }
    if (u.username || u.password) return { ok: false, reason: 'credentials-disallowed' }
    const host = u.hostname.toLowerCase().replace(/^\[|\]$/g, '')
    if (!host || PRIVATE_HOST_PATTERNS.some((rx) => rx.test(host))) {
      return { ok: false, reason: 'private-hostname' }
    }
    if (net.isIP(host) && isPrivateIpAddress(host)) {
      return { ok: false, reason: 'private-ip-literal' }
    }
    if (!host.includes('.')) return { ok: false, reason: 'non-fqdn-host' }
    return { ok: true, url: u.toString(), host }
  } catch {
    return { ok: false, reason: 'malformed-url' }
  }
}

function withDeadline(promise, ms, fallback = null) {
  let timer
  return Promise.race([
    promise,
    new Promise((resolve) => {
      timer = setTimeout(() => resolve(fallback), ms)
    })
  ]).finally(() => clearTimeout(timer))
}

async function verifyDnsPublic(host) {
  if (net.isIP(host)) return !isPrivateIpAddress(host)
  try {
    const records = await withDeadline(dns.lookup(host, { all: true, verbatim: true }), 1500, null)
    if (!records || records.length === 0) return false
    return records.every((r) => !isPrivateIpAddress(r.address))
  } catch {
    return false
  }
}

function stripTagBlocksLinear(html, lowerHtml, tags) {
  let out = html
  let lower = lowerHtml
  for (const tag of tags) {
    const openNeedle = `<${tag}`
    const closeNeedle = `</${tag}>`
    let cursor = 0
    const parts = []
    while (cursor < out.length) {
      const start = lower.indexOf(openNeedle, cursor)
      if (start === -1) {
        parts.push(out.slice(cursor))
        break
      }
      const nextCh = lower[start + openNeedle.length] || ''
      if (nextCh !== '>' && !/\s/.test(nextCh)) {
        parts.push(out.slice(cursor, start + openNeedle.length))
        cursor = start + openNeedle.length
        continue
      }
      parts.push(out.slice(cursor, start))
      const end = lower.indexOf(closeNeedle, start + openNeedle.length)
      if (end === -1) {
        cursor = out.length
        break
      }
      parts.push(' ')
      cursor = end + closeNeedle.length
    }
    out = parts.join('')
    lower = out.toLowerCase()
  }
  return out
}

function findPrimaryContainerLinear(html) {
  const lower = html.toLowerCase()
  for (const tag of ['article', 'main']) {
    const start = lower.indexOf(`<${tag}`)
    if (start === -1) continue
    const gt = lower.indexOf('>', start)
    if (gt === -1) continue
    const end = lower.indexOf(`</${tag}>`, gt + 1)
    if (end !== -1 && end - gt > 280) {
      return html.slice(gt + 1, end)
    }
  }
  return html
}

function extractParagraphBlocksLinear(html) {
  const lower = html.toLowerCase()
  const blocks = []
  const targetTags = ['p', 'li', 'h2', 'h3', 'blockquote', 'dd']
  let i = 0
  while (i < html.length && blocks.length < 60) {
    const lt = html.indexOf('<', i)
    if (lt === -1) break
    let matchedTag = null
    for (const t of targetTags) {
      if (lower.startsWith(`<${t}`, lt)) {
        const after = lower[lt + 1 + t.length] || ''
        if (after === '>' || after === ' ' || after === '\t' || after === '\n') {
          matchedTag = t
          break
        }
      }
    }
    if (!matchedTag) {
      i = lt + 1
      continue
    }
    const openEnd = html.indexOf('>', lt)
    if (openEnd === -1) break
    const closeTag = `</${matchedTag}>`
    const closeIdx = lower.indexOf(closeTag, openEnd + 1)
    if (closeIdx === -1) {
      i = openEnd + 1
      continue
    }
    const inner = html.slice(openEnd + 1, closeIdx)
    const plain = cleanExtract(decodeEntities(inner.replace(/<[^>]{1,200}>/g, ' ')))
    if (plain.length >= 35 && !isBoilerplateSentence(plain)) {
      blocks.push(plain)
    }
    i = closeIdx + closeTag.length
  }
  return blocks
}

export function extractReadableHtml(html = '', baseUrl = '', titleHint = '', queryTokens = []) {
  const raw = String(html || '').slice(0, 220_000)
  const headSlice = raw.slice(0, 12_000)
  const titleMatch = headSlice.match(/<title[^>]{0,100}>([^<]{1,200})<\/title>/i)
  const ogTitleMatch = headSlice.match(/<meta[^>]+property=["']og:title["'][^>]+content=["']([^"']{1,200})["']/i)
  const metaDescMatch =
    headSlice.match(/<meta[^>]+name=["']description["'][^>]+content=["']([^"']{1,400})["']/i) ||
    headSlice.match(/<meta[^>]+property=["']og:description["'][^>]+content=["']([^"']{1,400})["']/i)

  const pageTitle = decodeEntities((ogTitleMatch?.[1] || titleMatch?.[1] || titleHint || baseUrl).trim())
    .replace(/\s+/g, ' ')
    .slice(0, 140)
  const metaDesc = metaDescMatch ? cleanExtract(decodeEntities(metaDescMatch[1])) : ''

  const stripped = stripTagBlocksLinear(raw, raw.toLowerCase(), [
    'script',
    'style',
    'noscript',
    'svg',
    'iframe',
    'nav',
    'footer',
    'header',
    'aside',
    'form'
  ])

  const scope = findPrimaryContainerLinear(stripped)
  const blockTexts = extractParagraphBlocksLinear(scope)

  if (blockTexts.length === 0) {
    const fallback = cleanExtract(decodeEntities(scope.replace(/<[^>]{1,200}>/g, ' ')))
    for (const s of splitSentences(fallback)) {
      if (!isBoilerplateSentence(s)) blockTexts.push(s)
      if (blockTexts.length >= 20) break
    }
  }

  const qStems = new Set((queryTokens || []).map(stem))
  const scoredBlocks = blockTexts.map((blk, idx) => {
    const bStems = tokenize(blk, 'en', false).map(stem)
    let hits = 0
    for (const bs of bStems) if (qStems.has(bs)) hits += 1
    const posBonus = idx < 4 ? 1.5 - idx * 0.25 : 0
    return { blk, idx, score: hits * 2 + posBonus }
  })

  scoredBlocks.sort((a, b) => b.score - a.score || a.idx - b.idx)
  const picked = scoredBlocks.slice(0, 10).sort((a, b) => a.idx - b.idx).map((b) => b.blk)
  const combined = [metaDesc, ...picked].filter(Boolean).join(' ')
  const extract = truncateAtSentence(cleanExtract(combined), 2400)

  return { title: pageTitle, extract }
}

export async function scrapeExternalWebPage(rawUrl, titleHint = '', queryTokens = [], timeoutMs = 3800) {
  return withDeadline(
    (async () => {
      const safe = isSafePublicUrl(rawUrl)
      if (!safe.ok) return null
      if (/\.(?:pdf|zip|png|jpe?g|gif|webp|mp[34]|wav|exe|dmg)(?:$|\?)/i.test(safe.url)) return null

      const publicDns = await verifyDnsPublic(safe.host)
      if (!publicDns) return null

      try {
        const controller = new AbortController()
        const timer = setTimeout(() => controller.abort(), timeoutMs)
        try {
          const res = await fetch(safe.url, {
            signal: controller.signal,
            redirect: 'follow',
            headers: {
              'User-Agent': BROWSER_UA,
              Accept: 'text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5',
              'Accept-Language': 'en-US,en;q=0.9'
            }
          })
          const finalSafe = isSafePublicUrl(res.url || safe.url)
          if (!finalSafe.ok) return null

          if (res.status === 403 || res.status === 429 || res.status === 503) {
            return { blocked: true, url: finalSafe.url, status: res.status }
          }
          if (!res.ok) return null

          const ct = (res.headers.get('content-type') || '').toLowerCase()
          if (!ct.includes('text/html') && !ct.includes('text/plain') && !ct.includes('xhtml')) {
            return null
          }

          const html = (await withDeadline(res.text(), timeoutMs, '')).slice(0, 220_000)
          if (!html) return null
          if (CHALLENGE_REGEX.test(html.slice(0, 24_000))) {
            return { blocked: true, url: finalSafe.url, status: res.status }
          }

          const extracted = extractReadableHtml(html, finalSafe.url, titleHint, queryTokens)
          if (!extracted.extract || extracted.extract.length < 70) return null

          const hostLabel = finalSafe.host.replace(/^www\./, '')
          return {
            id: `web_${Buffer.from(finalSafe.url).toString('base64url').slice(0, 22)}`,
            title: extracted.title || titleHint || hostLabel,
            source: hostLabel,
            url: finalSafe.url,
            extract: extracted.extract,
            _liveWeb: true
          }
        } finally {
          clearTimeout(timer)
        }
      } catch {
        return null
      }
    })(),
    timeoutMs + 400,
    null
  )
}

export function parseDuckDuckGoHtml(html = '') {
  const capped = String(html || '').slice(0, 150_000)
  const results = []
  const seenUrls = new Set()
  const tagRx = /<a\s+([^>]{0,350}class=["'][^"']{0,80}(?:result__a|result-link)[^"']{0,80}["'][^>]{0,350})>([\s\S]{1,300}?)<\/a>/gi
  let m
  while ((m = tagRx.exec(capped)) !== null) {
    const attrs = m[1] || ''
    const hrefMatch = attrs.match(/\bhref=["']([^"']{1,600})["']/i)
    if (!hrefMatch) continue
    let href = decodeEntities(hrefMatch[1] || '')
    if (href.startsWith('//')) href = `https:${href}`
    try {
      const parsed = new URL(href, 'https://duckduckgo.com')
      const uddg = parsed.searchParams.get('uddg')
      if (uddg) href = uddg
    } catch {
      continue
    }
    const safe = isSafePublicUrl(href)
    if (!safe.ok || seenUrls.has(safe.url)) continue
    seenUrls.add(safe.url)

    const title = cleanExtract(decodeEntities(m[2].replace(/<[^>]{1,200}>/g, ' ')))
    const tail = capped.slice(m.index, m.index + 1400)
    const snipMatch = tail.match(/<(?:a|td)[^>]{0,200}class=["'][^"']{0,80}(?:result__snippet|result-snippet)[^"']{0,80}["'][^>]{0,200}>([\s\S]{1,600}?)<\/(?:a|td)>/i)
    const snippet = snipMatch ? cleanExtract(decodeEntities(snipMatch[1].replace(/<[^>]{1,200}>/g, ' '))) : ''
    if (title) {
      results.push({ title, url: safe.url, host: safe.host.replace(/^www\./, ''), snippet })
    }
    if (results.length >= 12) break
  }
  return results
}

async function searchDuckDuckGoWeb(query) {
  if (!query || query.length < 2) return []
  for (const base of [
    `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`,
    `https://lite.duckduckgo.com/lite/?q=${encodeURIComponent(query)}`
  ]) {
    try {
      const res = await fetch(base, {
        signal: AbortSignal.timeout(4000),
        headers: {
          'User-Agent': BROWSER_UA,
          Accept: 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.8',
          'Accept-Language': 'en-US,en;q=0.9'
        }
      })
      if (res.status !== 200) continue
      const html = await res.text()
      if (CHALLENGE_REGEX.test(html)) continue
      const parsed = parseDuckDuckGoHtml(html)
      if (parsed.length > 0) return parsed
    } catch {}
  }
  return []
}

async function fetchWikipediaExternalLinks(title, lang = 'en') {
  if (!title) return []
  try {
    const params = new URLSearchParams({
      action: 'query',
      prop: 'extlinks',
      ellimit: '18',
      titles: title,
      format: 'json'
    })
    const res = await fetch(`https://${lang}.wikipedia.org/w/api.php?${params}`, {
      signal: AbortSignal.timeout(3000),
      headers: { 'User-Agent': BROWSER_UA }
    })
    if (!res.ok) return []
    const data = await res.json()
    const pages = Object.values(data?.query?.pages || {})
    const links = pages[0]?.extlinks || []
    const candidates = []
    for (const item of links) {
      const raw = item['*'] || item.url || ''
      const safe = isSafePublicUrl(raw)
      if (!safe.ok) continue
      const host = safe.host.replace(/^www\./, '')
      if (BLOCKED_EXT_HOSTS.test(host) || /wikipedia\.org|wikimedia\.org/i.test(host)) continue
      if (/\.(?:pdf|jpg|png|svg|xml)(?:$|\?)/i.test(safe.url)) continue
      candidates.push({ title: `${title} (${host})`, url: safe.url, host, snippet: '' })
    }
    return candidates.slice(0, 4)
  } catch {
    return []
  }
}

async function fetchDuckDuckGoInstantAnswer(query) {
  try {
    const url = `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1&skip_disambig=1`
    const res = await fetch(url, {
      signal: AbortSignal.timeout(3000),
      headers: { 'User-Agent': BROWSER_UA }
    })
    if (!res.ok) return []
    const data = await res.json()
    const docs = []
    if (data.AbstractText && data.AbstractText.length > 50 && data.AbstractURL) {
      docs.push({
        id: `ddg_ia_${Buffer.from(data.AbstractURL).toString('base64url').slice(0, 16)}`,
        title: data.Heading || query,
        source: data.AbstractSource || 'DuckDuckGo Answer',
        url: data.AbstractURL,
        extract: cleanExtract(data.AbstractText),
        _liveWeb: true
      })
    }
    const related = Array.isArray(data.RelatedTopics) ? data.RelatedTopics : []
    for (const r of related.slice(0, 2)) {
      if (r?.Text && r?.FirstURL && r.Text.length > 45) {
        docs.push({
          id: `ddg_rel_${Buffer.from(r.FirstURL).toString('base64url').slice(0, 16)}`,
          title: r.Text.split(' - ')[0]?.slice(0, 90) || query,
          source: 'DuckDuckGo Reference',
          url: r.FirstURL,
          extract: cleanExtract(r.Text),
          _liveWeb: true
        })
      }
    }
    return docs
  } catch {
    return []
  }
}

export async function fetchWikipediaExtract(title, lang = 'en') {
  const api = `https://${lang}.wikipedia.org/w/api.php`
  const params = new URLSearchParams({
    action: 'query',
    prop: 'extracts',
    exintro: '1',
    explaintext: '1',
    exchars: '2400',
    redirects: '1',
    titles: title,
    format: 'json'
  })
  const res = await fetch(`${api}?${params}`, {
    signal: AbortSignal.timeout(3500),
    headers: { 'User-Agent': BROWSER_UA }
  })
  if (!res.ok) return null
  const data = await res.json()
  const pages = Object.values(data?.query?.pages || {})
  const page = pages[0]
  if (!page || page.missing || !page.extract || page.extract.length < 70) return null
  const slug = encodeURIComponent(page.title.replace(/ /g, '_'))
  return {
    id: `wiki_${lang}_${page.pageid}`,
    title: page.title,
    source: 'Wikipedia',
    url: `https://${lang}.wikipedia.org/wiki/${slug}`,
    extract: truncateAtSentence(cleanExtract(page.extract), 2300),
    _liveWeb: true
  }
}

export async function searchWikipediaTitles(query, lang = 'en', limit = 2) {
  const api = `https://${lang}.wikipedia.org/w/api.php`
  const params = new URLSearchParams({
    action: 'query',
    list: 'search',
    srsearch: query,
    srlimit: String(limit),
    format: 'json'
  })
  const res = await fetch(`${api}?${params}`, {
    signal: AbortSignal.timeout(3200),
    headers: { 'User-Agent': BROWSER_UA }
  })
  if (!res.ok) return []
  const data = await res.json()
  return (data?.query?.search || []).map((s) => s.title).filter(Boolean)
}

async function fetchWikidataEntities(query, lang = 'en') {
  try {
    const params = new URLSearchParams({
      action: 'wbsearchentities',
      search: query,
      language: lang,
      limit: '2',
      format: 'json'
    })
    const res = await fetch(`https://www.wikidata.org/w/api.php?${params}`, {
      signal: AbortSignal.timeout(2800),
      headers: { 'User-Agent': BROWSER_UA }
    })
    if (!res.ok) return []
    const data = await res.json()
    return (data?.search || [])
      .filter((item) => item.label && item.description && item.concepturi)
      .map((item) => ({
        id: `wd_${item.id}`,
        title: item.label,
        source: 'Wikidata',
        url: `https://www.wikidata.org/wiki/${item.id}`,
        extract: `${item.label}: ${item.description}.`,
        _liveWeb: true
      }))
  } catch {
    return []
  }
}

function reconstructAbstract(invertedIndex) {
  if (!invertedIndex || typeof invertedIndex !== 'object') return ''
  const words = []
  for (const [token, positions] of Object.entries(invertedIndex)) {
    for (const pos of positions) words[pos] = token
  }
  return cleanExtract(words.filter(Boolean).join(' '))
}

async function fetchOpenAlexWorks(query, maxResults = 2) {
  try {
    const url = `https://api.openalex.org/works?search=${encodeURIComponent(query)}&per-page=${maxResults}&select=id,title,publication_year,abstract_inverted_index,primary_location`
    const res = await fetch(url, {
      signal: AbortSignal.timeout(3500),
      headers: { 'User-Agent': BROWSER_UA }
    })
    if (!res.ok) return []
    const data = await res.json()
    const docs = []
    for (const w of data?.results || []) {
      const abs = reconstructAbstract(w.abstract_inverted_index)
      if (!w.title || !abs || abs.length < 60) continue
      const landing = w.primary_location?.landing_page_url || w.id
      docs.push({
        id: `openalex_${String(w.id).split('/').pop()}`,
        title: `${w.title}${w.publication_year ? ` (${w.publication_year})` : ''}`,
        source: 'OpenAlex Research',
        url: landing,
        extract: truncateAtSentence(abs, 1600),
        _liveWeb: true
      })
    }
    return docs
  } catch {
    return []
  }
}

async function fetchCrossRefWorks(query, maxResults = 2) {
  try {
    const url = `https://api.crossref.org/works?query=${encodeURIComponent(query)}&rows=${maxResults}&select=DOI,title,abstract,URL`
    const res = await fetch(url, {
      signal: AbortSignal.timeout(3500),
      headers: { 'User-Agent': BROWSER_UA }
    })
    if (!res.ok) return []
    const data = await res.json()
    const docs = []
    for (const item of data?.message?.items || []) {
      const title = Array.isArray(item.title) ? item.title[0] : item.title
      const abs = item.abstract ? cleanExtract(decodeEntities(String(item.abstract).replace(/<[^>]+>/g, ' '))) : ''
      if (!title || abs.length < 60) continue
      docs.push({
        id: `crossref_${Buffer.from(String(item.DOI || title)).toString('base64url').slice(0, 16)}`,
        title: cleanExtract(title),
        source: 'CrossRef Scholarly',
        url: item.URL || `https://doi.org/${encodeURIComponent(item.DOI || '')}`,
        extract: truncateAtSentence(abs, 1400),
        _liveWeb: true
      })
    }
    return docs
  } catch {
    return []
  }
}

export async function fetchArxivArticles(query, domain = null, maxResults = 2) {
  try {
    const url = `https://export.arxiv.org/api/query?search_query=all:${encodeURIComponent(query)}&start=0&max_results=${maxResults}`
    const res = await fetch(url, {
      signal: AbortSignal.timeout(3600),
      headers: { 'User-Agent': BROWSER_UA }
    })
    if (!res.ok) return []
    const xml = await res.text()
    const entries = xml.match(/<entry>[\s\S]*?<\/entry>/g) || []
    const out = []
    for (const entry of entries) {
      const title = (entry.match(/<title>([\s\S]*?)<\/title>/)?.[1] || '').replace(/\s+/g, ' ').trim()
      const summary = (entry.match(/<summary>([\s\S]*?)<\/summary>/)?.[1] || '').replace(/\s+/g, ' ').trim()
      const idUrl = (entry.match(/<id>([\s\S]*?)<\/id>/)?.[1] || '').trim()
      if (!title || summary.length < 50) continue
      out.push({
        id: `arxiv_${idUrl.split('/').pop() || Math.random().toString(36).slice(2, 8)}`,
        title: decodeEntities(title),
        source: 'ArXiv',
        url: idUrl || `https://arxiv.org/search/?query=${encodeURIComponent(query)}`,
        domain: domain || 'machine_learning_dl',
        extract: truncateAtSentence(cleanExtract(decodeEntities(summary)), 1500),
        _liveWeb: true
      })
    }
    return out
  } catch {
    return []
  }
}

async function fetchPubChemCompound(query) {
  try {
    const clean = query.replace(/\b(what is|molecule|compound|chemical|structure|formula|properties)\b/gi, '').trim()
    if (!clean || clean.length > 50) return []
    const url = `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${encodeURIComponent(clean)}/property/Title,MolecularFormula,MolecularWeight,IUPACName/JSON`
    const res = await fetch(url, { signal: AbortSignal.timeout(2800) })
    if (!res.ok) return []
    const data = await res.json()
    const prop = data?.PropertyTable?.Properties?.[0]
    if (!prop?.CID) return []
    return [
      {
        id: `pubchem_${prop.CID}`,
        title: prop.Title || clean,
        source: 'PubChem (NIH)',
        url: `https://pubchem.ncbi.nlm.nih.gov/compound/${prop.CID}`,
        extract: `${prop.Title || clean} (IUPAC: ${prop.IUPACName || 'N/A'}) has molecular formula ${prop.MolecularFormula || 'N/A'} and molecular weight ${prop.MolecularWeight || 'N/A'} g/mol.`,
        _liveWeb: true
      }
    ]
  } catch {
    return []
  }
}

async function fetchOpenLibraryBooks(query) {
  try {
    const url = `https://openlibrary.org/search.json?q=${encodeURIComponent(query)}&limit=2`
    const res = await fetch(url, { signal: AbortSignal.timeout(2800) })
    if (!res.ok) return []
    const data = await res.json()
    return (data?.docs || [])
      .filter((b) => b.title && (b.author_name || b.first_publish_year))
      .slice(0, 2)
      .map((b) => ({
        id: `ol_${String(b.key || b.title).replace(/\W+/g, '_')}`,
        title: b.title,
        source: 'OpenLibrary',
        url: b.key ? `https://openlibrary.org${b.key}` : `https://openlibrary.org/search?q=${encodeURIComponent(query)}`,
        extract: `"${b.title}"${b.author_name ? ` by ${b.author_name.slice(0, 2).join(', ')}` : ''}${b.first_publish_year ? ` (first published ${b.first_publish_year})` : ''}${b.subject ? `. Key themes: ${b.subject.slice(0, 5).join(', ')}.` : '.'}`,
        _liveWeb: true
      }))
  } catch {
    return []
  }
}

async function fetchSportsInfo(query) {
  try {
    const clean = query.replace(/\b(who is|what is|team|club|player|sport|cricket|football|soccer|basketball|tennis)\b/gi, '').trim()
    if (!clean) return []
    const url = `https://www.thesportsdb.com/api/v1/json/3/searchplayers.php?p=${encodeURIComponent(clean)}`
    const res = await fetch(url, { signal: AbortSignal.timeout(2800) })
    if (!res.ok) return []
    const data = await res.json()
    const p = data?.player?.[0]
    if (!p?.strPlayer) return []
    const desc = cleanExtract(p.strDescriptionEN || '')
    return [
      {
        id: `sports_${p.idPlayer}`,
        title: `${p.strPlayer} (${p.strSport || 'Sports'})`,
        source: 'TheSportsDB',
        url: `https://www.thesportsdb.com/player/${p.idPlayer}`,
        extract: desc ? truncateAtSentence(desc, 1200) : `${p.strPlayer} is a ${p.strSport || ''} athlete (${p.strNationality || ''}, Team: ${p.strTeam || 'N/A'}).`,
        _liveWeb: true
      }
    ]
  } catch {
    return []
  }
}

async function fetchMusicOrMediaInfo(query) {
  try {
    const url = `https://itunes.apple.com/search?term=${encodeURIComponent(query)}&limit=2`
    const res = await fetch(url, { signal: AbortSignal.timeout(2800) })
    if (!res.ok) return []
    const data = await res.json()
    return (data?.results || [])
      .filter((r) => r.artistName || r.trackName || r.collectionName)
      .slice(0, 2)
      .map((r, i) => ({
        id: `itunes_${r.trackId || r.collectionId || i}`,
        title: r.trackName || r.collectionName || r.artistName,
        source: 'Music & Media Index',
        url: r.trackViewUrl || r.collectionViewUrl || r.artistViewUrl || 'https://music.apple.com',
        extract: `${r.trackName || r.collectionName || r.artistName} by ${r.artistName || 'Unknown'} (${r.primaryGenreName || r.kind || 'Media'}${r.releaseDate ? `, released ${r.releaseDate.slice(0, 10)}` : ''})${r.longDescription ? `: ${truncateAtSentence(cleanExtract(r.longDescription), 500)}` : '.'}`,
        _liveWeb: true
      }))
  } catch {
    return []
  }
}

export async function browseWebForQuery(query, analysis = {}, domainHint = null, options = {}) {
  return withDeadline(
    browseWebForQueryInner(query, analysis, domainHint, options),
    options.timeoutMs || 9500,
    { docs: [], scrapedHosts: [], blockedUrls: [] }
  )
}

async function browseWebForQueryInner(query, analysis = {}, domainHint = null, options = {}) {
  const cleanQuery = String(query || '').trim()
  if (!cleanQuery && (!analysis.directUrls || analysis.directUrls.length === 0)) {
    return { docs: [], scrapedHosts: [], blockedUrls: [] }
  }

  const lang = options.lang || analysis.lang || 'en'
  const depthMode = analysis.depthMode || options.depthMode || 'standard'
  const qTokens = tokenize(cleanQuery, lang, false)
  const scrapeLimit = depthMode === 'detailed' ? 4 : depthMode === 'brief' ? 2 : 3
  const blockedUrls = []
  const docs = []

  // 1. Direct URL browsing when user pasted URL(s)
  if (Array.isArray(analysis.directUrls) && analysis.directUrls.length > 0) {
    const directPages = await Promise.all(
      analysis.directUrls.slice(0, 3).map((u) => scrapeExternalWebPage(u, '', qTokens, 5000))
    )
    for (const p of directPages) {
      if (!p) continue
      if (p.blocked) blockedUrls.push(p.url)
      else docs.push(p)
    }
    if (docs.length > 0 && cleanQuery.split(/\s+/).length <= 3) {
      return {
        docs,
        scrapedHosts: [...new Set(docs.map((d) => d.source))],
        blockedUrls
      }
    }
  }

  // 2. Determine specialized vertical APIs relevant to the domain/query
  const lower = cleanQuery.toLowerCase()
  const domain = String(domainHint || '').toLowerCase()
  const verticalTasks = []

  if (domain === 'chemistry_biology_medicine' || /\b(acid|molecule|chemical|compound|enzyme|vitamin|protein|drug)\b/.test(lower)) {
    verticalTasks.push(fetchPubChemCompound(cleanQuery))
  }
  if (domain === 'arts_literature_design' || /\b(novel|book|poem|author|painting|renaissance|sculpture|literature)\b/.test(lower)) {
    verticalTasks.push(fetchOpenLibraryBooks(cleanQuery))
  }
  if (domain === 'sports_athletics_games' || /\b(cricket|football|soccer|tennis|basketball|chess|olympic|badminton|athlete|player|world cup)\b/.test(lower)) {
    verticalTasks.push(fetchSportsInfo(cleanQuery))
  }
  if (domain === 'music_audio_theory' || domain === 'entertainment_cinema_gaming' || /\b(song|album|symphony|jazz|raga|movie|film|director|soundtrack|singer|band|opera)\b/.test(lower)) {
    verticalTasks.push(fetchMusicOrMediaInfo(cleanQuery))
  }
  if (
    depthMode === 'detailed' ||
    domain === 'machine_learning_dl' ||
    domain === 'physics_astronomy_earth' ||
    domain === 'chemistry_biology_medicine' ||
    domain === 'commerce_finance_economics' ||
    domain === 'engineering_robotics_energy' ||
    /\b(study|research|paper|clinical|trial|mechanism|theory|economic|equation)\b/.test(lower)
  ) {
    verticalTasks.push(fetchOpenAlexWorks(cleanQuery, depthMode === 'detailed' ? 2 : 1))
  }
  if (
    domain === 'machine_learning_dl' ||
    domain === 'speech_ai_asr' ||
    domain === 'physics_astronomy_earth' ||
    domain === 'numerical_analysis'
  ) {
    verticalTasks.push(fetchArxivArticles(cleanQuery, domainHint, 2))
  }
  if (depthMode === 'detailed') {
    verticalTasks.push(fetchCrossRefWorks(cleanQuery, 1))
  }

  // 3. Fire broad multi-web discovery in parallel with per-source deadlines
  const [ddgResults, ddgIaDocs, wikiTitles, wikidataDocs, ...verticalResults] = await Promise.all([
    withDeadline(searchDuckDuckGoWeb(cleanQuery), 4200, []),
    withDeadline(fetchDuckDuckGoInstantAnswer(cleanQuery), 3200, []),
    withDeadline(searchWikipediaTitles(cleanQuery, lang, 2), 3400, []),
    withDeadline(fetchWikidataEntities(cleanQuery, lang), 3000, []),
    ...verticalTasks.map((p) => withDeadline(p, 3800, []))
  ])

  // Select diverse non-Wikipedia URLs from DuckDuckGo web search (plus Wikipedia external links backup)
  let candidatePool = ddgResults.filter((r) => !/wikipedia\.org/i.test(r.host))
  if (candidatePool.length < scrapeLimit && wikiTitles[0]) {
    const extLinks = await withDeadline(fetchWikipediaExternalLinks(wikiTitles[0], lang).catch(() => []), 2500, [])
    candidatePool = [...candidatePool, ...extLinks]
  }

  const hostCounts = new Map()
  const targetsToScrape = []
  for (const r of candidatePool) {
    if (/wikipedia\.org/i.test(r.host) || BLOCKED_EXT_HOSTS.test(r.host)) continue
    const cnt = hostCounts.get(r.host) || 0
    if (cnt >= 1) continue // max 1 page per hostname to maximize independent site diversity
    hostCounts.set(r.host, cnt + 1)
    targetsToScrape.push(r)
    if (targetsToScrape.length >= scrapeLimit) break
  }

  const wikiFetchPromise = withDeadline(
    Promise.all(wikiTitles.slice(0, 2).map((t) => fetchWikipediaExtract(t, lang).catch(() => null))),
    3600,
    []
  )
  const externalScrapePromise = Promise.all(
    targetsToScrape.map((r) => scrapeExternalWebPage(r.url, r.title, qTokens, 3600))
  )

  const [wikiDocs, scrapedPages] = await Promise.all([wikiFetchPromise, externalScrapePromise])

  for (let i = 0; i < scrapedPages.length; i++) {
    const page = scrapedPages[i]
    const meta = targetsToScrape[i]
    if (page && page.blocked) {
      blockedUrls.push(page.url)
      if (meta?.snippet && meta.snippet.length > 45) {
        docs.push({
          id: `ddg_snip_${Buffer.from(meta.url).toString('base64url').slice(0, 16)}`,
          title: meta.title,
          source: meta.host,
          url: meta.url,
          extract: meta.snippet,
          _liveWeb: true
        })
      }
      continue
    }
    if (page && page.extract) {
      docs.push(page)
    } else if (meta?.snippet && meta.snippet.length > 45) {
      docs.push({
        id: `ddg_snip_${Buffer.from(meta.url).toString('base64url').slice(0, 16)}`,
        title: meta.title,
        source: meta.host,
        url: meta.url,
        extract: meta.snippet,
        _liveWeb: true
      })
    }
  }

  for (const d of ddgIaDocs) docs.push(d)
  for (const arr of verticalResults) {
    if (Array.isArray(arr)) docs.push(...arr)
  }
  for (const w of wikiDocs) if (w) docs.push(w)
  for (const wd of wikidataDocs) docs.push(wd)

  // Deduplicate by URL and give domain tag
  const seenUrls = new Set()
  const uniqueDocs = []
  for (const d of docs) {
    if (!d || !d.extract || d.extract.length < 40) continue
    const key = d.url || `${d.source}::${d.title}`
    if (seenUrls.has(key)) continue
    seenUrls.add(key)
    uniqueDocs.push({
      ...d,
      domain: d.domain || domainHint || 'general_world_knowledge',
      lang
    })
  }

  return {
    docs: uniqueDocs,
    scrapedHosts: [...new Set(uniqueDocs.map((d) => d.source).filter(Boolean))],
    blockedUrls
  }
}
