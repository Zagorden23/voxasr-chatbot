/**
 * Knowledge engine: an in-memory BM25 search index over the knowledge pool,
 * question analysis, subject resolution and extractive answer selection.
 *
 * The index holds the harvested pool (public/static/kb shards) plus every
 * article the bot learned by itself from the web; both are searched the same way.
 */

import { tokenize, FILLER, looseMatch, splitSentences, stem } from './text.js'
import { parseQueryDepthAndKind } from './query_analyzer.js'

const K1 = 1.4
const B = 0.6
const TITLE_BOOST = 3
const ALIAS_BOOST = 2

export function wikiUrl(title, lang = 'en') {
  return `https://${lang}.wikipedia.org/wiki/${encodeURIComponent(String(title).replace(/ /g, '_'))}`
}

/** Accepts the compact shard format ({t,x,d,a,u,s,l}) or a verbose learned record. */
export function normalizeDoc(raw) {
  if (!raw) return null
  const title = String(raw.t || raw.title || '').trim()
  const text = String(raw.x || raw.text || '').trim()
  if (!title || !text) return null
  const lang = String(raw.l || raw.lang || 'en').split('-')[0].toLowerCase()
  const source = raw.s || raw.source || 'Wikipedia'
  return {
    title,
    text,
    lang,
    source,
    url: raw.u || raw.url || (String(source).startsWith('Wikipedia') ? wikiUrl(title, lang) : ''),
    domain: raw.d || raw.domain || null,
    aliases: Array.isArray(raw.a || raw.aliases) ? (raw.a || raw.aliases).map(String) : [],
    learned: Boolean(raw.learned)
  }
}

export class KnowledgeIndex {
  constructor() {
    this.docs = []
    this.byTitle = new Map() // "lang|title" -> doc index
    this.postings = new Map() // term -> [docIdx, tf, docIdx, tf, ...]
    this.docLen = []
    this.totalLen = 0
    this.keys = new Map() // normalised title/alias phrase -> [docIdx * 4 + kind]
    this.scores = new Float32Array(0)
    this.learnedCount = 0
    this.version = 0
  }

  get size() {
    return this.docs.length
  }

  _addKey(phraseTokens, idx, kind) {
    if (!phraseTokens.length || phraseTokens.length > 8) return
    const key = phraseTokens.join(' ')
    const packed = idx * 4 + kind
    const list = this.keys.get(key)
    if (!list) this.keys.set(key, [packed])
    else if (!list.includes(packed)) list.push(packed)
  }

  _index(idx, tokens, weight) {
    if (!tokens.length) return
    const counts = new Map()
    for (let i = 0; i < tokens.length; i++) counts.set(tokens[i], (counts.get(tokens[i]) || 0) + weight)
    for (const [term, tf] of counts) {
      const list = this.postings.get(term)
      if (list) list.push(idx, tf)
      else this.postings.set(term, [idx, tf])
    }
    this.docLen[idx] += tokens.length * weight
    this.totalLen += tokens.length * weight
  }

  /**
   * Adds an article, or enriches the existing one with the same title
   * (new aliases, and extra text when the new record knows more).
   * Returns the document index.
   */
  add(raw) {
    const doc = normalizeDoc(raw)
    if (!doc) return -1
    const id = `${doc.lang}|${doc.title}`
    const existing = this.byTitle.get(id)

    if (existing !== undefined) {
      const cur = this.docs[existing]
      for (const alias of doc.aliases) {
        if (!cur.aliases.includes(alias)) {
          cur.aliases.push(alias)
          const toks = tokenize(alias, cur.lang)
          this._addKey(toks, existing, 2)
          this._index(existing, toks, ALIAS_BOOST)
        }
      }
      if (doc.text.length > cur.text.length + 40) {
        const known = this.tokenSet(existing)
        const fresh = tokenize(doc.text, cur.lang).filter((t) => !known.has(t))
        cur.text = doc.text
        cur._sentences = null
        this._index(existing, fresh, 1)
        for (const t of fresh) known.add(t)
      }
      if (doc.learned && !cur.learned) {
        cur.learned = true
        this.learnedCount++
      }
      return existing
    }

    const idx = this.docs.length
    this.docs.push(doc)
    this.byTitle.set(id, idx)
    this.docLen.push(0)
    if (doc.learned) this.learnedCount++

    const titleTokens = tokenize(doc.title, doc.lang)
    this._addKey(titleTokens, idx, 0)
    const bare = doc.title.replace(/\s*\([^)]*\)\s*$/, '')
    if (bare !== doc.title) this._addKey(tokenize(bare, doc.lang), idx, 1)
    this._index(idx, titleTokens, TITLE_BOOST)
    for (const alias of doc.aliases) {
      const toks = tokenize(alias, doc.lang)
      this._addKey(toks, idx, 2)
      this._index(idx, toks, ALIAS_BOOST)
    }
    this._index(idx, tokenize(doc.text, doc.lang), 1)
    return idx
  }

  df(term) {
    const list = this.postings.get(term)
    return list ? list.length >> 1 : 0
  }

  /** Inverse document frequency; unknown terms get the maximum weight. */
  idf(term) {
    const n = Math.max(this.docs.length, 1)
    const df = this.df(term)
    return Math.log(1 + (n - df + 0.5) / (df + 0.5))
  }

  tokenSet(idx) {
    const doc = this.docs[idx]
    if (!doc._tokens) {
      doc._tokens = new Set(tokenize(`${doc.title} ${doc.aliases.join(' ')} ${doc.text}`, doc.lang))
    }
    return doc._tokens
  }

  titleSet(idx) {
    const doc = this.docs[idx]
    if (!doc._titleTokens) {
      doc._titleTokens = new Set(tokenize(`${doc.title} ${doc.aliases.join(' ')}`, doc.lang))
    }
    return doc._titleTokens
  }

  sentences(idx) {
    const doc = this.docs[idx]
    if (!doc._sentences) doc._sentences = splitSentences(doc.text)
    return doc._sentences
  }

  /** Okapi BM25 over the whole pool. Returns [{idx, score}] best first. */
  search(tokens, k = 8) {
    const n = this.docs.length
    if (!n || !tokens.length) return []
    if (this.scores.length < n) this.scores = new Float32Array(Math.ceil(n * 1.25) + 64)
    const scores = this.scores
    const touched = []
    const avgLen = this.totalLen / n || 1
    const seen = new Set()

    for (const term of tokens) {
      if (seen.has(term)) continue
      seen.add(term)
      const list = this.postings.get(term)
      if (!list) continue
      const idf = this.idf(term)
      for (let i = 0; i < list.length; i += 2) {
        const d = list[i]
        const tf = list[i + 1]
        const s = (idf * tf * (K1 + 1)) / (tf + K1 * (1 - B + (B * this.docLen[d]) / avgLen))
        if (scores[d] === 0) touched.push(d)
        scores[d] += s
      }
    }

    const top = []
    for (let i = 0; i < touched.length; i++) {
      const d = touched[i]
      const s = scores[d]
      scores[d] = 0
      if (top.length < k) {
        top.push({ idx: d, score: s })
        if (top.length === k) top.sort((a, b) => b.score - a.score)
      } else if (s > top[k - 1].score) {
        let j = k - 1
        while (j > 0 && top[j - 1].score < s) {
          top[j] = top[j - 1]
          j--
        }
        top[j] = { idx: d, score: s }
      }
    }
    if (top.length < k) top.sort((a, b) => b.score - a.score)
    return top
  }

  /**
   * Resolves which article a question is about.
   * Candidates come from exact title/alias phrases found inside the question
   * and from BM25; each is scored by how much of the question it explains.
   */
  findSubjects(tokens, lang = 'en', limit = 5) {
    if (!tokens.length || !this.docs.length) return []
    const weights = tokens.map((t) => this.idf(t))
    const total = weights.reduce((a, b) => a + b, 0) || 1

    const spans = new Map() // docIdx -> { share, kind }
    const maxLen = Math.min(tokens.length, 7)
    for (let len = maxLen; len >= 1; len--) {
      for (let start = 0; start + len <= tokens.length; start++) {
        const hits = this.keys.get(tokens.slice(start, start + len).join(' '))
        if (!hits) continue
        let share = 0
        for (let i = start; i < start + len; i++) share += weights[i]
        share /= total
        for (const packed of hits) {
          const idx = packed >> 2
          const kind = packed & 3
          const prev = spans.get(idx)
          if (!prev || share > prev.share || (share === prev.share && kind < prev.kind)) {
            spans.set(idx, { share, kind, start, len })
          }
        }
      }
    }

    const pool = new Set(spans.keys())
    for (const hit of this.search(tokens, 10)) pool.add(hit.idx)

    const scorePool = (targetLang) => {
      const candidates = []
      for (const idx of pool) {
        const doc = this.docs[idx]
        if (targetLang && doc.lang !== targetLang) continue
        const titleSet = this.titleSet(idx)
        const allSet = this.tokenSet(idx)
        let cov = 0
        const matched = []
        for (let i = 0; i < tokens.length; i++) {
          if (titleSet.has(tokens[i])) {
            cov += weights[i]
            matched.push(tokens[i])
          } else if (allSet.has(tokens[i])) {
            cov += weights[i] * 0.7
          }
        }
        cov /= total

        const span = spans.get(idx)
        const nameTokens = tokenize(span && span.kind === 1 ? doc.title.replace(/\s*\([^)]*\)\s*$/, '') : doc.title, doc.lang)
        let named = 0
        const querySet = new Set(tokens)
        for (const t of nameTokens) if (querySet.has(t)) named++
        const precision = span && span.kind === 2 ? 1 : nameTokens.length ? named / nameTokens.length : 0

        let rel = cov + (span ? 0.3 * span.share : 0) + 0.15 * precision
        if (doc.domain) rel += 0.03
        if (span && span.kind === 2) rel -= 0.02
        if (doc.learned) rel += 0.01
        candidates.push({ idx, doc, rel, cov, span: span || null, matched })
      }
      candidates.sort((a, b) => b.rel - a.rel)
      return candidates
    }

    let out = scorePool(lang)
    if (out.length === 0 && lang !== 'en') {
      out = scorePool(null)
    }
    return out.slice(0, limit)
  }

  /** Fast article search for the Knowledge Explorer UI. */
  searchArticles(query, lang = 'en', limit = 12) {
    const tokens = tokenize(query, lang, true)
    if (!tokens.length) return []
    const hits = this.findSubjects(tokens, lang, limit)
    return hits.map((h) => ({
      title: h.doc.title,
      domain: h.doc.domain || 'general',
      source: h.doc.source,
      lang: h.doc.lang,
      url: h.doc.url,
      aliases: h.doc.aliases,
      learned: h.doc.learned,
      score: Number(h.rel.toFixed(3)),
      summary: leadOf(this.sentences(h.idx), 340) || h.doc.text.slice(0, 340)
    }))
  }

  stats() {
    return { articles: this.docs.length, learned: this.learnedCount, terms: this.postings.size, version: this.version }
  }
}

// ---------------------------------------------------------------------------
// Question analysis
// ---------------------------------------------------------------------------

const LEAD_IN = /^(?:hey|hi|hello|ok|okay|so|well|um|uh|please|pls|kindly|voxasr|vox|assistant|bot|chatbot|and|but|also|now)[,!.\s]+/i
const POLITE = /^(?:can|could|would|will|may) (?:you|u|i)(?: please| kindly)?(?: also| just)?\s+/i
const ASK = /^(?:please\s+)?(?:tell me|explain to me|explain|describe|define|teach me|give me|show me|i want to know|i wanna know|i would like to know|i'd like to know|i need to know|i want to learn|do you know|let me know|help me understand|help me with|talk about|search for|search the web for|search|look up|lookup|find out|find|google|scrape|learn about|research|elaborate on|brief me on|ask about|get me|any idea)(?: more)?(?: something| anything| everything| a bit| a little)?(?: about| on| regarding| for| of)?\s+/i
const WH = /^(what|who|whom|whose|where|when|which|why|how)(?:'s|’s| is| are| was| were| does| do| did| can| could| would| should| will| has| have| exactly)?\s+/i
const OF_PREFIX = /^(?:the |an? )?(?:brief |short |quick |simple |detailed |basic |full |exact |formal )?(?:definition|meaning|explanation|overview|summary|introduction|concept|idea|basics|description|full form) (?:of|for|behind)\s+/i
const TRAILING = /\s+(?:work|works|worked|mean|means|stand for|stands for|do|does|is|are|was|were|defined|called|all about|about|please|pls|for me|to me|in simple terms|in simple words|in layman'?s? terms|in detail|in brief|in short|briefly|simply|exactly|with examples?|with an example|step by step|in computer science|in computing|in programming|in cs|in math|in mathematics)\s*$/i
const COMPARE = [
  /\b(?:differences?|different|distinction) (?:between|of|among) (.+?) (?:and|vs\.?|versus|&) (.+)$/i,
  /^(?:compare|comparing|comparison (?:of|between)) (.+?) (?:and|with|to|vs\.?|versus|&) (.+)$/i,
  /^(.+?) (?:vs\.?|versus) (.+)$/i,
  /^how (?:is|are|does|do) (.+?) (?:differ(?:ent)?|differs) from (.+)$/i
]

const NON_LATIN_PARTICLES = /(什么是|是什么|什麼是|是什麼|如何工作|如何|怎么样|怎么|为什么|為什麼|请问|请解释|解释一下|解释|的原理|是啥|とは何ですか|とは|って何|何ですか|ですか|について教えてください|について教えて|について|を説明して|教えてください|教えて|무엇입니까|무엇인가요|이란 무엇|란 무엇|설명해 주세요|설명해줘|ما هو|ما هي|ماذا|كيف يعمل|كيف تعمل|كيف|اشرح|لماذا|[?？¿¡。！!])/g

/** Breaks a message into topic words plus the kind of answer it wants. */
export function analyzeQuery(message, lang = 'en') {
  let text = String(message || '').replace(/https?:\/\/\S+/g, ' ').replace(/\s+/g, ' ').trim()
  text = text.replace(/[?？!！。]+\s*$/u, '').trim()
  let wh = null
  let howTo = false
  let comparison = null

  if (lang === 'en') {
    let prev
    do {
      prev = text
      text = text.replace(LEAD_IN, '').replace(POLITE, '').replace(ASK, '')
    } while (text !== prev && text)

    for (const re of COMPARE) {
      const m = text.match(re)
      if (m && m[1].trim() && m[2].trim() && m[1].length < 80 && m[2].length < 80) {
        const clean = (s) => s.replace(WH, '').replace(/^(?:the|an?) /i, '').replace(/[?.!]+$/, '').trim()
        comparison = [clean(m[1]), clean(m[2])]
        break
      }
    }

    const m = text.match(WH)
    if (m) {
      wh = m[1].toLowerCase()
      const rest = text.slice(m[0].length)
      if (wh === 'how') {
        const measure = rest.match(/^(many|much|long|old|far|big|large|tall|fast|often|high|deep|heavy|small)\b\s*/i)
        if (measure) {
          wh = 'howmany'
          text = rest
        } else if (/^(?:to|do i|can i|should i|would i|do you|can you|do we|can we|can one)\b/i.test(rest) || / to \w+/.test(m[0])) {
          howTo = true
          text = rest.replace(/^(?:to|do i|can i|should i|would i|do you|can you|do we|can we|can one)\s+/i, '')
        } else {
          text = rest
        }
      } else {
        text = rest
      }
    } else if (/^how to\s+/i.test(text)) {
      wh = 'how'
      howTo = true
      text = text.replace(/^how to\s+/i, '')
    }

    let prev2
    do {
      prev2 = text
      text = text.replace(OF_PREFIX, '').replace(TRAILING, '').replace(/^(?:the|an?) /i, '').trim()
    } while (text !== prev2 && text)
  }

  let tokens = tokenize(text, lang, false)
  if (!tokens.length) tokens = tokenize(text, lang, true)
  if (!tokens.length) tokens = tokenize(message, lang, true)

  let searchQuery
  if (lang === 'en') {
    searchQuery = text
  } else {
    const stripped = String(message).replace(/https?:\/\/\S+/g, ' ').replace(NON_LATIN_PARTICLES, ' ').replace(/\s+/g, ' ').trim()
    const words = stripped.split(' ').filter((w) => tokenize(w, lang, true).length)
    searchQuery = words.length ? words.join(' ') : stripped
  }

  return parseQueryDepthAndKind(message, { raw: String(message || ''), lang, topic: text, tokens, wh, howTo, comparison, searchQuery: searchQuery || text })
}

// ---------------------------------------------------------------------------
// Extractive answer selection
// ---------------------------------------------------------------------------

const SYNONYM_GROUPS = [
  'invent invented inventor invention create created creator creation develop developed developer design designed designer found founded founder discover discovered discoverer introduce introduced propose proposed conceive conceived devise devised author authored write wrote written build built make made pioneer pioneered originate originated establish established formulate formulated publish published release released launch launched begin began begun start started father',
  'complexity runtime running performance efficiency cost worst average asymptotic',
  'capital located location situated headquarter headquartered seat',
  'born birth birthday birthplace',
  'die died death dead killed',
  'use used usage application purpose applied utilize utilized employ employed useful',
  'advantage benefit strength merit pro',
  'disadvantage drawback limitation weakness downside con problem issue',
  'fast faster fastest speed quick quicker efficient',
  'big bigger biggest large larger largest size huge',
  'old older oldest age ancient first earliest',
  'cause caused reason because due why',
  'population inhabitant resident people',
  'type kind category class variant form',
  'difference differ different versus unlike whereas contrast compared',
  'step procedure process method algorithm approach technique',
  'formula equation expression defined given'
]
const SYNONYM_OF = new Map()
SYNONYM_GROUPS.forEach((group, gi) => {
  for (const w of group.split(' ')) SYNONYM_OF.set(stem(w), gi)
})

const TYPE_HINTS = {
  who: /\b\p{Lu}[\p{Ll}'’-]+(?:\s(?:van|von|de|da|di|del|bin|al|la|le|der|den))?\s(?:\p{Lu}\.\s)?\p{Lu}[\p{Ll}'’-]+/u,
  whom: /\b\p{Lu}[\p{Ll}'’-]+\s\p{Lu}[\p{Ll}'’-]+/u,
  when: /\b(?:1[0-9]{3}|20[0-9]{2})\b|\b\d{1,2}(?:st|nd|rd|th) century\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b|\b\d+ (?:BC|BCE|AD|CE)\b/,
  where: /\b(?:in|at|near|on|located|capital|city|country|region|coast|north|south|east|west|continent)\b/i,
  howmany: /\d/,
  why: /\b(?:because|due to|since|so that|in order to|reason|therefore|thus|hence|as a result|caused by|owing to|allows?|enables?)\b/i
}

/**
 * Picks the passage of `sentences` that best answers the question.
 * `focus` are the question's words that the article title does not already explain.
 */
export function bestPassage(sentences, focus, weightOf, wh = null, subjectTokens = null, lang = 'en') {
  if (!sentences.length) return null
  const totalWeight = focus.reduce((sum, f) => sum + weightOf(f), 0)
  const hint = wh && TYPE_HINTS[wh]
  let best = null

  for (let i = 0; i < sentences.length; i++) {
    const sentence = sentences[i]
    if (sentence.length < 25) continue
    const toks = tokenize(sentence, lang)
    const tokSet = new Set(toks)
    const groups = new Set()
    for (const t of toks) {
      const g = SYNONYM_OF.get(t)
      if (g !== undefined) groups.add(g)
    }

    let covered = 0
    for (const f of focus) {
      let hit = tokSet.has(f)
      if (!hit) {
        const g = SYNONYM_OF.get(f)
        hit = g !== undefined && groups.has(g)
      }
      if (!hit && f.length >= 4) {
        for (const t of tokSet) {
          if (looseMatch(f, t)) {
            hit = true
            break
          }
        }
      }
      if (hit) covered += weightOf(f)
    }
    const focusScore = totalWeight ? covered / totalWeight : 0
    const typed = hint ? hint.test(sentence) : false

    let score = focusScore + (typed ? (focus.length ? 0.45 : 0.9) : 0)
    if (subjectTokens) {
      for (const t of subjectTokens) {
        if (tokSet.has(t)) {
          score += 0.15
          break
        }
      }
    }
    if (i < 3) score += 0.04 * (3 - i)
    if (sentence.length > 480) score -= 0.2

    if (!best || score > best.score) best = { index: i, score, focusScore, typed }
  }

  if (!best) return null
  let text = sentences[best.index]
  const next = sentences[best.index + 1]
  if (next && text.length + next.length < 430) text += ' ' + next
  const satisfied = focus.length
    ? best.focusScore >= 0.6 || (best.typed && best.focusScore >= 0.3)
    : hint
    ? best.typed
    : true
  return { ...best, text, satisfied }
}

/** First sentences of an article, up to roughly `limit` characters. */
export function leadOf(sentences, limit = 620) {
  let out = ''
  for (let i = 0; i < sentences.length; i++) {
    const candidate = out ? out + ' ' + sentences[i] : sentences[i]
    if (candidate.length > limit && i >= 1) break
    out = candidate
    if (out.length >= limit) break
  }
  return out
}

/** Question words the matched article name does not account for. */
export function focusTerms(tokens, subject, index) {
  const titleSet = index.titleSet(subject.idx)
  return tokens.filter((t) => !titleSet.has(t) && !FILLER.has(t))
}
