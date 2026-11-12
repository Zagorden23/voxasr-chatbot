/**
 * VoxASR Query Intent, Granularity Analyzer & Multi-Source Synthesizer
 *
 * Classifies response depth (`brief` | `standard` | `detailed`), question kind
 * (`definition`, `comparison`, `how_to`, `pros_cons`, `why`, `who`, `when`, `where`, `url_browse`),
 * cleans noisy phrasing into high-precision search queries, filters boilerplate from
 * scraped web text, deduplicates overlapping claims via token Jaccard similarity,
 * and synthesizes depth-appropriate answers citing multiple websites.
 */

import { cleanExtract, splitSentences, stem, tokenize, truncateAtSentence } from './text.js'

function localWikiUrl(title, lang = 'en') {
  return `https://${lang}.wikipedia.org/wiki/${encodeURIComponent(String(title || '').replace(/ /g, '_'))}`
}

function computeFocusTokens(tokens = [], subject = null) {
  if (!subject || !subject.title) return tokens
  const titleStems = new Set(tokenize(subject.title, subject.lang || 'en', false).map(stem))
  const filtered = tokens.filter((t) => !titleStems.has(stem(t)))
  return filtered.length > 0 ? filtered : tokens
}

const URL_REGEX = /https?:\/\/[^\s<>"')\]}]+/gi

const BRIEF_PATTERNS =
  /\b(in\s+brief|briefly|short\s+answer|in\s+short|shortly|tl;?dr|summarize|summary|one\s+sentence|two\s+sentences|concise|concisely|quick\s+overview|just\s+the\s+basics|quick\s+answer|key\s+point|in\s+a\s+nutshell|short\s+note|give\s+me\s+the\s+gist)\b/i

const DETAILED_PATTERNS =
  /\b(in\s+detail|in\s+details|detailed|comprehensive|comprehensively|thoroughly|deep\s+dive|in\s+depth|in-depth|elaborate|step\s+by\s+step|step-by-step|complete\s+guide|everything\s+about|explain\s+fully|full\s+explanation|extensive|extensively|with\s+examples|pros\s+and\s+cons|advantages\s+and\s+disadvantages|compare\s+in\s+detail|full\s+history|complete\s+overview|tell\s+me\s+all\s+about|full\s+details)\b/i

const DEPTH_MODIFIER_STRIP =
  /\b(please|kindly|for\s+me|in\s+detail|in\s+details|detailed|comprehensive|comprehensively|thoroughly|deep\s+dive|in\s+depth|in-depth|elaborate(?:\s+on)?|step\s+by\s+step|step-by-step|complete\s+guide\s+(?:to|on|for)|everything\s+about|explain\s+fully|full\s+explanation|extensive|extensively|with\s+examples|compare\s+in\s+detail|full\s+history\s+of|complete\s+overview\s+of|tell\s+me\s+all\s+about|full\s+details(?:\s+of)?|in\s+brief|briefly|short\s+answer|in\s+short|shortly|tl;?dr|summarize|summary\s+of|one\s+sentence|two\s+sentences|concise|concisely|quick\s+overview\s+of|just\s+the\s+basics(?:\s+of)?|quick\s+answer|in\s+a\s+nutshell|give\s+me\s+the\s+gist(?:\s+of)?|short\s+note\s+on)\b/gi

const HOW_TO_REGEX =
  /\b(how\s+to|steps\s+to|how\s+do\s+(?:i|we|you)|how\s+can\s+(?:i|we|one)|guide\s+to|recipe\s+for|tutorial\s+(?:on|for)|method\s+to)\b/i

const PROS_CONS_REGEX =
  /\b(pros\s+and\s+cons|advantages\s+and\s+disadvantages|benefits\s+and\s+drawbacks|merits\s+and\s+demerits)\b/i

const EXAMPLES_REGEX = /\b(examples?\s+of|types\s+of|kinds\s+of|categories\s+of|list\s+of)\b/i
const DEFINITION_REGEX = /\b(what\s+(?:is|are|was|were)|define|definition\s+of|meaning\s+of|what\s+does\s+.+\s+mean)\b/i

const ASPECT_KEYWORDS = [
  'history',
  'origin',
  'origins',
  'mechanism',
  'rules',
  'scoring',
  'examples',
  'causes',
  'symptoms',
  'treatment',
  'applications',
  'uses',
  'formula',
  'equations',
  'benefits',
  'risks',
  'economics',
  'biography',
  'career',
  'instruments',
  'styles',
  'techniques',
  'steps'
]

const BOILERPLATE_SENTENCE =
  /\b(accept\s+(?:all\s+)?cookies|cookie\s+(?:policy|settings|preferences)|privacy\s+policy|terms\s+(?:of\s+use|of\s+service)|all\s+rights\s+reserved|sign\s+in|log\s+in|create\s+an?\s+account|subscribe\s+(?:now|to)|newsletter|skip\s+to\s+(?:main\s+)?content|enable\s+javascript|javascript\s+is\s+disabled|click\s+here\s+to|read\s+more\s+at|advertisement|sponsored\s+content|table\s+of\s+contents|jump\s+to\s+navigation|edit\s+this\s+page)\b/i

const CAUSAL_WORDS = /\b(because|since|due\s+to|caused\s+by|causes|results?\s+in|leads?\s+to|therefore|consequently)\b/i
const PROCESS_WORDS = /\b(first|next|then|step|stage|phase|process|begin|mix|heat|apply|compile|compute|perform|execute|prepare)\b/i
const DEFINITION_CUES = /\b(is\s+an?|are\s+an?|refers?\s+to|defined\s+as|known\s+as|consists?\s+of|represents?)\b/i

export function extractDirectUrls(rawMessage = '') {
  const matches = String(rawMessage || '').match(URL_REGEX) || []
  const cleaned = []
  const seen = new Set()
  for (const raw of matches) {
    const clean = raw.replace(/[.,;:!?)'"\]]+$/, '')
    if (!seen.has(clean)) {
      seen.add(clean)
      cleaned.push(clean)
    }
  }
  return cleaned
}

export function cleanTopicPhrase(text = '') {
  return String(text || '')
    .replace(URL_REGEX, ' ')
    .replace(DEPTH_MODIFIER_STRIP, ' ')
    .replace(/^[\s,.:;-]+|[\s,.:;?!-]+$/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

export function parseQueryDepthAndKind(rawMessage = '', baseAnalysis = {}) {
  const raw = String(rawMessage || '').trim()
  const directUrls = extractDirectUrls(raw)
  const hasBrief = BRIEF_PATTERNS.test(raw)
  const hasDetailed = DETAILED_PATTERNS.test(raw)
  const questionMarks = (raw.match(/\?/g) || []).length
  const words = raw.split(/\s+/).filter(Boolean)

  let depthMode = 'standard'
  if (hasBrief && !hasDetailed) {
    depthMode = 'brief'
  } else if (hasDetailed) {
    depthMode = 'detailed'
  } else if (
    baseAnalysis.comparison ||
    PROS_CONS_REGEX.test(raw) ||
    questionMarks >= 2 ||
    words.length >= 14 ||
    (/\band\b/i.test(raw) && /\b(explain|compare|describe|history|how)\b/i.test(raw))
  ) {
    depthMode = 'detailed'
  } else if (words.length <= 3 && !baseAnalysis.wh && !HOW_TO_REGEX.test(raw)) {
    depthMode = 'standard'
  }

  let questionKind = 'general'
  if (directUrls.length > 0) questionKind = 'url_browse'
  else if (baseAnalysis.comparison) questionKind = 'comparison'
  else if (PROS_CONS_REGEX.test(raw)) questionKind = 'pros_cons'
  else if (HOW_TO_REGEX.test(raw)) questionKind = 'how_to'
  else if (baseAnalysis.wh) questionKind = baseAnalysis.wh
  else if (EXAMPLES_REGEX.test(raw)) questionKind = 'examples'
  else if (DEFINITION_REGEX.test(raw)) questionKind = 'definition'

  const lower = raw.toLowerCase()
  const targetAspects = ASPECT_KEYWORDS.filter((kw) => new RegExp(`\\b${kw}\\b`, 'i').test(lower))

  const cleanedTopic = cleanTopicPhrase(baseAnalysis.topic || raw) || cleanTopicPhrase(raw) || raw
  const cleanedSearch =
    cleanTopicPhrase(baseAnalysis.searchQuery || cleanedTopic) || cleanedTopic || raw

  return {
    ...baseAnalysis,
    topic: cleanedTopic,
    searchQuery: cleanedSearch,
    depthMode,
    questionKind,
    directUrls,
    targetAspects
  }
}

export function isBoilerplateSentence(sentence = '') {
  const s = String(sentence || '').trim()
  if (s.length < 24) return true
  if (BOILERPLATE_SENTENCE.test(s)) return true
  const letters = (s.match(/[a-zA-Z\p{L}]/gu) || []).length
  if (letters / Math.max(s.length, 1) < 0.45) return true
  if (/^(https?:\/\/|www\.|menu|home|search|copyright|©)/i.test(s)) return true
  return false
}

export function sentenceJaccard(a = '', b = '', lang = 'en') {
  const tokA = new Set(tokenize(a, lang, false).map(stem))
  const tokB = new Set(tokenize(b, lang, false).map(stem))
  if (tokA.size === 0 || tokB.size === 0) return 0
  let inter = 0
  for (const t of tokA) if (tokB.has(t)) inter += 1
  return inter / (tokA.size + tokB.size - inter)
}

export function extractRankedSentences(
  text,
  focusTokens = [],
  analysis = {},
  maxSentences = 5,
  charBudget = 1200
) {
  const cleaned = cleanExtract(String(text || ''))
  const rawSents = splitSentences(cleaned).filter((s) => !isBoilerplateSentence(s))
  if (rawSents.length === 0) return []

  const lang = analysis.lang || 'en'
  const wh = analysis.wh || null
  const qKind = analysis.questionKind || 'general'
  const aspectSet = new Set((analysis.targetAspects || []).map(stem))
  const focusStems = focusTokens.map((t) => stem(t))

  const scored = rawSents.map((sent, idx) => {
    const sentStems = new Set(tokenize(sent, lang, false).map(stem))
    let overlap = 0
    for (const fs of focusStems) {
      if (sentStems.has(fs)) overlap += 1
    }
    let aspectHits = 0
    for (const as of aspectSet) {
      if (sentStems.has(as)) aspectHits += 1
    }

    let score = overlap * 3.0 + aspectHits * 2.5
    if (idx === 0) score += 2.2
    else if (idx === 1) score += 1.4

    if ((qKind === 'definition' || qKind === 'general') && DEFINITION_CUES.test(sent)) score += 2.0
    if ((qKind === 'why' || wh === 'why') && CAUSAL_WORDS.test(sent)) score += 3.2
    if (qKind === 'how_to' && PROCESS_WORDS.test(sent)) score += 2.6
    if ((qKind === 'when' || wh === 'when') && /\b(1[0-9]{3}|20[0-2][0-9]|[0-9]{4}\s*BCE|century)\b/i.test(sent)) {
      score += 3.5
    }
    if ((qKind === 'howmany' || wh === 'howmany') && /\b\d+(?:\.\d+)?%?\b/.test(sent)) score += 2.8
    if (/\d/.test(sent)) score += 0.6
    if (sent.length > 340) score -= 0.7

    return { sent, idx, score }
  })

  scored.sort((a, b) => b.score - a.score || a.idx - b.idx)

  const chosen = []
  let usedChars = 0
  for (const item of scored) {
    if (chosen.length >= maxSentences) break
    if (usedChars + item.sent.length + 2 > charBudget && chosen.length > 0) continue
    const duplicate = chosen.some((c) => sentenceJaccard(c.sent, item.sent, lang) >= 0.52)
    if (duplicate) continue
    chosen.push(item)
    usedChars += item.sent.length + 2
  }

  chosen.sort((a, b) => a.idx - b.idx)
  return chosen.map((c) => c.sent)
}

function docUrlOf(doc, lang = 'en') {
  if (!doc) return localWikiUrl('Main_Page', lang)
  if (doc.url) return doc.url
  const src = String(doc.source || '').toLowerCase()
  if (src === 'arxiv' || String(doc.id || '').startsWith('arxiv_')) {
    return `https://arxiv.org/abs/${String(doc.id).replace(/^arxiv_/, '')}`
  }
  return localWikiUrl(doc.title || 'Knowledge', doc.lang || lang)
}

function dedupeSourcesList(docs, lang = 'en', maxSources = 6) {
  const seenUrls = new Set()
  const seenTitles = new Set()
  const sourceTypeCount = new Map()
  const out = []
  for (const d of docs) {
    if (!d || !d.title) continue
    const url = docUrlOf(d, lang)
    const srcLower = String(d.source || 'KB').toLowerCase()
    const keyTitle = `${srcLower}::${d.title.toLowerCase()}`
    if (seenUrls.has(url) || seenTitles.has(keyTitle)) continue

    // Prevent single encyclopedia/index from crowding out independent websites
    const bucket = srcLower.includes('wikipedia')
      ? 'wikipedia'
      : srcLower.includes('openalex') || srcLower.includes('crossref')
      ? 'academic'
      : srcLower
    const cap = bucket === 'wikipedia' || bucket === 'academic' ? 1 : 2
    if ((sourceTypeCount.get(bucket) || 0) >= cap && out.length >= 2) continue

    seenUrls.add(url)
    seenTitles.add(keyTitle)
    sourceTypeCount.set(bucket, (sourceTypeCount.get(bucket) || 0) + 1)
    out.push({
      title: d.title,
      source: d.source || (d.learned ? 'Live Web' : 'KB'),
      url,
      domain: d.domain || null,
      snippet: truncateAtSentence(cleanExtract(d.extract || ''), 180)
    })
    if (out.length >= maxSources) break
  }
  return out
}

function docPriority(doc, queryStems, rawQuery = '') {
  const txt = `${doc.title || ''} ${doc.extract || ''}`
  const docStems = new Set(tokenize(txt, doc.lang || 'en', false).map(stem))
  let overlap = 0
  for (const qs of queryStems) if (docStems.has(qs)) overlap += 1
  const src = String(doc.source || '').toLowerCase()
  const isResearchQ = /\b(research|paper|study|journal|literature|clinical trial|citation)\b/i.test(rawQuery)
  const isAcademic = src.includes('openalex') || src.includes('crossref') || src.includes('arxiv')
  const isExternalSite = Boolean(doc.url) && !src.includes('wikipedia') && !isAcademic && src !== 'kb' && src !== 'wikidata'
  const webBonus = isExternalSite ? 2.4 : src.includes('wikipedia') || src.includes('duckduckgo') ? 1.4 : isAcademic ? (isResearchQ ? 2.0 : -0.6) : 0.4
  const freshBonus = doc._liveWeb ? 1.2 : 0
  return (doc._score || 1.0) + overlap * 0.5 + webBonus + freshBonus
}

export function synthesizeMultiSourceAnswer({
  docs = [],
  analysis = {},
  intentTitle = 'Omni-Domain Knowledge',
  lang = 'en',
  compareBlocks = null,
  learnedCount = 0
}) {
  const depth = analysis.depthMode || 'standard'
  const qKind = analysis.questionKind || 'general'
  const qTokens = tokenize(analysis.searchQuery || analysis.topic || '', lang, false)
  const qStems = qTokens.map(stem)

  if (compareBlocks && compareBlocks.docA && compareBlocks.docB) {
    const { termA, termB, docA, docB, focusA, focusB } = compareBlocks
    const charPerSide = depth === 'brief' ? 240 : depth === 'detailed' ? 750 : 420
    const sentsA = extractRankedSentences(docA.extract, focusA, analysis, depth === 'detailed' ? 4 : 2, charPerSide)
    const sentsB = extractRankedSentences(docB.extract, focusB, analysis, depth === 'detailed' ? 4 : 2, charPerSide)
    const parts = [
      `## Comparison: **${docA.title}** vs **${docB.title}**`,
      `**${docA.title}:** ${sentsA.join(' ') || truncateAtSentence(cleanExtract(docA.extract || ''), charPerSide)}`,
      `**${docB.title}:** ${sentsB.join(' ') || truncateAtSentence(cleanExtract(docB.extract || ''), charPerSide)}`
    ]
    if (depth === 'detailed' && docs.length > 2) {
      const extraFacts = []
      for (const d of docs.slice(2, 5)) {
        const s = extractRankedSentences(d.extract, qTokens, analysis, 2, 340)
        if (s.length) extraFacts.push(`- **[${d.source || 'Web'}] ${d.title}:** ${s.join(' ')}`)
      }
      if (extraFacts.length) {
        parts.push(`**Broader Context & Multi-Source Insights:**\n${extraFacts.join('\n')}`)
      }
    }
    const sources = dedupeSourcesList([docA, docB, ...docs], lang, 6)
    return {
      reply: parts.join('\n\n'),
      sources,
      followUps: [
        `How is ${termA} used in practice?`,
        `Advantages and disadvantages of ${termB}`,
        `Explain ${termA} in brief`
      ]
    }
  }

  const rankedDocs = [...docs].sort(
    (a, b) => docPriority(b, qStems, analysis.raw || '') - docPriority(a, qStems, analysis.raw || '')
  )
  const sources = dedupeSourcesList(rankedDocs, lang, depth === 'brief' ? 3 : 6)
  const primary = rankedDocs[0]
  if (!primary) return null

  const pFocus = computeFocusTokens(analysis.tokens || qTokens, primary)
  const globalAccepted = []
  const sectionsByDoc = []

  const maxDocsToSynthesize = depth === 'brief' ? 2 : depth === 'detailed' ? 5 : 3
  const maxSentsPerPrimary = depth === 'brief' ? 2 : depth === 'detailed' ? 5 : 3
  const maxSentsSecondary = depth === 'brief' ? 1 : depth === 'detailed' ? 3 : 2

  for (let i = 0; i < Math.min(rankedDocs.length, maxDocsToSynthesize); i++) {
    const doc = rankedDocs[i]
    const rawSents = extractRankedSentences(
      doc.extract,
      i === 0 ? pFocus : qTokens,
      analysis,
      i === 0 ? maxSentsPerPrimary : maxSentsSecondary,
      i === 0 ? (depth === 'brief' ? 420 : depth === 'detailed' ? 1200 : 700) : depth === 'detailed' ? 700 : 420
    )
    const uniqueSents = []
    for (const s of rawSents) {
      if (!globalAccepted.some((prev) => sentenceJaccard(prev, s, lang) >= 0.5)) {
        globalAccepted.push(s)
        uniqueSents.push(s)
      }
    }
    if (uniqueSents.length > 0) {
      sectionsByDoc.push({
        doc,
        srcIndex: sources.findIndex((s) => s.url === docUrlOf(doc, lang)) + 1 || i + 1,
        text: uniqueSents.join(' '),
        sentences: uniqueSents
      })
    }
  }

  if (sectionsByDoc.length === 0) {
    const fallback = truncateAtSentence(cleanExtract(primary.extract || ''), depth === 'brief' ? 360 : 650)
    return {
      reply: `**${primary.title}** — ${fallback}`,
      sources,
      followUps: [`Explain ${primary.title} in detail`, `Examples of ${primary.title}`]
    }
  }

  const head = sectionsByDoc[0]
  const rest = sectionsByDoc.slice(1)

  if (depth === 'brief') {
    const summaryText = truncateAtSentence(
      [head.text, rest[0]?.sentences?.[0]].filter(Boolean).join(' '),
      460
    )
    const citeNote =
      sources.length > 1 ? `\n\n_Quick summary synthesized from ${sources.length} web sources._` : ''
    return {
      reply: `**${primary.title} (Brief Summary):** ${summaryText}${citeNote}`,
      sources,
      followUps: [
        `Explain ${primary.title} in detail`,
        `Key examples or applications of ${primary.title}`,
        `History and evolution of ${primary.title}`
      ]
    }
  }

  if (depth === 'standard') {
    const paras = [`**${primary.title}** — ${head.text} [${head.srcIndex}]`]
    if (rest.length > 0) {
      const bullets = rest.map(
        (r) => `- **[${r.doc.source || 'Web'}] ${r.doc.title} [${r.srcIndex}]:** ${truncateAtSentence(r.text, 340)}`
      )
      paras.push(`**Additional Web Context:**\n${bullets.join('\n')}`)
    }
    return {
      reply: paras.join('\n\n'),
      sources,
      followUps: [
        `Explain ${primary.title} in detail`,
        `Give me ${primary.title} in brief`,
        sources[1] ? `More about ${sources[1].title}` : `Real-world applications of ${primary.title}`
      ]
    }
  }

  // DETAILED mode: structured multi-website synthesis
  const blocks = []
  blocks.push(
    `## ${primary.title} — Comprehensive Multi-Source Answer\n${head.text} [${head.srcIndex}]`
  )

  if (head.sentences.length > 2) {
    const leadRemainder = extractRankedSentences(
      primary.extract,
      qTokens,
      analysis,
      4,
      900
    ).filter((s) => !head.sentences.includes(s))
    if (leadRemainder.length > 0) {
      blocks.push(
        `**Key Details (${primary.source || 'Primary Source'} [${head.srcIndex}]):**\n${leadRemainder
          .map((s) => `- ${s}`)
          .join('\n')}`
      )
    }
  }

  if (rest.length > 0) {
    const heading =
      qKind === 'how_to'
        ? '**Process, Techniques & Multi-Site Guidance:**'
        : qKind === 'pros_cons'
        ? '**Perspectives, Trade-offs & Analysis:**'
        : '**Corroborating Findings Across Independent Websites:**'
    const lines = rest.map(
      (r) =>
        `### ${r.srcIndex}. ${r.doc.title} (${r.doc.source || 'Web'})\n${r.text} [${r.srcIndex}]`
    )
    blocks.push(`${heading}\n\n${lines.join('\n\n')}`)
  }

  if (learnedCount > 0 || sources.length > 1) {
    const domainList = sources.map((s) => `[${s.source}]`).join(' • ')
    blocks.push(
      `_Synthesized across ${sources.length} source${sources.length === 1 ? '' : 's'} (${domainList}) for domain: ${intentTitle}._`
    )
  }

  return {
    reply: blocks.join('\n\n'),
    sources,
    followUps: [
      `Summarize ${primary.title} in brief`,
      `Practical examples of ${primary.title}`,
      rest[0] ? `Compare ${primary.title} and ${rest[0].doc.title}` : `History of ${primary.title}`
    ]
  }
}
