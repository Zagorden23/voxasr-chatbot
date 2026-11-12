/**
 * Pure JavaScript Inference Engine + Multilingual CS/Math Classifier +
 * Extractive Knowledge QA + Safe Math Evaluator + ASR WER/CER Metrics
 */

import fs from 'node:fs'
import path from 'node:path'
import {
  analyzeQuery,
  bestPassage,
  focusTerms,
  leadOf
} from './knowledge.js'
import { detectLanguage, isCjkChar, normalizeText, tokenize, wikiLang } from './text.js'
import { getKnowledgeIndex, learnTopicFromWeb } from './web_learner.js'

const MODEL_PATH = path.resolve(process.cwd(), 'src', 'model', 'chatbot_model.json')

let activeModel = JSON.parse(fs.readFileSync(MODEL_PATH, 'utf8'))

export function reloadModel(newModelData) {
  if (newModelData && newModelData.vocabulary && newModelData.weights && newModelData.classes) {
    activeModel = newModelData
    return true
  }
  return false
}

export function detectScriptAndLanguage(rawText = '', hint = '') {
  return detectLanguage(rawText, hint)
}

export function extractTokens(rawText) {
  const norm = normalizeText(rawText)
  if (!norm) return []

  const words = norm.split(' ').filter(Boolean)
  const tokens = [...words]

  for (let i = 0; i < words.length - 1; i++) {
    tokens.push(`${words[i]}_${words[i + 1]}`)
  }

  const chars = Array.from(norm)
  const cjkChars = chars.filter(isCjkChar)
  for (let i = 0; i < cjkChars.length; i++) {
    tokens.push(`c:${cjkChars[i]}`)
    if (i < cjkChars.length - 1) {
      tokens.push(`c:${cjkChars[i]}${cjkChars[i + 1]}`)
    }
  }

  for (const word of words) {
    if (word.length < 2 || Array.from(word).some(isCjkChar)) continue
    const padded = `<${word}>`
    for (const n of [3, 4]) {
      for (let i = 0; i <= padded.length - n; i++) {
        tokens.push(`c:${padded.slice(i, i + n)}`)
      }
    }
  }

  return tokens
}

// ---------------------------------------------------------------------------
// Injection-safe Recursive-Descent Mathematical Expression Evaluator
// ---------------------------------------------------------------------------

const MATH_CONSTANTS = {
  pi: Math.PI,
  e: Math.E,
  tau: Math.PI * 2,
  phi: (1 + Math.sqrt(5)) / 2
}

const MATH_FUNCS = {
  sqrt: (x) => Math.sqrt(x),
  abs: (x) => Math.abs(x),
  sin: (x) => Math.sin(x),
  cos: (x) => Math.cos(x),
  tan: (x) => Math.tan(x),
  ln: (x) => Math.log(x),
  log: (x) => Math.log10(x),
  log10: (x) => Math.log10(x),
  log2: (x) => Math.log2(x),
  exp: (x) => Math.exp(x),
  floor: (x) => Math.floor(x),
  ceil: (x) => Math.ceil(x),
  round: (x) => Math.round(x)
}

function tryEvaluateMathExpression(rawInput) {
  const cleaned = String(rawInput || '')
    .trim()
    .replace(/^(?:what\s+is|calculate|compute|evaluate|solve|find)\s+/i, '')
    .replace(/[?？=]+\s*$/g, '')
    .replace(/×/g, '*')
    .replace(/÷/g, '/')
    .replace(/−/g, '-')
    .replace(/\*\*/g, '^')
    .trim()

  if (!cleaned || cleaned.length > 100) return null
  // Must contain at least one digit and one operator or math function
  if (!/\d/.test(cleaned)) return null
  if (!/[+\-*/^%()]|\b(?:sqrt|sin|cos|tan|log|log2|log10|ln|exp|abs|floor|ceil|round)\b/i.test(cleaned)) {
    return null
  }
  // Reject prose words outside known functions/constants
  const wordTokens = cleaned.match(/[a-zA-Z_]+/g) || []
  for (const w of wordTokens) {
    const lw = w.toLowerCase()
    if (!(lw in MATH_FUNCS) && !(lw in MATH_CONSTANTS)) return null
  }

  try {
    const tokens = []
    let i = 0
    while (i < cleaned.length) {
      const ch = cleaned[i]
      if (/\s/.test(ch)) {
        i++
        continue
      }
      if (/[0-9.]/.test(ch)) {
        let numStr = ''
        while (i < cleaned.length && /[0-9.]/.test(cleaned[i])) {
          numStr += cleaned[i++]
        }
        const val = Number(numStr)
        if (!Number.isFinite(val)) return null
        tokens.push({ type: 'num', value: val })
        continue
      }
      if (/[a-zA-Z_]/.test(ch)) {
        let id = ''
        while (i < cleaned.length && /[a-zA-Z0-9_]/.test(cleaned[i])) {
          id += cleaned[i++]
        }
        tokens.push({ type: 'id', value: id.toLowerCase() })
        continue
      }
      if ('+-*/^%()'.includes(ch)) {
        tokens.push({ type: 'op', value: ch })
        i++
        continue
      }
      return null
    }

    let pos = 0
    const peek = () => tokens[pos]
    const consume = () => tokens[pos++]

    function parseExpr() {
      let left = parseTerm()
      while (peek()?.type === 'op' && (peek().value === '+' || peek().value === '-')) {
        const op = consume().value
        const right = parseTerm()
        left = op === '+' ? left + right : left - right
      }
      return left
    }

    function parseTerm() {
      let left = parsePower()
      while (peek()?.type === 'op' && (peek().value === '*' || peek().value === '/' || peek().value === '%')) {
        const op = consume().value
        const right = parsePower()
        if (op === '*') left *= right
        else if (op === '/') {
          if (right === 0) throw new Error('Division by zero')
          left /= right
        } else {
          if (right === 0) throw new Error('Modulo by zero')
          left %= right
        }
      }
      return left
    }

    function parsePower() {
      const base = parseUnary()
      if (peek()?.type === 'op' && peek().value === '^') {
        consume()
        const exp = parsePower()
        return Math.pow(base, exp)
      }
      return base
    }

    function parseUnary() {
      if (peek()?.type === 'op' && peek().value === '-') {
        consume()
        return -parseUnary()
      }
      if (peek()?.type === 'op' && peek().value === '+') {
        consume()
        return parseUnary()
      }
      return parsePrimary()
    }

    function parsePrimary() {
      const tok = peek()
      if (!tok) throw new Error('Unexpected end of expression')
      if (tok.type === 'num') {
        consume()
        return tok.value
      }
      if (tok.type === 'id') {
        consume()
        if (tok.value in MATH_CONSTANTS) return MATH_CONSTANTS[tok.value]
        if (tok.value in MATH_FUNCS) {
          if (peek()?.type !== 'op' || peek().value !== '(') throw new Error('Expected (')
          consume()
          const arg = parseExpr()
          if (peek()?.type !== 'op' || peek().value !== ')') throw new Error('Expected )')
          consume()
          return MATH_FUNCS[tok.value](arg)
        }
        throw new Error(`Unknown identifier ${tok.value}`)
      }
      if (tok.type === 'op' && tok.value === '(') {
        consume()
        const inner = parseExpr()
        if (peek()?.type !== 'op' || peek().value !== ')') throw new Error('Missing )')
        consume()
        return inner
      }
      throw new Error('Invalid syntax')
    }

    const result = parseExpr()
    if (pos !== tokens.length || !Number.isFinite(result)) return null
    const formatted = Number.isInteger(result)
      ? result.toString()
      : Number(result.toPrecision(10)).toString()
    return { expression: cleaned, result: formatted }
  } catch (err) {
    if (err.message === 'Division by zero' || err.message === 'Modulo by zero') {
      return { expression: cleaned, error: err.message }
    }
    return null
  }
}

// ---------------------------------------------------------------------------
// Softmax Intent Classifier (Sublinear TF-IDF + L2 Weights)
// ---------------------------------------------------------------------------

export function predictIntent(userMessage, langHint = '') {
  const start = performance.now()
  const tokens = extractTokens(userMessage)
  const langMeta = detectLanguage(userMessage, langHint)
  const model = activeModel

  const vocab = model.vocabulary
  const idf = model.idf
  const weights = model.weights
  const biases = model.biases

  const tfCounts = new Map()
  const matchedTokens = []

  for (const tok of tokens) {
    const idx = vocab[tok]
    if (idx !== undefined) {
      tfCounts.set(idx, (tfCounts.get(idx) || 0) + 1)
      if (!tok.startsWith('c:') && matchedTokens.length < 8 && !matchedTokens.includes(tok)) {
        matchedTokens.push(tok)
      }
    }
  }

  if (tfCounts.size === 0) {
    return {
      intentTag: 'unknown',
      intentTitle: 'General CS & Math Assistant',
      confidence: 0.15,
      detectedLanguage: langMeta,
      reply:
        "I didn't find matching keywords in my classifier vocabulary for that query. Try asking about a specific Computer Science or Mathematical Computing topic (e.g. **SVD**, **Dijkstra**, **Transformers**, **Raft**, **RSA**), or ask me to compute a math expression!",
      followUps: [
        'Explain Singular Value Decomposition (SVD)',
        'What is Dijkstra shortest path algorithm?',
        'How does OpenAI Whisper work?',
        'Compare B+ tree and LSM tree'
      ],
      topPredictions: [],
      matchedTokens: [],
      latencyMs: Number((performance.now() - start).toFixed(3))
    }
  }

  const sparseVec = []
  let normSq = 0

  for (const [idx, count] of tfCounts.entries()) {
    const val = (1 + Math.log(count)) * idf[idx]
    sparseVec.push([idx, val])
    normSq += val * val
  }

  const l2Norm = Math.sqrt(normSq) || 1
  for (let i = 0; i < sparseVec.length; i++) {
    sparseVec[i][1] /= l2Norm
  }

  const numClasses = model.classes.length
  const logits = new Array(numClasses)
  let maxLogit = -Infinity

  for (let c = 0; c < numClasses; c++) {
    let sum = biases[c]
    const wRow = weights[c]
    for (let j = 0; j < sparseVec.length; j++) {
      const [featIdx, featVal] = sparseVec[j]
      sum += wRow[featIdx] * featVal
    }
    logits[c] = sum
    if (sum > maxLogit) maxLogit = sum
  }

  let expSum = 0
  const probs = new Array(numClasses)
  for (let c = 0; c < numClasses; c++) {
    const e = Math.exp(logits[c] - maxLogit)
    probs[c] = e
    expSum += e
  }
  for (let c = 0; c < numClasses; c++) {
    probs[c] /= expSum
  }

  const ranked = probs
    .map((p, idx) => ({
      index: idx,
      tag: model.intents[idx].tag,
      title: model.intents[idx].title,
      probability: Number(p.toFixed(4))
    }))
    .sort((a, b) => b.probability - a.probability)

  const best = ranked[0]
  const intentObj = model.intents[best.index]
  const responses = intentObj.responses || []
  const reply =
    responses.length > 1
      ? responses[Math.floor(Math.random() * responses.length)]
      : responses[0] || ''

  return {
    intentTag: best.tag,
    intentTitle: intentObj.title,
    confidence: best.probability,
    detectedLanguage: langMeta,
    reply,
    followUps: intentObj.follow_ups || [],
    topPredictions: ranked.slice(0, 4),
    matchedTokens,
    latencyMs: Number((performance.now() - start).toFixed(3))
  }
}

// ---------------------------------------------------------------------------
// Intelligent Hybrid Reply (Softmax + Math + BM25 Extractive QA + Live Web)
// ---------------------------------------------------------------------------

const CONVERSATIONAL_INTENTS = new Set([
  'greeting',
  'thanks_goodbye',
  'language_support',
  'wer_evaluation',
  'knowledge_auto_upgrade'
])

const FOLLOWUP_PRONOUN_RE = /\b(it|its|this|that|they|them|their|same)\b/i

function formatKnowledgeAnswer(subject, passage, kbIndex, learnedNew = false) {
  const doc = subject.doc
  const sentences = kbIndex.sentences(subject.idx)
  const lead = leadOf(sentences, 680) || doc.text
  let body = ''

  if (passage && passage.text && !lead.startsWith(passage.text.slice(0, 40))) {
    body = `**${doc.title}**\n\n${passage.text}\n\n> **Background:** ${lead}`
  } else {
    body = `**${doc.title}**\n\n${lead}`
  }

  const badge = learnedNew
    ? `🌐 Learned live from ${doc.source}`
    : `📚 Knowledge Pool (${doc.source})`

  body += `\n\n_${badge}_`
  if (doc.url) {
    body += ` — ${doc.url}`
  }
  return body
}

export async function generateIntelligentReply(userMessage, options = {}) {
  const start = performance.now()
  const langHint = options.langHint || ''
  const contextTopic = options.contextTopic || null
  const autoLearn = options.autoLearn !== false

  const basePrediction = predictIntent(userMessage, langHint)
  const langMeta = basePrediction.detectedLanguage
  const shortLang = wikiLang(langMeta.lang || langHint || 'en-US')

  // 1. Mathematical expression evaluator
  const mathEval = tryEvaluateMathExpression(userMessage)
  if (mathEval) {
    const reply = mathEval.error
      ? `**Mathematical Evaluation Error**\n\nExpression: \`${mathEval.expression}\`\nResult: **${mathEval.error}**`
      : `**Computed Result**\n\n- Expression: \`${mathEval.expression}\`\n- Exact Value: **${mathEval.result}**`
    return {
      ...basePrediction,
      intentTag: 'numerical_analysis',
      intentTitle: 'Mathematical Expression Evaluator',
      confidence: 0.99,
      reply,
      sourceTitle: 'Math Evaluator',
      sourceUrl: null,
      sources: [],
      learnedNew: false,
      latencyMs: Number((performance.now() - start).toFixed(3))
    }
  }

  // 2. Conversational / meta-capability intents take priority when confident
  const wordCount = userMessage.trim().split(/\s+/).length
  if (
    CONVERSATIONAL_INTENTS.has(basePrediction.intentTag) &&
    (basePrediction.confidence >= 0.42 || (basePrediction.intentTag === 'greeting' && wordCount <= 5))
  ) {
    return {
      ...basePrediction,
      sources: [],
      sourceTitle: null,
      sourceUrl: null,
      learnedNew: false,
      latencyMs: Number((performance.now() - start).toFixed(3))
    }
  }

  const kbIndex = getKnowledgeIndex()

  // Resolve follow-up pronoun context if previous turn discussed a topic
  let effectiveQuery = userMessage
  if (contextTopic && FOLLOWUP_PRONOUN_RE.test(userMessage) && wordCount <= 10) {
    effectiveQuery = `${contextTopic} ${userMessage}`
  }

  const analysis = analyzeQuery(effectiveQuery, shortLang)

  // 3. Side-by-side comparison ("Compare A and B" / "A vs B")
  if (analysis.comparison && kbIndex.size > 0) {
    const [termA, termB] = analysis.comparison
    const findOrLearn = async (term) => {
      const toks = tokenize(term, shortLang, true)
      let hits = kbIndex.findSubjects(toks, shortLang, 1)
      if ((!hits.length || hits[0].rel < 0.42) && autoLearn) {
        await learnTopicFromWeb(term, shortLang, basePrediction.intentTag).catch(() => null)
        hits = kbIndex.findSubjects(toks, shortLang, 1)
      }
      return hits[0] || null
    }

    const [hitA, hitB] = await Promise.all([findOrLearn(termA), findOrLearn(termB)])
    if (hitA && hitB && hitA.idx !== hitB.idx) {
      const leadA = leadOf(kbIndex.sentences(hitA.idx), 420) || hitA.doc.text
      const leadB = leadOf(kbIndex.sentences(hitB.idx), 420) || hitB.doc.text
      const sources = [hitA.doc, hitB.doc]
        .filter((d) => d.url)
        .map((d) => ({ title: d.title, source: d.source, url: d.url }))

      const reply =
        `### Comparison: ${hitA.doc.title} vs. ${hitB.doc.title}\n\n` +
        `- **${hitA.doc.title}**: ${leadA}\n` +
        `- **${hitB.doc.title}**: ${leadB}`

      return {
        ...basePrediction,
        intentTitle: `${basePrediction.intentTitle} • Comparison`,
        reply,
        sourceTitle: `${hitA.doc.title} vs ${hitB.doc.title}`,
        sourceUrl: sources[0]?.url || null,
        sources,
        followUps: [
          `Explain ${hitA.doc.title} in detail`,
          `Explain ${hitB.doc.title} in detail`,
          `When should I use ${hitA.doc.title} over ${hitB.doc.title}?`
        ],
        learnedNew: Boolean(hitA.doc.learned || hitB.doc.learned),
        latencyMs: Number((performance.now() - start).toFixed(3))
      }
    }
  }

  // 4. Local BM25 Knowledge Pool + Extractive Passage Selection
  let learnedNew = false
  let hits = kbIndex.size > 0 ? kbIndex.findSubjects(analysis.tokens, shortLang, 3) : []
  let top = hits[0]

  const isWeakLocalHit = !top || (top.rel < 0.55 && !(top.span && top.span.share >= 0.45))

  // 5. Self-learning live web lookup if local pool has no strong match
  if (isWeakLocalHit && autoLearn && analysis.searchQuery && analysis.searchQuery.length >= 2) {
    const domainHint = basePrediction.intentTag !== 'unknown' ? basePrediction.intentTag : null
    const learnedDoc = await learnTopicFromWeb(analysis.searchQuery, shortLang, domainHint).catch(() => null)
    if (learnedDoc) {
      learnedNew = true
      hits = kbIndex.findSubjects(analysis.tokens, shortLang, 3)
      top = hits[0]
    }
  }

  if (top && (top.rel >= 0.36 || (top.span && top.span.share >= 0.3))) {
    const focus = focusTerms(analysis.tokens, top, kbIndex)
    const sentences = kbIndex.sentences(top.idx)
    const subjectTokens = tokenize(top.doc.title, top.doc.lang)
    const passage = bestPassage(
      sentences,
      focus,
      (t) => kbIndex.idf(t),
      analysis.wh,
      subjectTokens,
      top.doc.lang
    )

    const reply = formatKnowledgeAnswer(top, passage, kbIndex, learnedNew)
    const sources = top.doc.url
      ? [{ title: top.doc.title, source: top.doc.source, url: top.doc.url }]
      : []

    const relatedFollowUps = hits
      .slice(1, 3)
      .map((h) => `What is ${h.doc.title}?`)
      .concat([`How is ${top.doc.title} used in practice?`])
      .slice(0, 3)

    const effectiveTag = top.doc.domain || basePrediction.intentTag
    const matchedIntent = activeModel.intents.find((it) => it.tag === effectiveTag)

    return {
      ...basePrediction,
      intentTag: effectiveTag,
      intentTitle: matchedIntent ? matchedIntent.title : basePrediction.intentTitle,
      confidence: Number(Math.max(basePrediction.confidence, Math.min(0.98, top.rel)).toFixed(4)),
      reply,
      sourceTitle: top.doc.title,
      sourceUrl: top.doc.url || null,
      sources,
      followUps: relatedFollowUps.length ? relatedFollowUps : basePrediction.followUps,
      learnedNew,
      latencyMs: Number((performance.now() - start).toFixed(3))
    }
  }

  // 6. Fallback to trained Softmax domain summary + scraped insights
  const intentObj = activeModel.intents.find((it) => it.tag === basePrediction.intentTag)
  let reply = basePrediction.reply
  const sources = []
  if (intentObj?.scraped_insights?.length > 0 && basePrediction.confidence >= 0.25) {
    const insight = intentObj.scraped_insights[0]
    if (insight.url) {
      sources.push({ title: insight.title, source: insight.source, url: insight.url })
    }
    reply += `\n\n> 🌐 **Featured Knowledge (${insight.source})**: **${insight.title}**\n> ${insight.summary}`
  }

  return {
    ...basePrediction,
    reply,
    sourceTitle: intentObj?.title || null,
    sourceUrl: sources[0]?.url || null,
    sources,
    learnedNew: false,
    latencyMs: Number((performance.now() - start).toFixed(3))
  }
}

// ---------------------------------------------------------------------------
// Word Error Rate (WER), CER & Word-Level Levenshtein Alignment
// ---------------------------------------------------------------------------

export function computeWERAndAlignment(referenceRaw = '', hypothesisRaw = '') {
  const refNorm = normalizeText(referenceRaw)
  const hypNorm = normalizeText(hypothesisRaw)

  const refWords = refNorm ? refNorm.split(' ') : []
  const hypWords = hypNorm ? hypNorm.split(' ') : []

  const n = refWords.length
  const m = hypWords.length

  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0))
  const ops = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(''))

  for (let i = 1; i <= n; i++) {
    dp[i][0] = i
    ops[i][0] = 'D'
  }
  for (let j = 1; j <= m; j++) {
    dp[0][j] = j
    ops[0][j] = 'I'
  }

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      if (refWords[i - 1] === hypWords[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1]
        ops[i][j] = 'C'
      } else {
        const subCost = dp[i - 1][j - 1] + 1
        const delCost = dp[i - 1][j] + 1
        const insCost = dp[i][j - 1] + 1

        if (subCost <= delCost && subCost <= insCost) {
          dp[i][j] = subCost
          ops[i][j] = 'S'
        } else if (delCost <= insCost) {
          dp[i][j] = delCost
          ops[i][j] = 'D'
        } else {
          dp[i][j] = insCost
          ops[i][j] = 'I'
        }
      }
    }
  }

  let i = n
  let j = m
  const alignment = []
  let substitutions = 0
  let deletions = 0
  let insertions = 0
  let hits = 0

  while (i > 0 || j > 0) {
    const op = ops[i][j]
    if (op === 'C') {
      alignment.push({ type: 'correct', ref: refWords[i - 1], hyp: hypWords[j - 1] })
      hits++
      i--
      j--
    } else if (op === 'S') {
      alignment.push({ type: 'substitution', ref: refWords[i - 1], hyp: hypWords[j - 1] })
      substitutions++
      i--
      j--
    } else if (op === 'D') {
      alignment.push({ type: 'deletion', ref: refWords[i - 1], hyp: null })
      deletions++
      i--
    } else {
      alignment.push({ type: 'insertion', ref: null, hyp: hypWords[j - 1] })
      insertions++
      j--
    }
  }
  alignment.reverse()

  const wer = n > 0 ? (substitutions + deletions + insertions) / n : m > 0 ? 1 : 0
  const wordAccuracy = Math.max(0, 1 - wer)

  const refChars = refNorm.replace(/\s+/g, '').split('')
  const hypChars = hypNorm.replace(/\s+/g, '').split('')
  const charDist = levenshteinDistance(refChars, hypChars)
  const cer = refChars.length > 0 ? charDist / refChars.length : hypChars.length > 0 ? 1 : 0

  return {
    wer: Number((wer * 100).toFixed(2)),
    cer: Number((cer * 100).toFixed(2)),
    wordAccuracy: Number((wordAccuracy * 100).toFixed(2)),
    referenceWordCount: n,
    hypothesisWordCount: m,
    hits,
    substitutions,
    deletions,
    insertions,
    alignment
  }
}

function levenshteinDistance(a, b) {
  const n = a.length
  const m = b.length
  let prev = Array.from({ length: m + 1 }, (_, idx) => idx)
  for (let i = 1; i <= n; i++) {
    const curr = [i]
    for (let j = 1; j <= m; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1
      curr[j] = Math.min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
    }
    prev = curr
  }
  return prev[m]
}

export function getModelSummary() {
  const model = activeModel
  return {
    ...model.model_info,
    intents: model.intents.map((it) => ({
      tag: it.tag,
      title: it.title,
      pattern_count: it.pattern_count,
      sample_follow_ups: it.follow_ups
    }))
  }
}
