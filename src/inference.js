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
import { browseAndLearnTopic, getKnowledgeIndex } from './web_learner.js'
import { synthesizeMultiSourceAnswer } from './query_analyzer.js'
import {
  buildReactionVector,
  computeShannonEntropyNormalized,
  INTROSPECTION_ACTIONS,
  runTextCnnForward,
  softmaxArray,
  stepRecursiveIntrospection
} from './neural_introspection.js'

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

  if (tfCounts.size === 0 && !model.cnn) {
    return {
      intentTag: 'unknown',
      intentTitle: 'Omni-Domain Knowledge Assistant',
      confidence: 0.15,
      detectedLanguage: langMeta,
      reply:
        "Let me search across the live web and my multi-domain knowledge pool for that query! You can ask about any field—Science, Medicine, Engineering, Commerce, Arts, Music, Dance, Sports, Entertainment, History, CS, or Mathematics.",
      followUps: [
        'Explain quantum entanglement in brief',
        'Explain cricket LBW rules in detail',
        'How does Hindustani classical raga work?',
        'Compare Keynesian and Classical economics'
      ],
      topPredictions: [],
      matchedTokens: [],
      hiddenState: null,
      cnnLogits: null,
      allProbs: null,
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
  const linProbs = new Array(numClasses)
  for (let c = 0; c < numClasses; c++) {
    const e = Math.exp(logits[c] - maxLogit)
    linProbs[c] = e
    expSum += e
  }
  for (let c = 0; c < numClasses; c++) {
    linProbs[c] /= expSum
  }

  // Execute Multi-Scale 1D TextCNN forward pass when trained CNN weights are present
  const cnnRes = model.cnn ? runTextCnnForward(userMessage, sparseVec, model.cnn) : null
  const probs = cnnRes
    ? linProbs.map((lp, c) => 0.72 * cnnRes.probs[c] + 0.28 * lp)
    : linProbs

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
    allProbs: probs,
    hiddenState: cnnRes?.hidden || null,
    cnnLogits: cnnRes?.logits || logits,
    cnnSeqTokens: cnnRes?.seqTokens || [],
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

function docToSynthesizable(doc) {
  if (!doc) return null
  return {
    ...doc,
    extract: doc.extract || doc.text || ''
  }
}

function buildStepTrace(stepNum, actionTaken, stepOut, reaction) {
  if (!stepOut) {
    return {
      step: stepNum,
      action: actionTaken,
      nextAction: actionTaken,
      sufficiencyQ: Number(Math.min(0.95, 0.4 + reaction[6] * 0.5).toFixed(3)),
      webSources: Math.round(reaction[10] * 4),
      bm25Rel: Number((reaction[0] * 12).toFixed(2))
    }
  }
  return {
    step: stepNum,
    action: actionTaken,
    nextAction: stepOut.nextAction,
    nextActionProb: stepOut.nextActionProb,
    sufficiencyQ: stepOut.qHat,
    gateRetention: stepOut.gateRetention,
    webSources: Math.round(reaction[10] * 4),
    bm25Rel: Number((reaction[0] * 12).toFixed(2))
  }
}

function makeConversationalTrace(actionName, prediction, depthMode) {
  const model = activeModel
  const probs = prediction.allProbs || []
  const reaction = buildReactionVector({
    bm25TopScore: 8,
    spanCount: 2,
    tokenCoverage: 0.9,
    domainAgree: 1.0,
    clfTopProb: prediction.confidence || 0.9,
    clfEntropy: computeShannonEntropyNormalized(probs),
    passageChars: 600,
    depthMode,
    isComparison: false,
    webSourceCount: 0,
    sentenceRichness: 0.8
  })
  const actionIdx = Math.max(0, INTROSPECTION_ACTIONS.indexOf(actionName))
  const s1 = model.introspection && prediction.hiddenState
    ? stepRecursiveIntrospection(model.introspection, prediction.hiddenState, actionIdx, reaction)
    : null
  return {
    steps: [buildStepTrace(1, actionName, s1, reaction)],
    finalAction: s1?.nextAction || actionName,
    finalSufficiencyQ: s1?.qHat ?? 0.96,
    totalWebSources: 0,
    scrapedHosts: [],
    blockedChallengeCount: 0
  }
}

function mergeDocsForSynthesis(liveDocs, kbHits) {
  const out = []
  const seen = new Set()
  const addItem = (rawDoc, relScore = 1.0) => {
    const doc = docToSynthesizable(rawDoc)
    if (!doc || !doc.extract || doc.extract.length < 40) return
    const key = doc.url || `${String(doc.source || 'KB').toLowerCase()}::${String(doc.title || '').toLowerCase()}`
    if (seen.has(key)) return
    seen.add(key)
    out.push({ ...doc, _score: relScore })
  }

  // Interleave live external web sources first alongside high-relevance KB documents
  for (const d of liveDocs || []) addItem({ ...d, _liveWeb: true }, 2.2)
  for (const h of kbHits || []) addItem(h.doc, 1.0 + (h.rel || 0))
  return out
}

function refineIntentFromDeltas(basePrediction, deltaSum) {
  const model = activeModel
  if (!basePrediction.cnnLogits || !deltaSum || deltaSum.length !== model.classes.length) {
    return basePrediction
  }
  const combined = basePrediction.cnnLogits.map((l, c) => l + 0.3 * deltaSum[c])
  const refinedProbs = softmaxArray(combined)
  const ranked = refinedProbs
    .map((p, idx) => ({
      index: idx,
      tag: model.intents[idx].tag,
      title: model.intents[idx].title,
      probability: Number(p.toFixed(4))
    }))
    .sort((a, b) => b.probability - a.probability)
  const best = ranked[0]
  if (!best) return basePrediction
  return {
    ...basePrediction,
    intentTag: best.tag,
    intentTitle: best.title,
    confidence: best.probability,
    topPredictions: ranked.slice(0, 4)
  }
}

export async function generateIntelligentReply(userMessage, options = {}) {
  const start = performance.now()
  const langHint = options.langHint || ''
  const contextTopic = options.contextTopic || null
  const autoLearn = options.autoLearn !== false
  const skipLiveWeb = Boolean(options.skipLiveWeb)

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
      depthMode: 'brief',
      questionKind: 'math',
      scrapedHosts: [],
      introspection: makeConversationalTrace('evaluate_math', basePrediction, 'brief'),
      learnedNew: false,
      latencyMs: Number((performance.now() - start).toFixed(3))
    }
  }

  const kbIndex = getKnowledgeIndex()
  const wordCount = userMessage.trim().split(/\s+/).length

  // Resolve follow-up pronoun context if previous turn discussed a topic
  let effectiveQuery = userMessage
  if (contextTopic && FOLLOWUP_PRONOUN_RE.test(userMessage) && wordCount <= 10) {
    effectiveQuery = `${contextTopic} ${userMessage}`
  }

  const analysis = analyzeQuery(effectiveQuery, shortLang)

  // 2. Conversational / meta-capability intents take priority when confident & no URL pasted
  if (
    analysis.directUrls.length === 0 &&
    CONVERSATIONAL_INTENTS.has(basePrediction.intentTag) &&
    (basePrediction.confidence >= 0.48 || (basePrediction.intentTag === 'greeting' && wordCount <= 4))
  ) {
    return {
      ...basePrediction,
      sources: [],
      sourceTitle: null,
      sourceUrl: null,
      depthMode: analysis.depthMode,
      questionKind: 'conversational',
      scrapedHosts: [],
      introspection: makeConversationalTrace('answer_conversational', basePrediction, analysis.depthMode),
      learnedNew: false,
      latencyMs: Number((performance.now() - start).toFixed(3))
    }
  }

  const introModel = activeModel.introspection
  const probs0 = basePrediction.allProbs || []
  const entropy0 = computeShannonEntropyNormalized(probs0)
  const isCompare = Boolean(analysis.comparison || analysis.questionKind === 'pros_cons')

  // -------------------- STEP 1: Initial Local KB Probe + Reaction r^(0) --------------------
  let hits = kbIndex.size > 0 ? kbIndex.findSubjects(analysis.tokens, shortLang, 5) : []
  let top = hits[0]
  const localNonWikiDocs = hits.filter(
    (h) => h.doc?.learned && h.doc?.source && !String(h.doc.source).toLowerCase().includes('wikipedia')
  )
  const localCharSum = hits.slice(0, 3).reduce((acc, h) => acc + String(h.doc?.text || '').length, 0)
  const initActionName = isCompare ? 'compare_entities' : 'retrieve_local_kb'
  const initActionIdx = Math.max(0, INTROSPECTION_ACTIONS.indexOf(initActionName))

  const reaction0 = buildReactionVector({
    bm25TopScore: (top?.rel || 0) * 12,
    spanCount: hits.length,
    tokenCoverage: top?.span?.share || (top ? top.rel : 0),
    domainAgree: top?.doc?.domain === basePrediction.intentTag ? 1.0 : 0.45,
    clfTopProb: basePrediction.confidence,
    clfEntropy: entropy0,
    passageChars: localCharSum,
    depthMode: analysis.depthMode,
    isComparison: isCompare,
    webSourceCount: localNonWikiDocs.length,
    sentenceRichness: Math.min(1, localCharSum / 1800)
  })

  let zState = basePrediction.hiddenState ? [...basePrediction.hiddenState] : null
  const deltaSum = zState ? new Array(activeModel.classes.length).fill(0) : null
  const step1 = introModel && zState ? stepRecursiveIntrospection(introModel, zState, initActionIdx, reaction0) : null
  if (step1) {
    zState = step1.zNext
    for (let c = 0; c < deltaSum.length; c++) deltaSum[c] += step1.delta[c]
  }
  const traceSteps = [buildStepTrace(1, initActionName, step1, reaction0)]

  // -------------------- STEP 2: Multi-Website Browser Action Dispatch --------------------
  const needsLiveBrowse =
    !skipLiveWeb &&
    autoLearn &&
    (analysis.directUrls.length > 0 ||
      Boolean(analysis.searchQuery && analysis.searchQuery.length >= 2) &&
        (localNonWikiDocs.length < 2 ||
          !top ||
          top.rel < 0.72 ||
          analysis.depthMode === 'detailed' ||
          step1?.nextAction === 'browse_web_multi' ||
          step1?.nextAction === 'synthesize_deep' ||
          (step1 && step1.qHat < 0.78)))

  let liveDocs = []
  let scrapedHosts = []
  let blockedChallengeCount = 0
  let learnedNew = false
  let compareBlocks = null
  const domainHint = basePrediction.intentTag !== 'unknown' ? basePrediction.intentTag : null

  if (needsLiveBrowse) {
    if (analysis.comparison && analysis.comparison.length === 2) {
      const [termA, termB] = analysis.comparison
      const [resA, resB, resOverview] = await Promise.all([
        browseAndLearnTopic(termA, shortLang, domainHint, { ...analysis, directUrls: [] }).catch(() => ({
          docs: [],
          scrapedHosts: [],
          blockedUrls: []
        })),
        browseAndLearnTopic(termB, shortLang, domainHint, { ...analysis, directUrls: [] }).catch(() => ({
          docs: [],
          scrapedHosts: [],
          blockedUrls: []
        })),
        browseAndLearnTopic(analysis.searchQuery, shortLang, domainHint, analysis).catch(() => ({
          docs: [],
          scrapedHosts: [],
          blockedUrls: []
        }))
      ])
      liveDocs = [...(resOverview.docs || []), ...(resA.docs || []), ...(resB.docs || [])]
      scrapedHosts = [...new Set([...(resOverview.scrapedHosts || []), ...(resA.scrapedHosts || []), ...(resB.scrapedHosts || [])])]
      blockedChallengeCount += (resA.blockedUrls?.length || 0) + (resB.blockedUrls?.length || 0) + (resOverview.blockedUrls?.length || 0)
    } else {
      const resWeb = await browseAndLearnTopic(analysis.searchQuery, shortLang, domainHint, analysis).catch(() => ({
        docs: [],
        scrapedHosts: [],
        blockedUrls: []
      }))
      liveDocs = resWeb.docs || []
      scrapedHosts = resWeb.scrapedHosts || []
      blockedChallengeCount += resWeb.blockedUrls?.length || 0
    }

    if (liveDocs.length > 0) {
      learnedNew = true
      hits = kbIndex.size > 0 ? kbIndex.findSubjects(analysis.tokens, shortLang, 5) : []
      top = hits[0]
    }
  }

  const midChars =
    liveDocs.reduce((acc, d) => acc + String(d.extract || '').length, 0) +
    hits.slice(0, 2).reduce((acc, h) => acc + String(h.doc?.text || '').length, 0)
  const webSourceCount1 = new Set(liveDocs.map((d) => d.url || d.title)).size
  const action1Name = needsLiveBrowse ? 'browse_web_multi' : step1?.nextAction || 'retrieve_local_kb'
  const action1Idx = Math.max(0, INTROSPECTION_ACTIONS.indexOf(action1Name))

  const reaction1 = buildReactionVector({
    bm25TopScore: Math.max((top?.rel || 0) * 12, liveDocs.length > 0 ? 7.5 : 0),
    spanCount: Math.max(hits.length, liveDocs.length),
    tokenCoverage: Math.max(top?.span?.share || 0, liveDocs.length > 0 ? 0.78 : 0),
    domainAgree: top?.doc?.domain === basePrediction.intentTag || liveDocs.length > 0 ? 1.0 : 0.4,
    clfTopProb: basePrediction.confidence,
    clfEntropy: entropy0 * 0.85,
    passageChars: midChars,
    depthMode: analysis.depthMode,
    isComparison: isCompare,
    webSourceCount: webSourceCount1,
    sentenceRichness: Math.min(1, midChars / 2200)
  })

  const step2 = introModel && zState ? stepRecursiveIntrospection(introModel, zState, action1Idx, reaction1) : null
  if (step2) {
    zState = step2.zNext
    for (let c = 0; c < deltaSum.length; c++) deltaSum[c] += step2.delta[c]
  }
  traceSteps.push(buildStepTrace(2, action1Name, step2, reaction1))

  // -------------------- STEP 3: Deep Verification Hop & Final Synthesis --------------------
  if (
    !skipLiveWeb &&
    autoLearn &&
    analysis.depthMode === 'detailed' &&
    !analysis.comparison &&
    liveDocs.length > 0 &&
    liveDocs.length < 3 &&
    analysis.searchQuery
  ) {
    const aspectHint = analysis.targetAspects?.[0] || (analysis.questionKind === 'how_to' ? 'guide steps examples' : 'detailed overview')
    const hopQuery = `${analysis.searchQuery} ${aspectHint}`.trim()
    const hopRes = await browseAndLearnTopic(hopQuery, shortLang, domainHint, analysis).catch(() => null)
    if (hopRes?.docs?.length) {
      liveDocs = [...liveDocs, ...hopRes.docs]
      scrapedHosts = [...new Set([...scrapedHosts, ...(hopRes.scrapedHosts || [])])]
      blockedChallengeCount += hopRes.blockedUrls?.length || 0
    }
  }

  // Prepare side-by-side comparison entities if comparison query
  if (analysis.comparison && (kbIndex.size > 0 || liveDocs.length > 0)) {
    const [termA, termB] = analysis.comparison
    const hitPair = [termA, termB].map((term) => {
      const toks = tokenize(term, shortLang, false)
      const kbMatch = kbIndex.size > 0 ? kbIndex.findSubjects(toks, shortLang, 1)[0]?.doc : null
      if (kbMatch) return docToSynthesizable(kbMatch)
      const webMatch = liveDocs.find((d) => String(d.title || '').toLowerCase().includes(term.toLowerCase()))
      return docToSynthesizable(webMatch)
    })
    if (hitPair[0] && hitPair[1] && hitPair[0].title !== hitPair[1].title) {
      compareBlocks = {
        termA,
        termB,
        docA: hitPair[0],
        docB: hitPair[1],
        focusA: tokenize(termA, shortLang, false),
        focusB: tokenize(termB, shortLang, false)
      }
    }
  }

  const mergedDocs = mergeDocsForSynthesis(liveDocs, hits)
  const finalChars = mergedDocs.slice(0, 4).reduce((acc, d) => acc + d.extract.length, 0)
  const totalWebCount = new Set(mergedDocs.filter((d) => d._liveWeb || d.learned).map((d) => d.url || d.title)).size
  const finalActionIdx = Math.max(0, INTROSPECTION_ACTIONS.indexOf('synthesize_deep'))

  const reaction2 = buildReactionVector({
    bm25TopScore: Math.max((top?.rel || 0) * 12, mergedDocs.length > 0 ? 9 : 0),
    spanCount: mergedDocs.length,
    tokenCoverage: Math.max(top?.span?.share || 0, mergedDocs.length > 0 ? 0.85 : 0),
    domainAgree: 1.0,
    clfTopProb: basePrediction.confidence,
    clfEntropy: entropy0 * 0.7,
    passageChars: finalChars,
    depthMode: analysis.depthMode,
    isComparison: isCompare,
    webSourceCount: totalWebCount,
    sentenceRichness: Math.min(1, finalChars / 2600)
  })

  const step3 = introModel && zState ? stepRecursiveIntrospection(introModel, zState, finalActionIdx, reaction2) : null
  if (step3) {
    for (let c = 0; c < deltaSum.length; c++) deltaSum[c] += step3.delta[c]
  }
  traceSteps.push(buildStepTrace(3, 'synthesize_deep', step3, reaction2))

  const refinedPrediction = refineIntentFromDeltas(basePrediction, deltaSum)
  const docDomainValid =
    top?.doc?.domain && activeModel.classes.includes(top.doc.domain) ? top.doc.domain : null
  const effectiveTag = docDomainValid || refinedPrediction.intentTag
  const matchedIntent = activeModel.intents.find((it) => it.tag === effectiveTag)
  const intentTitle = matchedIntent ? matchedIntent.title : refinedPrediction.intentTitle

  const introspectionSummary = {
    steps: traceSteps,
    finalAction: step3?.nextAction || 'synthesize_deep',
    finalSufficiencyQ: step3?.qHat ?? step2?.qHat ?? step1?.qHat ?? 0.9,
    totalWebSources: totalWebCount,
    scrapedHosts,
    blockedChallengeCount
  }

  if (mergedDocs.length > 0) {
    const synthesized = synthesizeMultiSourceAnswer({
      docs: mergedDocs,
      analysis,
      intentTitle,
      lang: shortLang,
      compareBlocks,
      learnedCount: liveDocs.length
    })
    if (synthesized && synthesized.reply) {
      return {
        ...refinedPrediction,
        intentTag: effectiveTag,
        intentTitle,
        confidence: Number(
          Math.max(refinedPrediction.confidence, Math.min(0.98, top?.rel || (liveDocs.length > 0 ? 0.88 : 0.75))).toFixed(4)
        ),
        reply: synthesized.reply,
        sourceTitle: mergedDocs[0]?.title || null,
        sourceUrl: synthesized.sources?.[0]?.url || null,
        sources: synthesized.sources || [],
        followUps: synthesized.followUps?.length ? synthesized.followUps : basePrediction.followUps,
        depthMode: analysis.depthMode,
        questionKind: analysis.questionKind,
        scrapedHosts,
        introspection: introspectionSummary,
        learnedNew,
        latencyMs: Number((performance.now() - start).toFixed(3))
      }
    }
  }

  // 6. Fallback to trained TextCNN domain summary + scraped insights
  const intentObj = activeModel.intents.find((it) => it.tag === refinedPrediction.intentTag)
  let reply = refinedPrediction.reply
  const sources = []
  if (intentObj?.scraped_insights?.length > 0 && refinedPrediction.confidence >= 0.2) {
    const insight = intentObj.scraped_insights[0]
    if (insight.url) {
      sources.push({ title: insight.title, source: insight.source, url: insight.url })
    }
    reply += `\n\n> 🌐 **Featured Knowledge (${insight.source})**: **${insight.title}**\n> ${insight.summary}`
  }

  return {
    ...refinedPrediction,
    reply,
    sourceTitle: intentObj?.title || null,
    sourceUrl: sources[0]?.url || null,
    sources,
    depthMode: analysis.depthMode,
    questionKind: analysis.questionKind,
    scrapedHosts,
    introspection: introspectionSummary,
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
  const cnnMeta = model.cnn
    ? {
        kernels: model.cnn.kernels || [2, 3, 4],
        filters_per_kernel: model.cnn.filters || 24,
        embedding_dim: model.cnn.emb_dim || 32,
        max_seq_len: model.cnn.max_len || 36,
        hidden_dim: model.cnn.hidden_dim || 96,
        cnn_vocab_size: Object.keys(model.cnn.vocab || {}).length
      }
    : null
  const introMeta = model.introspection
    ? {
        steps: model.introspection.steps || 3,
        actions: model.introspection.actions || INTROSPECTION_ACTIONS,
        reaction_dim: model.introspection.reaction_dim || 12
      }
    : null
  return {
    ...model.model_info,
    cnn_architecture: cnnMeta,
    introspection_architecture: introMeta,
    intents: model.intents.map((it) => ({
      tag: it.tag,
      title: it.title,
      pattern_count: it.pattern_count,
      sample_follow_ups: it.follow_ups
    }))
  }
}
