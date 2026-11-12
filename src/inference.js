/**
 * Pure JavaScript Inference Engine + Multilingual CS/Math Classifier + ASR WER/CER Metrics
 * Loads the weights exported by `training/train_chatbot.py` and performs
 * exact Sublinear TF-IDF + L2-Normalized Softmax inference in <1ms across
 * all certified international languages and computer science / mathematical domains.
 */

import initialModel from './model/chatbot_model.json'

let activeModel = initialModel

export function reloadModel(newModelData) {
  if (newModelData && newModelData.vocabulary && newModelData.weights) {
    activeModel = newModelData
    return true
  }
  return false
}

export function normalizeText(text = '') {
  return String(text)
    .normalize('NFKC')
    .toLowerCase()
    .replace(/[^\p{L}\p{M}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

export function isCjkChar(ch) {
  if (!ch) return false
  const cp = ch.codePointAt(0)
  return (
    (cp >= 0x4e00 && cp <= 0x9fff) ||
    (cp >= 0x3400 && cp <= 0x4dbf) ||
    (cp >= 0x3040 && cp <= 0x309f) ||
    (cp >= 0x30a0 && cp <= 0x30ff) ||
    (cp >= 0xac00 && cp <= 0xd7af)
  )
}

export function detectScriptAndLanguage(rawText = '') {
  const text = String(rawText)
  if (/[\u0900-\u097F]/.test(text)) return { script: 'Devanagari', lang: 'hi-IN', label: 'Hindi' }
  if (/[\u0400-\u04FF]/.test(text)) return { script: 'Cyrillic', lang: 'ru-RU', label: 'Russian' }
  if (/[\u0600-\u06FF]/.test(text)) return { script: 'Arabic', lang: 'ar-SA', label: 'Arabic' }
  if (/[\u3040-\u309F\u30A0-\u30FF]/.test(text)) return { script: 'Kana', lang: 'ja-JP', label: 'Japanese' }
  if (/[\u4E00-\u9FFF]/.test(text)) return { script: 'Hanzi', lang: 'zh-CN', label: 'Mandarin Chinese' }
  if (/[\uAC00-\uD7AF]/.test(text)) return { script: 'Hangul', lang: 'ko-KR', label: 'Korean' }
  if (/[\u0980-\u09FF]/.test(text)) return { script: 'Bengali', lang: 'bn-IN', label: 'Bengali' }
  if (/[\u0B80-\u0BFF]/.test(text)) return { script: 'Tamil', lang: 'ta-IN', label: 'Tamil' }
  if (/[\u0C00-\u0C7F]/.test(text)) return { script: 'Telugu', lang: 'te-IN', label: 'Telugu' }
  if (/[\u0370-\u03FF]/.test(text)) return { script: 'Greek', lang: 'el-GR', label: 'Greek' }
  if (/[\u0590-\u05FF]/.test(text)) return { script: 'Hebrew', lang: 'he-IL', label: 'Hebrew' }
  if (/[\u0E00-\u0E7F]/.test(text)) return { script: 'Thai', lang: 'th-TH', label: 'Thai' }

  // Latin script language hints
  const lower = text.toLowerCase()
  if (/\b(qué|cómo|cuál|por qué|hola|gracias|buenos días)\b/i.test(lower)) {
    return { script: 'Latin', lang: 'es-ES', label: 'Spanish' }
  }
  if (/\b(qu'est-ce|comment|pourquoi|bonjour|merci|salut)\b/i.test(lower)) {
    return { script: 'Latin', lang: 'fr-FR', label: 'French' }
  }
  if (/\b(wie|warum|was ist|hallo|danke|guten morgen)\b/i.test(lower)) {
    return { script: 'Latin', lang: 'de-DE', label: 'German' }
  }
  if (/\b(você|como funciona|olá|obrigado|bom dia)\b/i.test(lower)) {
    return { script: 'Latin', lang: 'pt-BR', label: 'Portuguese' }
  }
  if (/\b(come funziona|cosa è|ciao|grazie|buongiorno)\b/i.test(lower)) {
    return { script: 'Latin', lang: 'it-IT', label: 'Italian' }
  }

  return { script: 'Latin', lang: 'en-US', label: 'English' }
}

export function extractTokens(rawText) {
  const norm = normalizeText(rawText)
  if (!norm) return []

  const words = norm.split(' ').filter(Boolean)
  const tokens = [...words]

  // Word 2-grams
  for (let i = 0; i < words.length - 1; i++) {
    tokens.push(`${words[i]}_${words[i + 1]}`)
  }

  // CJK character unigrams and bigrams
  const chars = Array.from(norm)
  const cjkChars = chars.filter(isCjkChar)
  for (let i = 0; i < cjkChars.length; i++) {
    tokens.push(`c:${cjkChars[i]}`)
    if (i < cjkChars.length - 1) {
      tokens.push(`c:${cjkChars[i]}${cjkChars[i + 1]}`)
    }
  }

  // Per-word Character 3-grams and 4-grams (mirrors Python training exactly)
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

export function predictIntent(userMessage) {
  const start = performance.now()
  const tokens = extractTokens(userMessage)
  const langMeta = detectScriptAndLanguage(userMessage)
  const model = activeModel

  const vocab = model.vocabulary
  const idf = model.idf
  const weights = model.weights
  const biases = model.biases

  // Term frequency over known vocabulary indices
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

  // Fallback if zero known features matched
  if (tfCounts.size === 0) {
    return {
      intentTag: 'algorithms_complexity',
      intentTitle: 'Computer Science & Mathematics AI Assistant',
      confidence: 0.15,
      detectedLanguage: langMeta,
      reply:
        "I didn't identify specific technical keywords in that message. You can ask me across any **Certified World Language** about:\n\n- **Algorithms & Complexity**: Dijkstra, A*, Dynamic Programming, P vs NP, Big-O\n- **Mathematical Computing**: SVD, QR/LU decomposition, Runge-Kutta RK4, FFT, Monte Carlo\n- **Machine Learning & Deep Learning**: Transformers, RoPE, Attention, Backprop, Diffusion, Loss Functions\n- **Speech AI & ASR**: OpenAI Whisper, Conformer, CTC Loss, WER Evaluation, Spectrograms\n- **Systems, Compilers & Cryptography**: Virtual Memory, Raft, LLVM IR, RSA, ECC, Zero-Knowledge Proofs\n\nOr click the **Auto-Upgrade Knowledge** button to trigger automated web scraping!",
      followUps: [
        'Explain Singular Value Decomposition (SVD)',
        'What is Dijkstra shortest path algorithm?',
        'How does OpenAI Whisper work?',
        'What is P vs NP in computational complexity?'
      ],
      topPredictions: [
        { tag: 'algorithms_complexity', title: 'Algorithms & Complexity', probability: 0.15 }
      ],
      matchedTokens: [],
      latencyMs: Number((performance.now() - start).toFixed(3))
    }
  }

  // Sublinear TF-IDF: (1 + ln(tf)) * idf, followed by L2 normalization
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

  // Compute logits = W * x + b
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

  // Numerically stable Softmax
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

  // Rank predictions
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

  let reply =
    intentObj.responses[0] ||
    intentObj.responses[Math.floor(Math.random() * intentObj.responses.length)]

  // Append dynamic web-scraped insight from ArXiv / Wikipedia if available
  if (intentObj.scraped_insights && intentObj.scraped_insights.length > 0) {
    const insight =
      intentObj.scraped_insights[Math.floor(Math.random() * intentObj.scraped_insights.length)]
    reply += `\n\n> 🌐 **Live Web-Harvested Ingestion (${insight.source})**: **${insight.title}**\n> ${insight.summary}`
  }

  if (best.probability < 0.28) {
    reply =
      `*Matched domain:* **${intentObj.title}** (${Math.round(best.probability * 100)}% match)\n\n` +
      reply
  }

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

/**
 * Compute Word Error Rate (WER), Character Error Rate (CER), and token-level
 * Levenshtein alignment backtrace (correct, substitution, deletion, insertion).
 */
export function computeWERAndAlignment(referenceRaw = '', hypothesisRaw = '') {
  const refNorm = normalizeText(referenceRaw)
  const hypNorm = normalizeText(hypothesisRaw)

  const refWords = refNorm ? refNorm.split(' ') : []
  const hypWords = hypNorm ? hypNorm.split(' ') : []

  const n = refWords.length
  const m = hypWords.length

  // DP cost matrix & operation backpointer matrix
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

  // Backtrace alignment
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

  // Character Error Rate (CER) ignoring extra spaces
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
