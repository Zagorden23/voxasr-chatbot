/**
 * Pure JavaScript Inference Engine + ASR Evaluation Metrics (WER / CER)
 * Loads the weights exported by `training/train_chatbot.py` and performs
 * exact Sublinear TF-IDF + L2-Normalized Softmax inference in <1ms.
 */

import trainedModel from './model/chatbot_model.json'

export function normalizeText(text = '') {
  return String(text)
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

export function extractTokens(rawText) {
  const norm = normalizeText(rawText)
  if (!norm) return []

  const words = norm.split(' ')
  const tokens = [...words]

  // Word 2-grams
  for (let i = 0; i < words.length - 1; i++) {
    tokens.push(`${words[i]}_${words[i + 1]}`)
  }

  // Per-word Character 3-grams and 4-grams (mirrors Python training exactly)
  for (const word of words) {
    if (word.length < 2) continue
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
  const vocab = trainedModel.vocabulary
  const idf = trainedModel.idf
  const weights = trainedModel.weights
  const biases = trainedModel.biases

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
      intentTag: 'asr_overview',
      intentTitle: 'ASR Assistant Fallback',
      confidence: 0.18,
      reply:
        "I didn't catch a specific ASR keyword in that phrase! You can ask me about **Automatic Speech Recognition**, **OpenAI Whisper**, **MFCC & Log-Mel Spectrograms**, **CTC vs. RNN-T vs. Conformer**, **Wav2Vec 2.0**, **Voice Activity Detection (VAD)**, or **Word Error Rate (WER)**.",
      followUps: [
        'What is Automatic Speech Recognition?',
        'How does OpenAI Whisper work?',
        'How do you calculate Word Error Rate?'
      ],
      topPredictions: [
        { tag: 'asr_overview', title: 'ASR Fundamentals & Pipeline', probability: 0.18 }
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
  const numClasses = trainedModel.classes.length
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
      tag: trainedModel.intents[idx].tag,
      title: trainedModel.intents[idx].title,
      probability: Number(p.toFixed(4))
    }))
    .sort((a, b) => b.probability - a.probability)

  const best = ranked[0]
  const intentObj = trainedModel.intents[best.index]

  // Check if user directly asked to compute WER inline like: "wer: ref | hyp"
  let reply =
    intentObj.responses[Math.floor(Math.random() * intentObj.responses.length)]

  if (best.probability < 0.24) {
    reply =
      `I matched your message closest to **${intentObj.title}** (${Math.round(best.probability * 100)}% confidence):\n\n` +
      reply
  }

  return {
    intentTag: best.tag,
    intentTitle: intentObj.title,
    confidence: best.probability,
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
  return {
    ...trainedModel.model_info,
    intents: trainedModel.intents.map((it) => ({
      tag: it.tag,
      title: it.title,
      pattern_count: it.pattern_count,
      sample_follow_ups: it.follow_ups
    }))
  }
}
