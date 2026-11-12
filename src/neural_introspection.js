/**
 * VoxASR Pure-JS Multi-Scale 1D TextCNN & 3-Step Recursive Action-Reaction Introspection Engine
 *
 * Executes exact inference matching `training/cnn_introspection.py`:
 *  1. Multi-Scale 1D Convolutional Neural Network (k = 2, 3, 4 filters + Max/Mean pooling)
 *     fused with TF-IDF Dense Projection -> Latent Representation z^(0) & Softmax Probabilities.
 *  2. 3-Step Gated Recursive Introspection Network:
 *     z^(s+1) = g * z^(s) + (1 - g) * tanh(W_u [z^(s); a^(s); r^(s)] + b_u)
 *     producing next-action policy pi^(s+1), epistemic sufficiency critic q^(s+1),
 *     and domain refinement logit delta^(s+1).
 */

import { isCjkChar, normalizeText } from './text.js'

export const INTROSPECTION_ACTIONS = [
  'answer_conversational',
  'evaluate_math',
  'retrieve_local_kb',
  'browse_web_multi',
  'compare_entities',
  'synthesize_deep'
]

export const NUM_ACTIONS = INTROSPECTION_ACTIONS.length
export const REACTION_DIM = 12

export function softmaxArray(logits) {
  let max = -Infinity
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > max) max = logits[i]
  }
  const exps = new Array(logits.length)
  let sum = 0
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max)
    exps[i] = e
    sum += e
  }
  const denom = sum || 1
  for (let i = 0; i < exps.length; i++) exps[i] /= denom
  return exps
}

export function sigmoidScalar(x) {
  const clipped = x < -20 ? -20 : x > 20 ? 20 : x
  return 1 / (1 + Math.exp(-clipped))
}

export function extractCnnSequenceTokens(rawText = '', maxLen = 36) {
  const norm = normalizeText(rawText)
  if (!norm) return []
  const seq = []
  for (const word of norm.split(' ')) {
    if (!word) continue
    let hasCjk = false
    for (let i = 0; i < word.length; i++) {
      if (isCjkChar(word[i])) {
        hasCjk = true
        break
      }
    }
    if (hasCjk) {
      for (let i = 0; i < word.length; i++) {
        const ch = word[i]
        if (isCjkChar(ch) || seq.length === 0 || seq[seq.length - 1] !== ch) {
          seq.push(ch)
        }
      }
    } else {
      seq.push(word)
      if (word.length >= 6) {
        seq.push(`#${word.slice(0, 4)}`)
        seq.push(`#${word.slice(-4)}`)
      }
    }
    if (seq.length >= maxLen) break
  }
  return seq.slice(0, maxLen)
}

export function runTextCnnForward(rawText, tfidfEntries, cnnModel) {
  if (!cnnModel || !cnnModel.vocab || !cnnModel.E || !cnnModel.convs) {
    return null
  }

  const maxLen = cnnModel.max_len || 36
  const embDim = cnnModel.emb_dim || 32
  const hiddenDim = cnnModel.hidden_dim || 96
  const tfProjDim = cnnModel.tfidf_proj_dim || 48
  const kernels = cnnModel.kernels || [2, 3, 4]
  const filters = cnnModel.filters || 24
  const vocab = cnnModel.vocab
  const unkId = vocab['<UNK>'] ?? 1

  const seqToks = extractCnnSequenceTokens(rawText, maxLen)
  const seqLen = Math.max(1, seqToks.length)
  const seqIds = new Int32Array(maxLen)
  for (let i = 0; i < seqToks.length && i < maxLen; i++) {
    seqIds[i] = vocab[seqToks[i]] ?? unkId
  }

  // Embedding lookup (masked on padding 0)
  const E = cnnModel.E
  const emb = new Float32Array(maxLen * embDim)
  for (let t = 0; t < maxLen; t++) {
    const id = seqIds[t]
    if (id === 0) continue
    const row = E[id]
    if (!row) continue
    const off = t * embDim
    for (let d = 0; d < embDim; d++) emb[off + d] = row[d]
  }

  // Multi-scale 1D convolutions + ReLU + Max/Mean pooling
  const zCnnParts = []
  for (const k of kernels) {
    const convLayer = cnnModel.convs[String(k)]
    if (!convLayer) return null
    const W = convLayer.W // (k*D) x F
    const b = convLayer.b // F
    const L = maxLen - k + 1
    const validWins = Math.max(1, seqLen - k + 1)
    const count = Math.min(L, validWins)

    const maxPool = new Float32Array(filters)
    const sumPool = new Float32Array(filters)

    for (let f = 0; f < filters; f++) maxPool[f] = -1e9

    for (let pos = 0; pos < count; pos++) {
      const kD = k * embDim
      for (let f = 0; f < filters; f++) {
        let acc = b[f] || 0
        for (let j = 0; j < k; j++) {
          const eOff = (pos + j) * embDim
          const wBase = j * embDim
          for (let d = 0; d < embDim; d++) {
            acc += emb[eOff + d] * W[wBase + d][f]
          }
        }
        const act = acc > 0 ? acc : 0
        sumPool[f] += act
        if (act > maxPool[f]) maxPool[f] = act
      }
    }

    const denom = Math.max(1, count)
    for (let f = 0; f < filters; f++) {
      zCnnParts.push(maxPool[f] > 0 ? maxPool[f] : 0)
    }
    for (let f = 0; f < filters; f++) {
      zCnnParts.push(sumPool[f] / denom)
    }
  }

  // Sparse TF-IDF dense branch
  const Wsp = cnnModel.W_sp
  const bsp = cnnModel.b_sp
  const zSp = new Float32Array(tfProjDim)
  for (let f = 0; f < tfProjDim; f++) zSp[f] = bsp[f] || 0
  for (const [idx, val] of tfidfEntries) {
    const row = Wsp[idx]
    if (!row) continue
    for (let f = 0; f < tfProjDim; f++) {
      zSp[f] += val * row[f]
    }
  }
  for (let f = 0; f < tfProjDim; f++) zSp[f] = Math.tanh(zSp[f])

  // Concatenate z_cnn + z_sp -> Fusion Hidden Layer h^(0) -> Logits
  const zCat = new Float32Array(zCnnParts.length + tfProjDim)
  for (let i = 0; i < zCnnParts.length; i++) zCat[i] = zCnnParts[i]
  for (let i = 0; i < tfProjDim; i++) zCat[zCnnParts.length + i] = zSp[i]

  const Wh = cnnModel.W_h
  const bh = cnnModel.b_h
  const hidden = new Float32Array(hiddenDim)
  for (let j = 0; j < hiddenDim; j++) {
    let acc = bh[j] || 0
    for (let i = 0; i < zCat.length; i++) acc += zCat[i] * Wh[i][j]
    hidden[j] = Math.tanh(acc)
  }

  const Wcls = cnnModel.W_cls
  const bcls = cnnModel.b_cls
  const numClasses = bcls.length
  const logits = new Array(numClasses)
  for (let c = 0; c < numClasses; c++) {
    let acc = bcls[c] || 0
    for (let j = 0; j < hiddenDim; j++) acc += hidden[j] * Wcls[j][c]
    logits[c] = acc
  }

  const probs = softmaxArray(logits)
  return {
    logits,
    probs,
    hidden: Array.from(hidden),
    seqTokens: seqToks.slice(0, 12)
  }
}

export function buildReactionVector({
  bm25TopScore = 0,
  spanCount = 0,
  tokenCoverage = 0,
  domainAgree = 0.5,
  clfTopProb = 0.5,
  clfEntropy = 0.4,
  passageChars = 0,
  depthMode = 'standard',
  isComparison = false,
  webSourceCount = 0,
  sentenceRichness = 0.4
}) {
  const targetChars = depthMode === 'detailed' ? 1450 : depthMode === 'brief' ? 280 : 650
  return [
    Math.min(1, Math.max(0, bm25TopScore / 12)),
    Math.min(1, Math.max(0, spanCount / 5)),
    Math.min(1, Math.max(0, tokenCoverage)),
    Math.min(1, Math.max(0, domainAgree)),
    Math.min(1, Math.max(0, clfTopProb)),
    Math.min(1, Math.max(0, clfEntropy)),
    Math.min(1, Math.max(0, passageChars / targetChars)),
    depthMode === 'brief' ? 1.0 : 0.0,
    depthMode === 'detailed' ? 1.0 : 0.0,
    isComparison ? 1.0 : 0.0,
    Math.min(1, Math.max(0, webSourceCount / 4)),
    Math.min(1, Math.max(0, sentenceRichness))
  ]
}

export function stepRecursiveIntrospection(introModel, zState, actionIdx, reactionVec) {
  if (!introModel || !introModel.W_g || !introModel.W_u || !zState) {
    return null
  }
  const hiddenDim = zState.length
  const inG = zState.concat(reactionVec)
  const g = new Array(hiddenDim)
  for (let j = 0; j < hiddenDim; j++) {
    let acc = introModel.b_g[j] || 0
    for (let i = 0; i < inG.length; i++) acc += inG[i] * introModel.W_g[i][j]
    g[j] = sigmoidScalar(acc)
  }

  const actionOh = new Array(NUM_ACTIONS).fill(0)
  if (actionIdx >= 0 && actionIdx < NUM_ACTIONS) actionOh[actionIdx] = 1
  const inU = zState.concat(actionOh, reactionVec)
  const zNext = new Array(hiddenDim)
  for (let j = 0; j < hiddenDim; j++) {
    let acc = introModel.b_u[j] || 0
    for (let i = 0; i < inU.length; i++) acc += inU[i] * introModel.W_u[i][j]
    const u = Math.tanh(acc)
    zNext[j] = g[j] * zState[j] + (1 - g[j]) * u
  }

  const piLogits = new Array(NUM_ACTIONS)
  for (let a = 0; a < NUM_ACTIONS; a++) {
    let acc = introModel.b_pi[a] || 0
    for (let j = 0; j < hiddenDim; j++) acc += zNext[j] * introModel.W_pi[j][a]
    piLogits[a] = acc
  }
  const piProbs = softmaxArray(piLogits)
  let bestActionIdx = 0
  for (let a = 1; a < NUM_ACTIONS; a++) {
    if (piProbs[a] > piProbs[bestActionIdx]) bestActionIdx = a
  }

  let qAcc = introModel.b_q?.[0] || 0
  for (let j = 0; j < hiddenDim; j++) qAcc += zNext[j] * introModel.w_q[j][0]
  const qHat = sigmoidScalar(qAcc)

  const numClasses = introModel.b_delta.length
  const delta = new Array(numClasses)
  for (let c = 0; c < numClasses; c++) {
    let acc = introModel.b_delta[c] || 0
    for (let j = 0; j < hiddenDim; j++) acc += zNext[j] * introModel.W_delta[j][c]
    delta[c] = acc
  }

  let gateMean = 0
  for (let j = 0; j < hiddenDim; j++) gateMean += g[j]
  gateMean /= hiddenDim

  return {
    zNext,
    piProbs,
    nextActionIdx: bestActionIdx,
    nextAction: INTROSPECTION_ACTIONS[bestActionIdx],
    nextActionProb: Number(piProbs[bestActionIdx].toFixed(3)),
    qHat: Number(qHat.toFixed(3)),
    gateRetention: Number(gateMean.toFixed(3)),
    delta
  }
}

export function computeShannonEntropyNormalized(probs = []) {
  if (!probs || probs.length <= 1) return 0
  let h = 0
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i]
    if (p > 1e-9) h -= p * Math.log(p)
  }
  return Math.min(1, h / Math.log(probs.length))
}
