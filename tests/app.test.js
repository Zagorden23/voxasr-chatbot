import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

// Use isolated temp directory for DB during tests
const tmpDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'voxasr-test-'))
process.env.DATA_DIR = tmpDataDir

const { closeDatabase, initDatabase } = await import('../src/db.js')
const { initKnowledgeEngine } = await import('../src/web_learner.js')
const {
  computeWERAndAlignment,
  generateIntelligentReply,
  predictIntent,
  reloadModel
} = await import('../src/inference.js')
const { detectLanguage, tokenize } = await import('../src/text.js')
const { default: app } = await import('../src/index.js')

test.before(async () => {
  await initDatabase()
  await initKnowledgeEngine(reloadModel)
})

test.after(async () => {
  await closeDatabase()
  fs.rmSync(tmpDataDir, { recursive: true, force: true })
})

test('detectLanguage detects scripts, apostrophes, and 35-language variants', () => {
  assert.equal(detectLanguage('नमस्ते चैटबॉट').lang, 'hi-IN')
  assert.equal(detectLanguage('什么是Transformer架构').lang, 'zh-CN')
  assert.equal(detectLanguage('特異値分解とは何ですか').lang, 'ja-JP')
  assert.equal(detectLanguage("Qu'est-ce que le théorème CAP?").lang, 'fr-FR')
  assert.equal(detectLanguage('Ce este un arbore binar și cum funcționează?').lang, 'ro-RO')
  assert.equal(detectLanguage('مرحبا كيف يعمل').lang, 'ar-SA')
})

test('tokenize removes stopwords, stems plurals, and segments CJK', () => {
  const en = tokenize('What are binary trees and graphs?', 'en')
  assert.deepEqual(en, ['binary', 'tree', 'graph'])
  const zh = tokenize('神经网络', 'zh')
  assert.ok(zh.includes('神经'))
  assert.ok(zh.includes('经网'))
})

test('computeWERAndAlignment computes exact WER, CER, and edit ops', () => {
  const exact = computeWERAndAlignment('hello world', 'hello world')
  assert.equal(exact.wer, 0)
  assert.equal(exact.cer, 0)
  assert.equal(exact.wordAccuracy, 100)

  const diff = computeWERAndAlignment('the quick brown fox', 'the fast brown')
  assert.equal(diff.substitutions, 1)
  assert.equal(diff.deletions, 1)
  assert.equal(diff.insertions, 0)
  assert.equal(diff.wer, 50)
})

test('Safe math evaluator solves expressions and rejects code injection', async () => {
  const res = await generateIntelligentReply('Calculate 2^10 + sqrt(144)', { autoLearn: false })
  assert.equal(res.intentTag, 'numerical_analysis')
  assert.match(res.reply, /1036/)

  const divZero = await generateIntelligentReply('10 / 0', { autoLearn: false })
  assert.match(divZero.reply, /Division by zero/)

  const unsafe = await generateIntelligentReply('process.exit(1)', { autoLearn: false })
  assert.notEqual(unsafe.intentTitle, 'Mathematical Expression Evaluator')
})

test('predictIntent classifies CS & Math queries accurately', () => {
  const svd = predictIntent('What is Singular Value Decomposition (SVD)?')
  assert.equal(svd.intentTag, 'numerical_analysis')
  assert.ok(svd.confidence > 0.5)

  const dijkstra = predictIntent('Explain Dijkstra shortest path algorithm')
  assert.equal(dijkstra.intentTag, 'algorithms_complexity')
})

test('HTTP API endpoints: health, languages, model, knowledge, chat, sessions, wer, analytics', async () => {
  // Health
  const healthRes = await app.request('/health')
  assert.equal(healthRes.status, 200)
  assert.equal(healthRes.headers.get('X-Content-Type-Options'), 'nosniff')
  assert.equal(healthRes.headers.get('Permissions-Policy'), 'microphone=(self)')
  const healthJson = await healthRes.json()
  assert.equal(healthJson.ok, true)
  assert.equal(healthJson.status, 'healthy')

  // 35 Certified Languages
  const langRes = await app.request('/api/languages')
  assert.equal(langRes.status, 200)
  const langJson = await langRes.json()
  assert.equal(langJson.total, 35)

  // Model & Knowledge Status & Search
  const modelRes = await app.request('/api/model')
  assert.equal(modelRes.status, 200)
  const modelJson = await modelRes.json()
  assert.equal(modelJson.ok, true)

  const kbSearchRes = await app.request('/api/knowledge/search?q=Dijkstra')
  assert.equal(kbSearchRes.status, 200)
  const kbSearchJson = await kbSearchRes.json()
  assert.equal(kbSearchJson.ok, true)

  // Chat validation
  const badChat = await app.request('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: '   ' })
  })
  assert.equal(badChat.status, 400)

  // Valid Chat + Session Persistence + Feedback + Export + Rename + Delete
  const sessId = 'sess_test_001'
  const chatRes = await app.request('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      sessionId: sessId,
      message: 'What is Dijkstra algorithm?',
      inputMode: 'voice',
      asrConfidence: 0.96,
      asrLang: 'en-US',
      autoLearn: false
    })
  })
  assert.equal(chatRes.status, 200)
  const chatJson = await chatRes.json()
  assert.equal(chatJson.ok, true)
  assert.ok(chatJson.botResponse.reply.length > 20)
  assert.ok(chatJson.botMessageId)

  // Feedback
  const fbRes = await app.request('/api/chat/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messageId: chatJson.botMessageId, feedback: 'up' })
  })
  assert.equal(fbRes.status, 200)
  assert.equal((await fbRes.json()).ok, true)

  // Sessions list, get, rename, export, delete
  const listRes = await app.request('/api/sessions')
  const listJson = await listRes.json()
  assert.ok(listJson.sessions.some((s) => s.id === sessId))

  const renameRes = await app.request(`/api/sessions/${sessId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: 'Dijkstra Study Session' })
  })
  assert.equal((await renameRes.json()).ok, true)

  const exportRes = await app.request(`/api/sessions/${sessId}/export?format=markdown`)
  assert.equal(exportRes.status, 200)
  const mdText = await exportRes.text()
  assert.match(mdText, /Dijkstra Study Session/)

  // WER API
  const werRes = await app.request('/api/wer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      reference: 'hello world test',
      hypothesis: 'hello word test'
    })
  })
  assert.equal(werRes.status, 200)
  const werJson = await werRes.json()
  assert.equal(werJson.metrics.substitutions, 1)

  const werHist = await app.request('/api/wer/history')
  assert.ok((await werHist.json()).history.length >= 1)

  // Analytics
  const analyticsRes = await app.request('/api/analytics')
  const analyticsJson = await analyticsRes.json()
  assert.equal(analyticsJson.ok, true)
  assert.ok(analyticsJson.analytics.totalSessions >= 1)
  assert.equal(analyticsJson.analytics.feedback.positive, 1)

  // Delete session
  const delRes = await app.request(`/api/sessions/${sessId}`, { method: 'DELETE' })
  assert.equal((await delRes.json()).ok, true)

  // Favicon & SPA Shell
  const favRes = await app.request('/favicon.ico')
  assert.equal(favRes.status, 200)
  const rootRes = await app.request('/')
  assert.equal(rootRes.status, 200)
  assert.match(await rootRes.text(), /VoxASR/)
})
