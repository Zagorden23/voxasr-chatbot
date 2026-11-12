import test from 'node:test'
import assert from 'node:assert/strict'

function matchesFilter(doc, filter = {}) {
  for (const [k, cond] of Object.entries(filter)) {
    if (cond && typeof cond === 'object' && !Array.isArray(cond) && !(cond instanceof Date)) {
      if (cond.$type === 'string' && typeof doc[k] !== 'string') return false
      if ('$ne' in cond && doc[k] === cond.$ne) return false
    } else if (doc[k] !== cond) {
      return false
    }
  }
  return true
}

function createInMemoryMongoDb() {
  const store = new Map()
  const getDocs = (name) => {
    if (!store.has(name)) store.set(name, [])
    return store.get(name)
  }

  return {
    async command(cmd) {
      if (cmd?.ping) return { ok: 1 }
      return { ok: 1 }
    },
    collection(name) {
      const docs = getDocs(name)
      return {
        async createIndex() {
          return 'ok'
        },
        async countDocuments(filter = {}) {
          return docs.filter((d) => matchesFilter(d, filter)).length
        },
        async findOne(filter = {}, opts = {}) {
          let list = docs.filter((d) => matchesFilter(d, filter))
          if (opts.sort) {
            const [[k, dir]] = Object.entries(opts.sort)
            list = [...list].sort((a, b) => (a[k] < b[k] ? -dir : a[k] > b[k] ? dir : 0))
          }
          return list[0] ? structuredClone(list[0]) : null
        },
        find(filter = {}) {
          let list = docs.filter((d) => matchesFilter(d, filter)).map((d) => structuredClone(d))
          return {
            sort(s = {}) {
              const [[k, dir]] = Object.entries(s)
              list.sort((a, b) => (a[k] < b[k] ? -dir : a[k] > b[k] ? dir : 0))
              return this
            },
            limit(n) {
              list = list.slice(0, n)
              return this
            },
            async toArray() {
              return list
            }
          }
        },
        async findOneAndUpdate(filter, update, opts = {}) {
          let target = docs.find((d) => matchesFilter(d, filter))
          if (!target && opts.upsert) {
            target = { ...filter, seq: 0 }
            docs.push(target)
          }
          if (!target) return null
          if (update.$inc) {
            for (const [k, v] of Object.entries(update.$inc)) {
              target[k] = Number(target[k] || 0) + v
            }
          }
          return structuredClone(target)
        },
        async insertOne(doc) {
          docs.push(structuredClone(doc))
          return { insertedId: doc.id ?? docs.length }
        },
        async insertMany(arr) {
          for (const d of arr) docs.push(structuredClone(d))
          return { insertedCount: arr.length }
        },
        async updateOne(filter, update, opts = {}) {
          let target = docs.find((d) => matchesFilter(d, filter))
          if (!target) {
            if (!opts.upsert) return { matchedCount: 0, modifiedCount: 0 }
            target = { ...(update.$setOnInsert || {}), ...(update.$set || {}) }
            docs.push(target)
            return { matchedCount: 0, modifiedCount: 0, upsertedCount: 1 }
          }
          if (update.$set) Object.assign(target, update.$set)
          return { matchedCount: 1, modifiedCount: 1 }
        },
        async deleteOne(filter) {
          const idx = docs.findIndex((d) => matchesFilter(d, filter))
          if (idx >= 0) {
            docs.splice(idx, 1)
            return { deletedCount: 1 }
          }
          return { deletedCount: 0 }
        },
        async deleteMany(filter = {}) {
          const before = docs.length
          for (let i = docs.length - 1; i >= 0; i--) {
            if (matchesFilter(docs[i], filter)) docs.splice(i, 1)
          }
          return { deletedCount: before - docs.length }
        },
        aggregate() {
          return {
            async toArray() {
              if (name === 'chat_sessions') {
                const msgs = getDocs('chat_messages')
                return [...docs]
                  .sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at))
                  .map((s) => ({
                    ...s,
                    message_count: msgs.filter((m) => m.session_id === s.id).length
                  }))
              }
              if (name === 'chat_messages') {
                const userMsgs = docs.filter((m) => m.role === 'user')
                const botMsgs = docs.filter((m) => m.role === 'bot')
                if (userMsgs.length === 0 && botMsgs.length === 0) return []
                const confs = docs.map((m) => m.asr_confidence).filter((v) => typeof v === 'number')
                const lats = docs.map((m) => m.latency_ms).filter((v) => typeof v === 'number')
                return [
                  {
                    total_queries: userMsgs.length,
                    voice_queries: userMsgs.filter((m) => m.input_mode === 'voice').length,
                    text_queries: userMsgs.filter((m) => m.input_mode === 'text').length,
                    avg_asr_confidence: confs.length ? confs.reduce((a, b) => a + b, 0) / confs.length : null,
                    avg_latency_ms: lats.length ? lats.reduce((a, b) => a + b, 0) / lats.length : null,
                    positive: botMsgs.filter((m) => m.feedback === 'up').length,
                    negative: botMsgs.filter((m) => m.feedback === 'down').length
                  }
                ]
              }
              if (name === 'wer_evaluations') {
                if (docs.length === 0) return []
                return [
                  {
                    total_wer_runs: docs.length,
                    avg_wer: docs.reduce((a, b) => a + b.wer, 0) / docs.length,
                    avg_word_accuracy: docs.reduce((a, b) => a + b.word_accuracy, 0) / docs.length
                  }
                ]
              }
              return []
            }
          }
        }
      }
    }
  }
}

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
  await initDatabase({ db: createInMemoryMongoDb() })
  await initKnowledgeEngine(reloadModel)
})

test.after(async () => {
  await closeDatabase()
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
  const healthRes = await app.request('/health')
  assert.equal(healthRes.status, 200)
  assert.equal(healthRes.headers.get('X-Content-Type-Options'), 'nosniff')
  assert.equal(healthRes.headers.get('Permissions-Policy'), 'microphone=(self)')
  const healthJson = await healthRes.json()
  assert.equal(healthJson.ok, true)
  assert.equal(healthJson.status, 'healthy')
  assert.equal(healthJson.database, 'mongodb')

  const langRes = await app.request('/api/languages')
  assert.equal(langRes.status, 200)
  assert.equal((await langRes.json()).total, 35)

  const modelRes = await app.request('/api/model')
  assert.equal(modelRes.status, 200)
  assert.equal((await modelRes.json()).ok, true)

  const kbSearchRes = await app.request('/api/knowledge/search?q=Dijkstra')
  assert.equal(kbSearchRes.status, 200)
  assert.equal((await kbSearchRes.json()).ok, true)

  const badChat = await app.request('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: '   ' })
  })
  assert.equal(badChat.status, 400)

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

  const fbRes = await app.request('/api/chat/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messageId: chatJson.botMessageId, feedback: 'up' })
  })
  assert.equal(fbRes.status, 200)
  assert.equal((await fbRes.json()).ok, true)

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
  assert.match(await exportRes.text(), /Dijkstra Study Session/)

  const werRes = await app.request('/api/wer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      reference: 'hello world test',
      hypothesis: 'hello word test'
    })
  })
  assert.equal(werRes.status, 200)
  assert.equal((await werRes.json()).metrics.substitutions, 1)

  const werHist = await app.request('/api/wer/history')
  assert.ok((await werHist.json()).history.length >= 1)

  const analyticsRes = await app.request('/api/analytics')
  const analyticsJson = await analyticsRes.json()
  assert.equal(analyticsJson.ok, true)
  assert.equal(analyticsJson.analytics.databaseBackend, 'mongodb')
  assert.ok(analyticsJson.analytics.totalSessions >= 1)
  assert.equal(analyticsJson.analytics.feedback.positive, 1)

  const delRes = await app.request(`/api/sessions/${sessId}`, { method: 'DELETE' })
  assert.equal((await delRes.json()).ok, true)

  const favRes = await app.request('/favicon.ico')
  assert.equal(favRes.status, 200)
  const rootRes = await app.request('/')
  assert.equal(rootRes.status, 200)
  assert.match(await rootRes.text(), /VoxASR/)
})
