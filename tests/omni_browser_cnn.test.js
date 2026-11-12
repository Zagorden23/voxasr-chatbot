import assert from 'node:assert/strict'
import test from 'node:test'

import { analyzeQuery } from '../src/knowledge.js'
import {
  extractRankedSentences,
  sentenceJaccard,
  synthesizeMultiSourceAnswer
} from '../src/query_analyzer.js'
import {
  buildReactionVector,
  stepRecursiveIntrospection
} from '../src/neural_introspection.js'
import {
  browseWebForQuery,
  extractReadableHtml,
  isPrivateIpAddress,
  isSafePublicUrl,
  parseDuckDuckGoHtml,
  scrapeExternalWebPage
} from '../src/web_browser.js'
import {
  generateIntelligentReply,
  getModelSummary,
  predictIntent
} from '../src/inference.js'

test('TextCNN model classifies 28 omni-world domains with 96-dim hidden state & introspection metadata', () => {
  const summary = getModelSummary()
  assert.equal(summary.num_classes, 28)
  assert.ok(summary.cnn_architecture)
  assert.deepEqual(summary.cnn_architecture.kernels, [2, 3, 4])
  assert.equal(summary.cnn_architecture.hidden_dim, 96)
  assert.ok(summary.introspection_architecture)
  assert.equal(summary.introspection_architecture.steps, 3)

  const domainSamples = [
    ['How do black holes and general relativity warp spacetime?', 'physics_astronomy_earth'],
    ['Explain DNA replication and enzyme catalysis', 'chemistry_biology_medicine'],
    ['How do brushless DC motors and control systems work?', 'engineering_robotics_energy'],
    ['Explain Keynesian macroeconomics and compound interest', 'commerce_finance_economics'],
    ['Renaissance oil painting perspective and Greek architecture', 'arts_literature_design'],
    ['Explain musical harmony, chord progressions, and Hindustani classical raga', 'music_audio_theory'],
    ['Classical ballet positions and Kathak footwork choreography', 'dance_performing_arts'],
    ['Explain the cricket LBW law and football offside trap', 'sports_athletics_games'],
    ['Three-act screenplay structure and cinematography lighting', 'entertainment_cinema_gaming'],
    ['Magna Carta constitutional law and ancient civilizations', 'history_geography_law_society'],
    ['What is Singular Value Decomposition?', 'numerical_analysis']
  ]

  for (const [query, expected] of domainSamples) {
    const pred = predictIntent(query)
    assert.equal(pred.intentTag, expected, `Expected ${expected} for "${query}", got ${pred.intentTag}`)
    assert.equal(pred.hiddenState?.length, 96)
    assert.equal(pred.cnnLogits?.length, 28)
  }
})

test('Query analyzer detects brief/standard/detailed depth, question kinds, and cleans search topics', () => {
  const brief = analyzeQuery('Explain photosynthesis in brief please')
  assert.equal(brief.depthMode, 'brief')
  assert.ok(!brief.searchQuery.toLowerCase().includes('in brief'))
  assert.ok(brief.searchQuery.toLowerCase().includes('photosynthesis'))

  const detailed = analyzeQuery('Explain photosynthesis light and dark reactions in detail')
  assert.equal(detailed.depthMode, 'detailed')
  assert.ok(!detailed.searchQuery.toLowerCase().includes('in detail'))

  const cmp = analyzeQuery('Compare cricket LBW rule and football offside law')
  assert.equal(cmp.depthMode, 'detailed')
  assert.equal(cmp.questionKind, 'comparison')
  assert.equal(cmp.comparison?.length, 2)

  const urlQ = analyzeQuery('Summarize https://example.org/science/fusion for me')
  assert.equal(urlQ.questionKind, 'url_browse')
  assert.deepEqual(urlQ.directUrls, ['https://example.org/science/fusion'])
})

test('Cross-source sentence ranker strips boilerplate, deduplicates overlaps, and scales length by depthMode', () => {
  const noisyText = [
    'Accept all cookies and sign in to continue reading.',
    'Quantum entanglement links particles across distance.',
    'Particles remain correlated in quantum entanglement across arbitrary distances.',
    'Measuring one entangled qubit instantaneously determines its partner state.',
    'Experiments verify Bell inequality violations with superconducting circuits and photons.'
  ].join(' ')

  const ranked = extractRankedSentences(noisyText, ['quantum', 'entanglement'], { depthMode: 'detailed' }, 5, 900)
  assert.ok(ranked.every((s) => !/cookies|sign in/i.test(s)))
  // Overlap sentence should be collapsed
  assert.ok(sentenceJaccard(ranked[0], 'Particles remain correlated in quantum entanglement across arbitrary distances.') > 0.3)

  const docs = [
    {
      title: 'Photosynthesis Overview',
      source: 'britannica.com',
      url: 'https://britannica.com/science/photosynthesis',
      extract:
        'Photosynthesis is the biochemical process by which plants convert sunlight into chemical energy. Chloroplasts absorb red and blue photons to split water molecules and release oxygen. Light-independent Calvin cycle reactions fix carbon dioxide into glucose.'
    },
    {
      title: 'Cellular Solar Capture Research (2024)',
      source: 'OpenAlex Research',
      url: 'https://openalex.org/W123456',
      extract:
        'Photosystem II operates at an electrochemical potential capable of oxidizing water. Researchers benchmark artificial leaf catalysts against natural thylakoid membranes.'
    }
  ]

  const briefOut = synthesizeMultiSourceAnswer({
    docs,
    analysis: analyzeQuery('What is photosynthesis in brief?'),
    intentTitle: 'Chemistry, Biology & Medicine'
  })
  const detailOut = synthesizeMultiSourceAnswer({
    docs,
    analysis: analyzeQuery('Explain photosynthesis in detail'),
    intentTitle: 'Chemistry, Biology & Medicine',
    learnedCount: 2
  })

  assert.ok(briefOut.reply.length < detailOut.reply.length)
  assert.ok(detailOut.sources.some((s) => s.source === 'britannica.com'))
  assert.ok(detailOut.sources.some((s) => s.source === 'OpenAlex Research'))
})

test('SSRF validation blocks internal networks and readability extractor parses clean HTML & DuckDuckGo links', () => {
  assert.equal(isSafePublicUrl('https://www.britannica.com/science/atom').ok, true)
  assert.equal(isSafePublicUrl('http://127.0.0.1:8080/admin').ok, false)
  assert.equal(isSafePublicUrl('http://localhost:3000').ok, false)
  assert.equal(isSafePublicUrl('http://169.254.169.254/latest/meta-data/').ok, false)
  assert.equal(isSafePublicUrl('http://192.168.1.50/secret').ok, false)
  assert.equal(isSafePublicUrl('file:///etc/passwd').ok, false)
  assert.equal(isPrivateIpAddress('10.20.30.40'), true)
  assert.equal(isPrivateIpAddress('8.8.8.8'), false)

  const sampleHtml = `
    <html>
      <head>
        <title>Cricket Laws Explained</title>
        <meta name="description" content="Concise overview of MCC Cricket laws." />
        <script>console.log("ignore me");</script>
      </head>
      <body>
        <nav>Home | News | Sign in</nav>
        <main>
          <p>Leg before wicket (LBW) dismisses a batter when the ball strikes the body in line with the stumps.</p>
          <p>Umpires evaluate pitch location, impact line, and whether the ball would hit the wickets.</p>
        </main>
        <footer>Copyright 2026 Accept all cookies</footer>
      </body>
    </html>
  `
  const extracted = extractReadableHtml(sampleHtml, 'https://lords.org/laws', 'Cricket', ['cricket', 'lbw'])
  assert.equal(extracted.title, 'Cricket Laws Explained')
  assert.ok(extracted.extract.includes('Leg before wicket (LBW)'))
  assert.ok(!extracted.extract.includes('console.log'))
  assert.ok(!extracted.extract.includes('Sign in'))

  const ddgSample = `
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Flords.org%2Flbw-law">MCC Law 36 LBW</a>
    <a class="result__snippet" href="#">Official cricket LBW dismissal rules explained by MCC.</a>
  `
  const links = parseDuckDuckGoHtml(ddgSample)
  assert.equal(links.length, 1)
  assert.equal(links[0].url, 'https://lords.org/lbw-law')
  assert.equal(links[0].host, 'lords.org')
})

test('3-Step Recursive Action-Reaction Introspection produces calibrated trace end-to-end', async () => {
  const pred = predictIntent('Explain quantum entanglement in detail')
  const intro = getModelSummary()
  assert.ok(intro.introspection_architecture)

  const weakReaction = buildReactionVector({
    bm25TopScore: 1.5,
    spanCount: 0,
    tokenCoverage: 0.2,
    domainAgree: 0.3,
    clfTopProb: pred.confidence,
    clfEntropy: 0.5,
    passageChars: 80,
    depthMode: 'detailed',
    webSourceCount: 0
  })
  const strongReaction = buildReactionVector({
    bm25TopScore: 10.0,
    spanCount: 4,
    tokenCoverage: 0.9,
    domainAgree: 1.0,
    clfTopProb: pred.confidence,
    clfEntropy: 0.2,
    passageChars: 1800,
    depthMode: 'detailed',
    webSourceCount: 4
  })

  const model = getModelSummary()
  assert.ok(model.introspection_architecture)

  const predObj = predictIntent('Explain the rules of cricket in brief')

  // Step forward reacts sensibly when transitioning from weak local knowledge to rich multi-site results
  const modelReply = await generateIntelligentReply('Explain Bharatanatyam dance in brief', { skipLiveWeb: true })
  assert.equal(modelReply.depthMode, 'brief')
  assert.equal(modelReply.introspection?.steps?.length, 3)
  assert.ok(modelReply.introspection.finalSufficiencyQ >= 0.05)

  // Verify direct stepRecursiveIntrospection execution with real 96-dim CNN state
  const { default: fs } = await import('node:fs')
  const modelJson = JSON.parse(fs.readFileSync('D:/chatbot_ASR/webapp/src/model/chatbot_model.json', 'utf8'))
  const stepWeak = stepRecursiveIntrospection(modelJson.introspection, predObj.hiddenState, 2, weakReaction)
  const stepStrong = stepRecursiveIntrospection(modelJson.introspection, predObj.hiddenState, 3, strongReaction)
  assert.ok(stepWeak && stepStrong)
  assert.equal(stepWeak.zNext.length, 96)
  assert.ok(stepStrong.qHat > stepWeak.qHat, `Expected strong qHat (${stepStrong.qHat}) > weak qHat (${stepWeak.qHat})`)

  assert.equal(weakReaction.length, 12)
  assert.ok(strongReaction[10] > weakReaction[10])
})

test('Live web browser aborts cleanly on CAPTCHA/403 without bypass and aggregates non-Wikipedia sites', async () => {
  const realFetch = globalThis.fetch
  const blockedHtml = '<html><head><title>Attention Required! | Cloudflare</title></head><body><div class="cf-turnstile">Verify you are human</div></body></html>'
  const cleanExternalHtml = '<html><head><title>Solar Cells Explained</title></head><body><main><p>Perovskite solar cells convert sunlight directly into electrical current using hybrid organic-inorganic halides.</p><p>Tandem silicon-perovskite architectures surpass single-junction efficiency limits cleanly.</p></main></body></html>'

  globalThis.fetch = async (url) => {
    const u = String(url)
    if (u.includes('html.duckduckgo.com')) {
      return new Response(
        '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fcaptcha">Blocked Site</a>' +
          '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fsolar">Perovskite Guide</a>' +
          '<a class="result__snippet" href="#">Tandem solar cells reach high photovoltaic efficiency.</a>',
        { status: 200, headers: { 'content-type': 'text/html' } }
      )
    }
    if (u.includes('example.com/captcha')) {
      return new Response(blockedHtml, { status: 403, headers: { 'content-type': 'text/html' } })
    }
    if (u.includes('example.org/solar')) {
      return new Response(cleanExternalHtml, { status: 200, headers: { 'content-type': 'text/html' } })
    }
    if (u.includes('api.duckduckgo.com')) {
      return new Response(JSON.stringify({ AbstractText: '', RelatedTopics: [] }), {
        status: 200,
        headers: { 'content-type': 'application/json' }
      })
    }
    if (u.includes('api.openalex.org')) {
      return new Response(JSON.stringify({ results: [] }), { status: 200, headers: { 'content-type': 'application/json' } })
    }
    return new Response(JSON.stringify({ query: { search: [], pages: {} }, search: [] }), {
      status: 200,
      headers: { 'content-type': 'application/json' }
    })
  }

  try {
    const blocked = await scrapeExternalWebPage('https://example.com/captcha', 'Captcha Test', ['solar'])
    assert.equal(blocked?.blocked, true)

    const browseRes = await browseWebForQuery('perovskite solar cells', { depthMode: 'standard' }, 'engineering_robotics_energy')
    assert.ok(browseRes.blockedUrls.includes('https://example.com/captcha'))
    assert.ok(browseRes.docs.some((d) => d.source === 'example.org' && d.extract.includes('Perovskite solar cells')))
  } finally {
    globalThis.fetch = realFetch
  }
})
