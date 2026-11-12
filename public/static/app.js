/**
 * VoxASR Commercial Frontend SPA — Pure JavaScript + React 18
 * Features:
 *  - 35 Certified World Languages (Web Speech API STT + SpeechSynthesis TTS + RTL support)
 *  - Persistent Multi-Session History, Rename, Delete & Markdown/JSON Transcript Export
 *  - Hybrid Softmax + BM25 Extractive QA + Live Web Self-Learning + Math Evaluator
 *  - Per-Message Copy, Read Aloud (TTS), Thumbs Up/Down Feedback & Source Citations
 *  - Web Audio API (AnalyserNode) Live Microphone Frequency Spectrum Visualizer
 *  - Interactive Knowledge Search Explorer & One-Click Live Web Scraping
 *  - WER / CER Evaluator Lab with Live Microphone Dictation & History Log
 *  - Commercial Telemetry & Model Inspector Dashboard
 */

;(function () {
  const e = React.createElement
  const { useState, useEffect, useRef } = React

  const DEFAULT_LANGUAGES = [
    { code: 'en-US', iso: 'en', name: 'English (US)', nativeName: 'English (US)', flag: '🇺🇸', script: 'Latin', rtl: false },
    { code: 'en-GB', iso: 'en', name: 'English (UK)', nativeName: 'English (UK)', flag: '🇬🇧', script: 'Latin', rtl: false },
    { code: 'en-IN', iso: 'en', name: 'English (India)', nativeName: 'English (India)', flag: '🇮🇳', script: 'Latin', rtl: false },
    { code: 'es-ES', iso: 'es', name: 'Spanish (Spain)', nativeName: 'Español (España)', flag: '🇪🇸', script: 'Latin', rtl: false },
    { code: 'es-MX', iso: 'es', name: 'Spanish (Mexico)', nativeName: 'Español (México)', flag: '🇲🇽', script: 'Latin', rtl: false },
    { code: 'zh-CN', iso: 'zh', name: 'Chinese (Mandarin)', nativeName: '简体中文', flag: '🇨🇳', script: 'Hanzi', rtl: false },
    { code: 'zh-TW', iso: 'zh', name: 'Chinese (Traditional)', nativeName: '繁體中文', flag: '🇹🇼', script: 'Hanzi', rtl: false },
    { code: 'hi-IN', iso: 'hi', name: 'Hindi', nativeName: 'हिन्दी', flag: '🇮🇳', script: 'Devanagari', rtl: false },
    { code: 'ar-SA', iso: 'ar', name: 'Arabic', nativeName: 'العربية', flag: '🇸🇦', script: 'Arabic', rtl: true },
    { code: 'fr-FR', iso: 'fr', name: 'French (France)', nativeName: 'Français', flag: '🇫🇷', script: 'Latin', rtl: false },
    { code: 'ru-RU', iso: 'ru', name: 'Russian', nativeName: 'Русский', flag: '🇷🇺', script: 'Cyrillic', rtl: false },
    { code: 'pt-BR', iso: 'pt', name: 'Portuguese (Brazil)', nativeName: 'Português (Brasil)', flag: '🇧🇷', script: 'Latin', rtl: false },
    { code: 'de-DE', iso: 'de', name: 'German', nativeName: 'Deutsch', flag: '🇩🇪', script: 'Latin', rtl: false },
    { code: 'ja-JP', iso: 'ja', name: 'Japanese', nativeName: '日本語', flag: '🇯🇵', script: 'Kanji/Kana', rtl: false },
    { code: 'ko-KR', iso: 'ko', name: 'Korean', nativeName: '한국어', flag: '🇰🇷', script: 'Hangul', rtl: false },
    { code: 'it-IT', iso: 'it', name: 'Italian', nativeName: 'Italiano', flag: '🇮🇹', script: 'Latin', rtl: false },
    { code: 'bn-IN', iso: 'bn', name: 'Bengali', nativeName: 'বাংলা', flag: '🇧🇩', script: 'Bengali', rtl: false },
    { code: 'tr-TR', iso: 'tr', name: 'Turkish', nativeName: 'Türkçe', flag: '🇹🇷', script: 'Latin', rtl: false },
    { code: 'vi-VN', iso: 'vi', name: 'Vietnamese', nativeName: 'Tiếng Việt', flag: '🇻🇳', script: 'Latin', rtl: false },
    { code: 'id-ID', iso: 'id', name: 'Indonesian', nativeName: 'Bahasa Indonesia', flag: '🇮🇩', script: 'Latin', rtl: false },
    { code: 'nl-NL', iso: 'nl', name: 'Dutch', nativeName: 'Nederlands', flag: '🇳🇱', script: 'Latin', rtl: false },
    { code: 'pl-PL', iso: 'pl', name: 'Polish', nativeName: 'Polski', flag: '🇵🇱', script: 'Latin', rtl: false },
    { code: 'fa-IR', iso: 'fa', name: 'Persian', nativeName: 'فارسی', flag: '🇮🇷', script: 'Arabic', rtl: true },
    { code: 'sw-KE', iso: 'sw', name: 'Swahili', nativeName: 'Kiswahili', flag: '🇰🇪', script: 'Latin', rtl: false },
    { code: 'ta-IN', iso: 'ta', name: 'Tamil', nativeName: 'தமிழ்', flag: '🇮🇳', script: 'Tamil', rtl: false },
    { code: 'te-IN', iso: 'te', name: 'Telugu', nativeName: 'తెలుగు', flag: '🇮🇳', script: 'Telugu', rtl: false },
    { code: 'ur-PK', iso: 'ur', name: 'Urdu', nativeName: 'اردو', flag: '🇵🇰', script: 'Arabic', rtl: true },
    { code: 'mr-IN', iso: 'mr', name: 'Marathi', nativeName: 'मराठी', flag: '🇮🇳', script: 'Devanagari', rtl: false },
    { code: 'uk-UA', iso: 'uk', name: 'Ukrainian', nativeName: 'Українська', flag: '🇺🇦', script: 'Cyrillic', rtl: false },
    { code: 'el-GR', iso: 'el', name: 'Greek', nativeName: 'Ελληνικά', flag: '🇬🇷', script: 'Greek', rtl: false },
    { code: 'sv-SE', iso: 'sv', name: 'Swedish', nativeName: 'Svenska', flag: '🇸🇪', script: 'Latin', rtl: false },
    { code: 'he-IL', iso: 'he', name: 'Hebrew', nativeName: 'עברית', flag: '🇮🇱', script: 'Hebrew', rtl: true },
    { code: 'th-TH', iso: 'th', name: 'Thai', nativeName: 'ไทย', flag: '🇹🇭', script: 'Thai', rtl: false },
    { code: 'cs-CZ', iso: 'cs', name: 'Czech', nativeName: 'Čeština', flag: '🇨🇿', script: 'Latin', rtl: false },
    { code: 'ro-RO', iso: 'ro', name: 'Romanian', nativeName: 'Română', flag: '🇷🇴', script: 'Latin', rtl: false }
  ]

  const CS_MATH_PILLARS = [
    {
      id: 'algorithms_complexity',
      title: 'Algorithms & Complexity',
      icon: 'fa-sitemap',
      desc: 'Dijkstra, A*, Dynamic Programming, NP-completeness, Big-O, Master Theorem',
      query: 'Explain Dijkstra algorithm'
    },
    {
      id: 'numerical_analysis',
      title: 'Numerical Computing & SVD',
      icon: 'fa-square-root-variable',
      desc: 'Singular Value Decomposition (SVD), QR, LU, Runge-Kutta RK4, FFT, IEEE 754',
      query: 'What is Singular Value Decomposition?'
    },
    {
      id: 'machine_learning_dl',
      title: 'Deep Learning & Transformers',
      icon: 'fa-network-wired',
      desc: 'Transformers, Attention, Backpropagation, Diffusion Models, RLHF',
      query: 'Explain Transformer deep learning architecture'
    },
    {
      id: 'speech_ai_asr',
      title: 'Automatic Speech Recognition (ASR)',
      icon: 'fa-microphone-lines',
      desc: 'Whisper, CTC Loss, Wav2Vec, Mel Cepstrum, Spectrograms, WER',
      query: 'How does speech recognition work?'
    },
    {
      id: 'theoretical_cs_discrete_math',
      title: 'Theoretical CS & Discrete Math',
      icon: 'fa-atom',
      desc: 'Turing Machines, Chomsky Hierarchy, Halting Problem, Graph Theory, Lambda Calculus',
      query: 'What is a Turing machine?'
    },
    {
      id: 'systems_distributed_computing',
      title: 'OS & Distributed Systems',
      icon: 'fa-server',
      desc: 'CAP Theorem, Raft Consensus, Virtual Memory, Consistent Hashing, Docker',
      query: 'Explain the CAP theorem'
    },
    {
      id: 'compilers_programming_languages',
      title: 'Compilers & Programming Languages',
      icon: 'fa-code',
      desc: 'AST, LLVM, SSA Form, Parsing, Type Inference, Garbage Collection',
      query: 'How does static single-assignment form work?'
    },
    {
      id: 'cryptography_security',
      title: 'Modern Cryptography & Security',
      icon: 'fa-shield-halved',
      desc: 'RSA, Elliptic Curve (ECC), AES, Zero-Knowledge Proofs, Post-Quantum',
      query: 'How does RSA cryptosystem work?'
    },
    {
      id: 'calculus_linear_algebra_prob',
      title: 'Calculus, Linear Algebra & Probability',
      icon: 'fa-chart-line',
      desc: 'Jacobian, Hessian, Eigenvalues, Bayes Theorem, Markov Chains, CLT',
      query: 'What are eigenvalues and eigenvectors?'
    },
    {
      id: 'gpu_hpc_computing',
      title: 'GPU Computing & HPC',
      icon: 'fa-microchip',
      desc: 'CUDA, MPI, OpenMP, BLAS, SIMD, Pipelining, Cache Architecture',
      query: 'What is CUDA?'
    },
    {
      id: 'database_vector_search',
      title: 'Database Internals & Vector Search',
      icon: 'fa-database',
      desc: 'B+ Trees, LSM Trees, ACID, MVCC, Nearest Neighbor Search, Cosine Similarity',
      query: 'Compare B+ tree and LSM tree'
    },
    {
      id: 'nlp_large_language_models',
      title: 'NLP & Large Language Models',
      icon: 'fa-robot',
      desc: 'BPE Tokenization, RAG, BM25, Word Embeddings, BERT, Fine-Tuning',
      query: 'What is retrieval-augmented generation?'
    },
    {
      id: 'physics_astronomy_earth',
      title: 'Physics, Space & Earth',
      icon: 'fa-meteor',
      desc: 'Relativity, Quantum Mechanics, Exoplanets, Thermodynamics, Optics',
      query: 'Explain quantum entanglement in brief'
    },
    {
      id: 'chemistry_biology_medicine',
      title: 'Chemistry, Biology & Medicine',
      icon: 'fa-dna',
      desc: 'CRISPR, Genomics, Organic Chemistry, Neuroscience, Pharmacology',
      query: 'How does mRNA vaccine work in detail?'
    },
    {
      id: 'commerce_finance_economics',
      title: 'Commerce, Finance & Economics',
      icon: 'fa-coins',
      desc: 'Macroeconomics, Portfolio Theory, Accounting, Trade, Supply & Demand',
      query: 'Compare Keynesian and Classical economics'
    },
    {
      id: 'music_audio_theory',
      title: 'Music & Acoustics Theory',
      icon: 'fa-music',
      desc: 'Harmony, Counterpoint, Hindustani Ragas, Jazz Improvisation, Sound Design',
      query: 'Explain Hindustani classical raga music theory'
    },
    {
      id: 'dance_performing_arts',
      title: 'Dance & Performing Arts',
      icon: 'fa-person-walking',
      desc: 'Bharatanatyam, Kathak, Classical Ballet, Hip-Hop Choreography, Theatre',
      query: 'Compare Bharatanatyam and Ballet techniques in detail'
    },
    {
      id: 'sports_athletics_games',
      title: 'Sports, Athletics & Strategy',
      icon: 'fa-trophy',
      desc: 'Cricket Laws, Football Tactics, Tennis, Chess Strategy, Olympic Athletics',
      query: 'Explain the cricket LBW rule and swing bowling physics'
    }
  ]

  const MULTILINGUAL_SHOWCASE = [
    { flag: '🌍', lang: 'Detail Mode', text: 'Explain photosynthesis and cellular respiration in detail' },
    { flag: '⚡', lang: 'Brief Mode', text: 'What is CRISPR gene editing in brief?' },
    { flag: '⚽', lang: 'Sports', text: 'Explain the offside rule in football with examples' },
    { flag: '🎵', lang: 'Music', text: 'How do chord progressions work in jazz music?' },
    { flag: '💃', lang: 'Dance', text: 'Tell me about Kathak classical dance history' },
    { flag: '🇪🇸', lang: 'Spanish', text: '¿Cómo funciona la retropropagación en redes neuronales?' },
    { flag: '🇨🇳', lang: 'Mandarin', text: '什么是Transformer架构和自注意力机制？' },
    { flag: '🇮🇳', lang: 'Hindi', text: 'डाइक्स्ट्रा एल्गोरिदम और टाइम कॉम्प्लेक्सिटी क्या है?' }
  ]

  const PRESET_WER_EXAMPLES = [
    {
      label: 'Homophone Error ("recognize speech")',
      ref: 'how to recognize speech with an acoustic and language model',
      hyp: 'how to wreck a nice beach with an acoustic model'
    },
    {
      label: 'Whisper Clean Audiobook Sample',
      ref: 'automatic speech recognition converts spoken audio waveforms into text tokens',
      hyp: 'automatic speech recognition converts spoken audio wave forms into text tokens'
    },
    {
      label: 'Mathematical Computing ASR Sample',
      ref: 'singular value decomposition factorizes a matrix into orthogonal singular vectors',
      hyp: 'single value decomposition factor rises a matrix into orthogonal single vectors'
    }
  ]

  const WELCOME_MESSAGE = {
    id: 'welcome',
    role: 'bot',
    content:
      'Welcome to **VoxASR** — Commercial Multilingual Voice & Text AI Assistant for **Computer Science** and **Mathematical Computing**!\n\n- **35 Certified World Languages**: Speak or type in any language (English, Spanish, Mandarin, Hindi, Arabic, French, German, Japanese, Russian, Romanian, etc.) with automatic RTL support.\n- **Hybrid Extractive + Self-Learning Engine**: Answers directly from our indexed CS/Math knowledge pool, evaluates math expressions (`sqrt(144) + 2^10`), compares algorithms (`Compare Dijkstra and A*`), and learns unknown topics live from Wikipedia & ArXiv!\n- **WER Benchmarking Lab**: Measure Speech-to-Text accuracy live with your microphone.',
    intentTag: 'greeting',
    intentTitle: 'Multilingual CS/Math Assistant Ready',
    confidence: 1.0,
    detectedLanguage: { lang: 'en-US', label: 'English', script: 'Latin' },
    followUps: [
      'What is Singular Value Decomposition?',
      'Compare B+ tree and LSM tree',
      'Calculate 2^10 + sqrt(144)',
      'What languages do you understand?'
    ]
  }

  function renderInline(line, keyPrefix) {
    const parts = String(line)
      .split(/(\*\*[^*]+\*\*|`[^`]+`|\$[^$]+\$|_[^_]+_|https?:\/\/[^\s)]+)/g)
      .filter(Boolean)

    return parts.map((part, idx) => {
      const k = `${keyPrefix}-${idx}`
      if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
        return e('strong', { key: k, className: 'font-semibold text-indigo-200' }, part.slice(2, -2))
      }
      if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
        return e(
          'code',
          {
            key: k,
            className: 'font-mono text-xs bg-slate-800/90 text-emerald-300 px-1.5 py-0.5 rounded border border-slate-700'
          },
          part.slice(1, -1)
        )
      }
      if (part.startsWith('$') && part.endsWith('$') && part.length > 2) {
        return e(
          'span',
          { key: k, className: 'font-mono text-xs text-amber-300 bg-amber-950/40 px-1.5 py-0.5 rounded border border-amber-500/20' },
          part.slice(1, -1)
        )
      }
      if (part.startsWith('_') && part.endsWith('_') && part.length > 2) {
        return e('em', { key: k, className: 'italic text-slate-300' }, part.slice(1, -1))
      }
      if (/^https?:\/\//.test(part)) {
        return e(
          'a',
          {
            key: k,
            href: part,
            target: '_blank',
            rel: 'noopener noreferrer',
            className: 'text-indigo-400 hover:text-indigo-300 underline break-all'
          },
          part
        )
      }
      return part
    })
  }

  function renderFormattedText(text) {
    if (!text) return null
    const segments = String(text).split(/(```[\s\S]*?```)/g)
    const blocks = []

    segments.forEach((seg, segIdx) => {
      if (seg.startsWith('```') && seg.endsWith('```')) {
        const inner = seg.slice(3, -3)
        const firstNewline = inner.indexOf('\n')
        const lang = firstNewline > -1 ? inner.slice(0, firstNewline).trim() : ''
        const code = firstNewline > -1 ? inner.slice(firstNewline + 1) : inner
        blocks.push(
          e(
            'div',
            { key: `code-${segIdx}`, className: 'my-2 rounded-xl overflow-hidden border border-slate-700/80 bg-slate-950' },
            lang && e('div', { className: 'px-3 py-1 bg-slate-900 text-[10px] font-mono text-slate-400 border-b border-slate-800 uppercase' }, lang),
            e('pre', { className: 'p-3 text-xs font-mono text-emerald-300 overflow-x-auto leading-relaxed' }, code)
          )
        )
        return
      }

      const lines = seg.split('\n')
      lines.forEach((line, lineIdx) => {
        const k = `${segIdx}-${lineIdx}`
        if (!line.trim()) {
          blocks.push(e('div', { key: k, className: 'h-1.5' }))
          return
        }
        if (/^>\s*/.test(line)) {
          blocks.push(
            e(
              'blockquote',
              { key: k, className: 'border-l-2 border-indigo-500/60 pl-3 py-1 my-1.5 text-xs text-slate-300 bg-slate-950/50 rounded-r-lg' },
              renderInline(line.replace(/^>\s*/, ''), k)
            )
          )
          return
        }
        const isBullet = /^(\s*[-*]|\s*\d+\.)\s+/.test(line)
        const isHeader = /^#{1,4}\s+/.test(line)
        const cleanLine = isBullet
          ? line.replace(/^(\s*[-*]|\s*\d+\.)\s+/, '')
          : isHeader
          ? line.replace(/^#{1,4}\s+/, '')
          : line

        if (isHeader) {
          blocks.push(e('h4', { key: k, className: 'text-sm font-bold text-indigo-300 mt-2 mb-1' }, renderInline(cleanLine, k)))
          return
        }
        if (isBullet) {
          blocks.push(
            e(
              'div',
              { key: k, className: 'flex items-start gap-2 pl-1 my-1 text-slate-200 text-xs sm:text-sm' },
              e('span', { className: 'text-indigo-400 mt-1 text-[10px]' }, '●'),
              e('span', { className: 'leading-relaxed' }, renderInline(cleanLine, k))
            )
          )
          return
        }
        blocks.push(e('p', { key: k, className: 'leading-relaxed my-0.5 text-slate-200 text-xs sm:text-sm' }, renderInline(cleanLine, k)))
      })
    })

    return blocks
  }

  function App() {
    const [activeTab, setActiveTab] = useState('chat') // 'chat' | 'knowledge' | 'wer' | 'model'
    const [languages, setLanguages] = useState(DEFAULT_LANGUAGES)
    const [sessionId, setSessionId] = useState(() => `sess_${Date.now().toString(36)}`)
    const [sessions, setSessions] = useState([])
    const [sessionSearch, setSessionSearch] = useState('')
    const [messages, setMessages] = useState([WELCOME_MESSAGE])

    const [inputText, setInputText] = useState('')
    const [interimTranscript, setInterimTranscript] = useState('')
    const [isListening, setIsListening] = useState(false)
    const [asrLang, setAsrLang] = useState('en-US')
    const [autoSendVoice, setAutoSendVoice] = useState(true)
    const [ttsEnabled, setTtsEnabled] = useState(false)
    const [autoLearn, setAutoLearn] = useState(true)
    const [speechSupported, setSpeechSupported] = useState(true)
    const [micError, setMicError] = useState('')
    const [isSending, setIsSending] = useState(false)
    const [lastAsrConfidence, setLastAsrConfidence] = useState(null)
    const [copiedId, setCopiedId] = useState(null)
    const [expandedDetails, setExpandedDetails] = useState({})

    const [audioBars, setAudioBars] = useState(new Array(20).fill(6))

    const [modelInfo, setModelInfo] = useState(null)
    const [knowledgeStatus, setKnowledgeStatus] = useState(null)
    const [analytics, setAnalytics] = useState(null)
    const [isUpgrading, setIsUpgrading] = useState(false)
    const [upgradeMessage, setUpgradeMessage] = useState('')
    const [customScrapeTopic, setCustomScrapeTopic] = useState('')
    const [customScrapeLang, setCustomScrapeLang] = useState('en')
    const [kbSearchQuery, setKbSearchQuery] = useState('')
    const [kbSearchResults, setKbSearchResults] = useState([])
    const [kbSearching, setKbSearching] = useState(false)

    // WER Lab state
    const [werRef, setWerRef] = useState(PRESET_WER_EXAMPLES[0].ref)
    const [werHyp, setWerHyp] = useState(PRESET_WER_EXAMPLES[0].hyp)
    const [werResult, setWerResult] = useState(null)
    const [werHistory, setWerHistory] = useState([])
    const [isListeningWer, setIsListeningWer] = useState(false)

    const recognitionRef = useRef(null)
    const werRecognitionRef = useRef(null)
    const audioCtxRef = useRef(null)
    const animFrameRef = useRef(null)
    const mediaStreamRef = useRef(null)
    const chatBottomRef = useRef(null)
    const upgradePollRef = useRef(null)
    const isSendingRef = useRef(false)
    const asrLangRef = useRef(asrLang)

    useEffect(() => {
      asrLangRef.current = asrLang
    }, [asrLang])

    useEffect(() => {
      isSendingRef.current = isSending
    }, [isSending])

    useEffect(() => {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition
      if (!SR) setSpeechSupported(false)

      fetchLanguages()
      fetchModelInfo()
      fetchKnowledgeStatus()
      fetchSessions()
      fetchAnalytics()
      fetchWerHistory()
      runWerEvaluation(PRESET_WER_EXAMPLES[0].ref, PRESET_WER_EXAMPLES[0].hyp, false)
      searchKnowledge('')

      return () => {
        stopAudioVisualizer()
        if (upgradePollRef.current) clearInterval(upgradePollRef.current)
        if (recognitionRef.current) {
          try { recognitionRef.current.stop() } catch (_) {}
        }
        if (werRecognitionRef.current) {
          try { werRecognitionRef.current.stop() } catch (_) {}
        }
        if (window.speechSynthesis) window.speechSynthesis.cancel()
      }
    }, [])

    useEffect(() => {
      if (chatBottomRef.current) {
        chatBottomRef.current.scrollIntoView({ behavior: 'smooth' })
      }
    }, [messages, interimTranscript])

    const currentLangMeta = languages.find((l) => l.code === asrLang) || { rtl: false }

    async function fetchLanguages() {
      try {
        const res = await fetch('/api/languages')
        const data = await res.json()
        if (data.ok && Array.isArray(data.languages) && data.languages.length > 0) {
          setLanguages(data.languages)
        }
      } catch (_) {}
    }

    async function fetchModelInfo() {
      try {
        const res = await fetch('/api/model')
        const data = await res.json()
        if (data.ok) setModelInfo(data.model)
      } catch (_) {}
    }

    async function fetchKnowledgeStatus() {
      try {
        const res = await fetch('/api/knowledge/status')
        const data = await res.json()
        if (data.ok) {
          setKnowledgeStatus(data)
          setIsUpgrading(Boolean(data.is_upgrading))
        }
      } catch (_) {}
    }

    async function fetchAnalytics() {
      try {
        const res = await fetch('/api/analytics')
        const data = await res.json()
        if (data.ok) setAnalytics(data.analytics)
      } catch (_) {}
    }

    async function fetchSessions() {
      try {
        const res = await fetch('/api/sessions')
        const data = await res.json()
        if (data.ok) setSessions(data.sessions || [])
      } catch (_) {}
    }

    async function fetchWerHistory() {
      try {
        const res = await fetch('/api/wer/history')
        const data = await res.json()
        if (data.ok) setWerHistory(data.history || [])
      } catch (_) {}
    }

    async function searchKnowledge(q) {
      setKbSearching(true)
      try {
        const res = await fetch(`/api/knowledge/search?q=${encodeURIComponent(q)}&limit=9`)
        const data = await res.json()
        if (data.ok) setKbSearchResults(data.results || [])
      } catch (_) {
      } finally {
        setKbSearching(false)
      }
    }

    function startNewChat() {
      if (window.speechSynthesis) window.speechSynthesis.cancel()
      setSessionId(`sess_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`)
      setMessages([WELCOME_MESSAGE])
      setInputText('')
      setInterimTranscript('')
      setLastAsrConfidence(null)
    }

    async function loadSession(id) {
      try {
        const res = await fetch(`/api/sessions/${encodeURIComponent(id)}`)
        const data = await res.json()
        if (data.ok) {
          setSessionId(id)
          if (data.messages && data.messages.length > 0) {
            setMessages(
              data.messages.map((m) => ({
                id: m.id,
                dbMessageId: m.role === 'bot' ? m.id : null,
                role: m.role,
                content: m.content,
                inputMode: m.input_mode,
                asrConfidence: m.asr_confidence,
                intentTag: m.intent_tag,
                confidence: m.intent_confidence,
                detectedLanguage: m.detected_language ? { lang: m.detected_language, label: m.detected_language } : null,
                sourceTitle: m.source_title,
                sourceUrl: m.source_url,
                latencyMs: m.latency_ms,
                feedback: m.feedback
              }))
            )
          } else {
            setMessages([WELCOME_MESSAGE])
          }
          setActiveTab('chat')
        }
      } catch (_) {}
    }

    async function removeSession(id, ev) {
      ev.stopPropagation()
      try {
        await fetch(`/api/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' })
        setSessions((prev) => prev.filter((s) => s.id !== id))
        if (id === sessionId) startNewChat()
      } catch (_) {}
    }

    async function triggerAutoUpgrade() {
      if (upgradePollRef.current) clearInterval(upgradePollRef.current)
      setIsUpgrading(true)
      setUpgradeMessage('Initiating live ArXiv & Wikipedia knowledge harvest...')
      try {
        const res = await fetch('/api/knowledge/upgrade', { method: 'POST' })
        const data = await res.json()
        if (data.ok) {
          setUpgradeMessage('Harvesting latest CS/Math papers & articles in background...')
          upgradePollRef.current = setInterval(async () => {
            try {
              const pollRes = await fetch('/api/knowledge/status')
              const pollData = await pollRes.json()
              if (pollData.ok) {
                setKnowledgeStatus(pollData)
                if (!pollData.is_upgrading) {
                  clearInterval(upgradePollRef.current)
                  upgradePollRef.current = null
                  setIsUpgrading(false)
                  setUpgradeMessage(pollData.last_progress || 'Knowledge pool upgraded successfully!')
                  fetchModelInfo()
                  fetchAnalytics()
                  searchKnowledge(kbSearchQuery)
                } else {
                  setUpgradeMessage(pollData.last_progress || 'Upgrading...')
                }
              }
            } catch (pollErr) {
              clearInterval(upgradePollRef.current)
              upgradePollRef.current = null
              setIsUpgrading(false)
              setUpgradeMessage(`Status polling interrupted: ${pollErr.message}`)
            }
          }, 2000)
        } else {
          setIsUpgrading(false)
          setUpgradeMessage(data.error || 'Upgrade request rejected.')
        }
      } catch (err) {
        setIsUpgrading(false)
        setUpgradeMessage(`Network error: ${err.message}`)
      }
    }

    async function triggerOnDemandScrape() {
      const topic = customScrapeTopic.trim()
      if (!topic || isUpgrading) return
      setIsUpgrading(true)
      setUpgradeMessage(`Scraping Wikipedia & ArXiv for "${topic}"...`)
      try {
        const res = await fetch('/api/knowledge/scrape', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ topic, lang: customScrapeLang })
        })
        const data = await res.json()
        if (data.ok) {
          setUpgradeMessage(data.message || `Ingested "${topic}" into Knowledge Pool!`)
          setCustomScrapeTopic('')
          await Promise.all([fetchKnowledgeStatus(), fetchModelInfo(), fetchAnalytics()])
          searchKnowledge(topic)
        } else {
          setUpgradeMessage(data.error || 'Scrape failed.')
        }
      } catch (err) {
        setUpgradeMessage(`Network error: ${err.message}`)
      } finally {
        setIsUpgrading(false)
      }
    }

    function speakReply(rawText, langOverride) {
      if (!window.speechSynthesis) return
      window.speechSynthesis.cancel()
      const clean = String(rawText)
        .replace(/```[\s\S]*?```/g, ' code block ')
        .replace(/https?:\/\/\S+/g, '')
        .replace(/\*\*([^*]+)\*\*/g, '$1')
        .replace(/`([^`]+)`/g, '$1')
        .replace(/\$([^$]+)\$/g, '$1')
        .replace(/[#>*_]/g, '')
      const utter = new SpeechSynthesisUtterance(clean)
      utter.lang = langOverride || asrLangRef.current
      utter.rate = 1.02
      window.speechSynthesis.speak(utter)
    }

    async function sendFeedback(msg, rating) {
      if (!msg.dbMessageId) return
      try {
        const res = await fetch('/api/chat/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messageId: msg.dbMessageId, feedback: rating })
        })
        const data = await res.json()
        if (data.ok) {
          setMessages((prev) =>
            prev.map((m) => (m.id === msg.id ? { ...m, feedback: rating } : m))
          )
          fetchAnalytics()
        }
      } catch (_) {}
    }

    function copyText(id, text) {
      if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
          setCopiedId(id)
          setTimeout(() => setCopiedId(null), 1500)
        }).catch(() => {})
      }
    }

    async function sendMessage(textToSend, mode = 'text', asrConf = null) {
      const msg = (textToSend !== undefined ? textToSend : inputText).trim()
      if (!msg || isSendingRef.current) return

      setInputText('')
      setInterimTranscript('')
      setIsSending(true)
      isSendingRef.current = true

      const userMsgId = `usr_${Date.now()}`
      setMessages((prev) => [
        ...prev,
        {
          id: userMsgId,
          role: 'user',
          content: msg,
          inputMode: mode,
          asrConfidence: asrConf
        }
      ])

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sessionId,
            message: msg,
            inputMode: mode,
            asrConfidence: asrConf,
            asrLang: asrLangRef.current,
            autoLearn
          })
        })
        const data = await res.json()
        if (data.ok && data.botResponse) {
          const botResp = data.botResponse
          setMessages((prev) => [
            ...prev,
            {
              id: `bot_${Date.now()}`,
              dbMessageId: data.botMessageId || null,
              role: 'bot',
              content: botResp.reply,
              intentTag: botResp.intentTag,
              intentTitle: botResp.intentTitle,
              confidence: botResp.confidence,
              detectedLanguage: botResp.detectedLanguage,
              followUps: botResp.followUps || [],
              topPredictions: botResp.topPredictions || [],
              matchedTokens: botResp.matchedTokens || [],
              sources: botResp.sources || [],
              depthMode: botResp.depthMode || null,
              questionKind: botResp.questionKind || null,
              scrapedHosts: botResp.scrapedHosts || [],
              introspection: botResp.introspection || null,
              learnedNew: Boolean(botResp.learnedNew),
              latencyMs: botResp.latencyMs
            }
          ])
          if (ttsEnabled) {
            speakReply(botResp.reply, botResp.detectedLanguage?.lang || asrLangRef.current)
          }
          fetchSessions()
          fetchAnalytics()
          if (botResp.learnedNew) fetchKnowledgeStatus()
        } else {
          setMessages((prev) => [
            ...prev,
            {
              id: `err_${Date.now()}`,
              role: 'bot',
              content: data.error || 'Request could not be processed.',
              intentTag: 'error',
              confidence: 0
            }
          ])
        }
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          {
            id: `err_${Date.now()}`,
            role: 'bot',
            content: `Network error: ${err.message}`,
            intentTag: 'error',
            confidence: 0
          }
        ])
      } finally {
        setIsSending(false)
        isSendingRef.current = false
      }
    }

    async function startAudioVisualizer() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        mediaStreamRef.current = stream
        const AudioCtx = window.AudioContext || window.webkitAudioContext
        const ctx = new AudioCtx()
        audioCtxRef.current = ctx
        const source = ctx.createMediaStreamSource(stream)
        const analyser = ctx.createAnalyser()
        analyser.fftSize = 64
        source.connect(analyser)
        const bufferLength = analyser.frequencyBinCount
        const dataArray = new Uint8Array(bufferLength)

        function tick() {
          analyser.getByteFrequencyData(dataArray)
          const bars = []
          for (let i = 0; i < 20; i++) {
            const v = dataArray[i % bufferLength] || 0
            bars.push(Math.max(6, Math.min(40, Math.round((v / 255) * 40))))
          }
          setAudioBars(bars)
          animFrameRef.current = requestAnimationFrame(tick)
        }
        tick()
      } catch (_) {}
    }

    function stopAudioVisualizer() {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
      animFrameRef.current = null
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((t) => t.stop())
        mediaStreamRef.current = null
      }
      if (audioCtxRef.current) {
        audioCtxRef.current.close().catch(() => {})
        audioCtxRef.current = null
      }
      setAudioBars(new Array(20).fill(6))
    }

    function toggleMicrophone() {
      setMicError('')
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition
      if (!SR) {
        setMicError('Web Speech API is not supported in this browser. Please use Chrome, Edge, or Safari.')
        return
      }

      if (isListening && recognitionRef.current) {
        recognitionRef.current.stop()
        setIsListening(false)
        stopAudioVisualizer()
        return
      }

      if (window.speechSynthesis) window.speechSynthesis.cancel()

      const recognition = new SR()
      recognition.lang = asrLangRef.current
      recognition.interimResults = true
      recognition.continuous = false

      recognition.onstart = () => {
        setIsListening(true)
        startAudioVisualizer()
      }

      recognition.onresult = (event) => {
        let interim = ''
        let finalTxt = ''
        let conf = null

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          const res = event.results[i]
          if (res.isFinal) {
            finalTxt += res[0].transcript
            conf = res[0].confidence
          } else {
            interim += res[0].transcript
          }
        }

        setInterimTranscript(interim)
        if (finalTxt) {
          setLastAsrConfidence(conf)
          if (autoSendVoice) {
            sendMessage(finalTxt, 'voice', conf)
          } else {
            setInputText((prev) => (prev ? `${prev} ${finalTxt}` : finalTxt))
          }
        }
      }

      recognition.onerror = (event) => {
        const errCode = event.error || 'unknown'
        const friendly =
          errCode === 'not-allowed'
            ? 'Microphone permission denied. Please allow microphone access in browser settings.'
            : errCode === 'no-speech'
            ? 'No speech detected. Please try speaking closer to your microphone.'
            : `Microphone error: ${errCode}`
        setMicError(friendly)
        setIsListening(false)
        stopAudioVisualizer()
      }

      recognition.onend = () => {
        setIsListening(false)
        stopAudioVisualizer()
        setInterimTranscript('')
      }

      recognitionRef.current = recognition
      try {
        recognition.start()
      } catch (err) {
        setMicError(err.message)
        setIsListening(false)
        stopAudioVisualizer()
      }
    }

    function toggleWerMicrophone() {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition
      if (!SR) {
        setMicError('Web Speech API is not supported in this browser.')
        return
      }
      if (isListeningWer && werRecognitionRef.current) {
        werRecognitionRef.current.stop()
        setIsListeningWer(false)
        return
      }
      const recognition = new SR()
      recognition.lang = asrLangRef.current
      recognition.interimResults = false
      recognition.continuous = false
      recognition.onstart = () => setIsListeningWer(true)
      recognition.onresult = (event) => {
        const transcript = Array.from(event.results).map((r) => r[0].transcript).join(' ').trim()
        if (transcript) {
          setWerHyp(transcript)
          runWerEvaluation(werRef, transcript, true)
        }
      }
      recognition.onerror = () => setIsListeningWer(false)
      recognition.onend = () => setIsListeningWer(false)
      werRecognitionRef.current = recognition
      try {
        recognition.start()
      } catch (_) {
        setIsListeningWer(false)
      }
    }

    async function runWerEvaluation(ref, hyp, persist = true) {
      if (!String(ref || '').trim()) return
      try {
        const res = await fetch('/api/wer', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ reference: ref, hypothesis: hyp, language: asrLangRef.current, persist })
        })
        const data = await res.json()
        if (data.ok) {
          setWerResult(data.metrics)
          if (persist) {
            fetchWerHistory()
            fetchAnalytics()
          }
        }
      } catch (_) {}
    }

    async function clearWerLog() {
      try {
        await fetch('/api/wer/history', { method: 'DELETE' })
        setWerHistory([])
      } catch (_) {}
    }

    // ==================== HEADER ====================
    const headerNode = e(
      'header',
      { className: 'border-b border-slate-800 bg-slate-950/90 backdrop-blur sticky top-0 z-30 px-4 py-3' },
      e(
        'div',
        { className: 'max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3' },
        e(
          'div',
          { className: 'flex items-center gap-3' },
          e(
            'div',
            { className: 'w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-emerald-400 flex items-center justify-center shadow-lg shadow-indigo-500/25' },
            e('i', { className: 'fas fa-brain text-white text-lg' })
          ),
          e(
            'div',
            null,
            e(
              'h1',
              { className: 'text-base font-bold bg-gradient-to-r from-white via-indigo-100 to-indigo-300 bg-clip-text text-transparent' },
              'VoxASR — Multilingual CS & Mathematical AI'
            ),
            e(
              'div',
              { className: 'flex items-center gap-2 text-xs text-slate-400' },
              e('span', { className: 'w-2 h-2 rounded-full bg-emerald-400 animate-pulse' }),
              e(
                'span',
                null,
                `${languages.length} Certified Languages • ${knowledgeStatus?.total_scraped_articles || 0} KB Articles • Live Web Learning`
              )
            )
          )
        ),
        e(
          'nav',
          { className: 'flex flex-wrap items-center gap-1.5 bg-slate-900/90 p-1 rounded-xl border border-slate-800' },
          [
            { id: 'chat', icon: 'fa-microphone-lines', label: 'ASR Voice Chat' },
            { id: 'knowledge', icon: 'fa-database', label: 'Knowledge Pool' },
            { id: 'wer', icon: 'fa-scale-balanced', label: 'WER Lab' },
            { id: 'model', icon: 'fa-chart-pie', label: 'Analytics & Model' }
          ].map((tab) =>
            e(
              'button',
              {
                key: tab.id,
                onClick: () => setActiveTab(tab.id),
                className: `px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2 ${
                  activeTab === tab.id
                    ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`
              },
              e('i', { className: `fas ${tab.icon}` }),
              tab.label
            )
          )
        )
      )
    )

    const filteredSessions = sessions.filter((s) =>
      !sessionSearch.trim() || String(s.title || '').toLowerCase().includes(sessionSearch.toLowerCase())
    )

    // ==================== TAB 1: VOICE CHAT ====================
    const chatView = e(
      'div',
      { className: 'grid grid-cols-1 lg:grid-cols-12 gap-5 max-w-7xl mx-auto p-4 lg:p-6 w-full flex-1' },
      // Left Sidebar
      e(
        'aside',
        { className: 'lg:col-span-3 space-y-4' },
        // Speech Settings
        e(
          'div',
          { className: 'bg-slate-900/80 border border-slate-800 rounded-2xl p-4 space-y-3' },
          e(
            'div',
            { className: 'flex items-center justify-between' },
            e('h2', { className: 'text-xs font-semibold uppercase tracking-wider text-slate-400' }, 'Speech & AI Settings'),
            e('span', { className: 'text-[11px] px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20' }, `${languages.length} Langs`)
          ),
          e(
            'div',
            { className: 'space-y-1' },
            e('label', { className: 'text-xs text-slate-400 block' }, 'Speech Recognition Locale:'),
            e(
              'select',
              {
                value: asrLang,
                onChange: (ev) => setAsrLang(ev.target.value),
                className: 'w-full bg-slate-950 border border-slate-700/80 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500'
              },
              languages.map((l) =>
                e('option', { key: l.code, value: l.code }, `${l.flag} ${l.name} (${l.code})`)
              )
            )
          ),
          e(
            'label',
            { className: 'flex items-center justify-between text-xs text-slate-300 cursor-pointer pt-1' },
            e('span', null, 'Auto-send after voice dictation'),
            e('input', {
              type: 'checkbox',
              checked: autoSendVoice,
              onChange: (ev) => setAutoSendVoice(ev.target.checked),
              className: 'accent-indigo-500 rounded'
            })
          ),
          e(
            'label',
            { className: 'flex items-center justify-between text-xs text-slate-300 cursor-pointer' },
            e('span', null, 'Auto-speak Bot Replies (TTS)'),
            e('input', {
              type: 'checkbox',
              checked: ttsEnabled,
              onChange: (ev) => setTtsEnabled(ev.target.checked),
              className: 'accent-indigo-500 rounded'
            })
          ),
          e(
            'label',
            { className: 'flex items-center justify-between text-xs text-slate-300 cursor-pointer' },
            e('span', null, 'Live Web Learning (Wiki/ArXiv)'),
            e('input', {
              type: 'checkbox',
              checked: autoLearn,
              onChange: (ev) => setAutoLearn(ev.target.checked),
              className: 'accent-indigo-500 rounded'
            })
          ),
          e(
            'div',
            { className: 'pt-2 border-t border-slate-800 space-y-2' },
            e(
              'button',
              {
                onClick: toggleMicrophone,
                disabled: !speechSupported,
                className: `w-full py-2.5 px-3 rounded-xl font-medium text-xs flex items-center justify-center gap-2 transition ${
                  !speechSupported
                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                    : isListening
                    ? 'bg-rose-600 text-white mic-recording-pulse shadow-lg shadow-rose-600/30'
                    : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-md shadow-indigo-600/25'
                }`
              },
              e('i', { className: `fas ${isListening ? 'fa-microphone-slash' : 'fa-microphone'}` }),
              !speechSupported
                ? 'Speech API Unavailable'
                : isListening
                ? 'Stop Listening...'
                : 'Click to Speak Live'
            ),
            lastAsrConfidence !== null && e(
              'div',
              { className: 'text-[11px] text-center text-emerald-400' },
              `Last ASR Confidence: ${Math.round(lastAsrConfidence * 100)}%`
            )
          )
        ),
        // Sessions History
        e(
          'div',
          { className: 'bg-slate-900/80 border border-slate-800 rounded-2xl p-4 space-y-2.5' },
          e(
            'div',
            { className: 'flex items-center justify-between' },
            e('h3', { className: 'text-xs font-semibold text-slate-400 uppercase tracking-wider' }, 'Conversations'),
            e(
              'button',
              {
                onClick: startNewChat,
                className: 'px-2 py-1 rounded-lg bg-indigo-600/20 hover:bg-indigo-600 text-indigo-300 hover:text-white text-[11px] font-medium transition'
              },
              '+ New Chat'
            )
          ),
          e('input', {
            type: 'text',
            value: sessionSearch,
            onChange: (ev) => setSessionSearch(ev.target.value),
            placeholder: 'Filter history...',
            className: 'w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500'
          }),
          e(
            'div',
            { className: 'space-y-1 max-h-44 overflow-y-auto pr-1' },
            filteredSessions.length === 0
              ? e('div', { className: 'text-[11px] text-slate-500 py-2 text-center' }, 'No saved sessions yet.')
              : filteredSessions.map((s) =>
                  e(
                    'div',
                    {
                      key: s.id,
                      onClick: () => loadSession(s.id),
                      className: `group flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-lg cursor-pointer text-xs transition ${
                        s.id === sessionId
                          ? 'bg-indigo-600/20 border border-indigo-500/40 text-indigo-200'
                          : 'bg-slate-950/60 hover:bg-slate-800 text-slate-300 border border-transparent'
                      }`
                    },
                    e('span', { className: 'truncate flex-1' }, s.title || 'Voice ASR Session'),
                    e('span', { className: 'text-[10px] text-slate-500 shrink-0' }, s.message_count || 0),
                    e(
                      'button',
                      {
                        onClick: (ev) => removeSession(s.id, ev),
                        title: 'Delete session',
                        className: 'opacity-60 hover:opacity-100 text-rose-400 px-1'
                      },
                      e('i', { className: 'fas fa-trash text-[10px]' })
                    )
                  )
                )
          ),
          e(
            'div',
            { className: 'flex items-center gap-2 pt-1 border-t border-slate-800' },
            e(
              'a',
              {
                href: `/api/sessions/${encodeURIComponent(sessionId)}/export?format=markdown`,
                className: 'flex-1 text-center py-1 rounded-lg bg-slate-950 hover:bg-slate-800 border border-slate-800 text-[11px] text-slate-300 transition'
              },
              'Export MD'
            ),
            e(
              'a',
              {
                href: `/api/sessions/${encodeURIComponent(sessionId)}/export?format=json`,
                className: 'flex-1 text-center py-1 rounded-lg bg-slate-950 hover:bg-slate-800 border border-slate-800 text-[11px] text-slate-300 transition'
              },
              'Export JSON'
            )
          )
        ),
        // Multilingual Query Suggestions
        e(
          'div',
          { className: 'bg-slate-900/80 border border-slate-800 rounded-2xl p-4 space-y-2' },
          e('h3', { className: 'text-xs font-semibold text-slate-400 uppercase tracking-wider' }, 'Multilingual Prompts'),
          e(
            'div',
            { className: 'space-y-1.5' },
            MULTILINGUAL_SHOWCASE.slice(0, 5).map((q, idx) =>
              e(
                'button',
                {
                  key: idx,
                  onClick: () => sendMessage(q.text, 'text'),
                  className: 'w-full text-left p-2 rounded-lg bg-slate-950/60 hover:bg-slate-800 text-xs text-slate-300 hover:text-white transition flex items-start gap-2 border border-slate-800/80'
                },
                e('span', null, q.flag),
                e('span', { className: 'truncate' }, q.text)
              )
            )
          )
        )
      ),
      // Chat Stream
      e(
        'main',
        { className: 'lg:col-span-9 flex flex-col h-[760px] bg-slate-900/60 border border-slate-800 rounded-2xl overflow-hidden' },
        // Visible Microphone Error Banner
        micError && e(
          'div',
          { className: 'px-4 py-2.5 bg-rose-950/80 border-b border-rose-500/40 text-xs text-rose-200 flex items-center justify-between gap-3' },
          e('div', { className: 'flex items-center gap-2' }, e('i', { className: 'fas fa-triangle-exclamation text-rose-400' }), micError),
          e('button', { onClick: () => setMicError(''), className: 'text-rose-300 hover:text-white' }, e('i', { className: 'fas fa-xmark' }))
        ),
        // Message List
        e(
          'div',
          { className: 'flex-1 overflow-y-auto p-4 sm:p-5 space-y-4' },
          messages.map((m) => {
            const isBot = m.role === 'bot'
            const isRtl = ['ar', 'fa', 'ur', 'he'].some((p) => (m.detectedLanguage?.lang || '').startsWith(p))
            return e(
              'div',
              { key: m.id, className: `flex gap-3 ${isBot ? 'items-start' : 'items-start flex-row-reverse'}` },
              e(
                'div',
                {
                  className: `w-8 h-8 rounded-lg flex items-center justify-center shrink-0 text-xs ${
                    isBot ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/20' : 'bg-slate-700 text-slate-200'
                  }`
                },
                e('i', { className: `fas ${isBot ? 'fa-robot' : 'fa-user'}` })
              ),
              e(
                'div',
                { className: `max-w-[85%] space-y-1.5 ${isBot ? 'text-left' : 'text-right'}` },
                // Header tags
                isBot
                  ? m.intentTitle && e(
                      'div',
                      { className: 'flex flex-wrap items-center gap-2 text-[11px] text-slate-400' },
                      e('span', { className: 'px-2 py-0.5 rounded-md bg-indigo-500/10 text-indigo-300 border border-indigo-500/20' }, m.intentTitle),
                      m.depthMode && e('span', { className: 'px-2 py-0.5 rounded-md bg-amber-500/10 text-amber-300 border border-amber-500/20 capitalize' }, `${m.depthMode} mode`),
                      m.scrapedHosts && m.scrapedHosts.length > 0 && e('span', { className: 'px-2 py-0.5 rounded-md bg-cyan-500/10 text-cyan-300 border border-cyan-500/20' }, `🌐 ${m.scrapedHosts.length} web site${m.scrapedHosts.length === 1 ? '' : 's'}`),
                      m.detectedLanguage && e('span', { className: 'text-slate-400' }, `🌐 ${m.detectedLanguage.label || m.detectedLanguage.lang}`),
                      typeof m.confidence === 'number' && e('span', { className: 'text-emerald-400' }, `${Math.round(m.confidence * 100)}% match`),
                      m.learnedNew && e('span', { className: 'px-2 py-0.5 rounded-md bg-emerald-500/15 text-emerald-300 border border-emerald-500/30' }, 'Live Browsed'),
                      typeof m.latencyMs === 'number' && e('span', { className: 'text-slate-500 font-mono' }, `${m.latencyMs}ms`)
                    )
                  : e(
                      'div',
                      { className: 'flex items-center justify-end gap-2 text-[11px] text-slate-400' },
                      m.inputMode === 'voice' && e(
                        'span',
                        { className: 'px-2 py-0.5 rounded-md bg-emerald-500/10 text-emerald-300 border border-emerald-500/20' },
                        `🎤 Voice${typeof m.asrConfidence === 'number' ? ` • ${Math.round(m.asrConfidence * 100)}% ASR` : ''}`
                      )
                    ),
                // Bubble
                e(
                  'div',
                  {
                    dir: isRtl ? 'rtl' : 'auto',
                    className: `p-3.5 rounded-2xl text-xs sm:text-sm ${
                      isBot
                        ? 'bg-slate-900 border border-slate-800 text-slate-200 shadow-sm'
                        : 'bg-indigo-600 text-white text-left'
                    }`
                  },
                  isBot ? renderFormattedText(m.content) : m.content
                ),
                // Sources pills
                isBot && m.sources && m.sources.length > 0 && e(
                  'div',
                  { className: 'flex flex-wrap gap-1.5 pt-0.5' },
                  m.sources.map((src, si) =>
                    e(
                      'a',
                      {
                        key: si,
                        href: src.url,
                        target: '_blank',
                        rel: 'noopener noreferrer',
                        className: 'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-950 hover:bg-slate-800 border border-slate-800 text-[11px] text-emerald-300 transition'
                      },
                      e('i', { className: 'fas fa-arrow-up-right-from-square text-[9px]' }),
                      `${src.source}: ${src.title}`
                    )
                  )
                ),
                // Message Actions (Copy, Speak, Feedback, Probability Details)
                isBot && e(
                  'div',
                  { className: 'flex flex-wrap items-center gap-2 text-[11px] text-slate-500 pt-0.5' },
                  e(
                    'button',
                    {
                      onClick: () => copyText(m.id, m.content),
                      className: 'hover:text-slate-200 flex items-center gap-1 transition'
                    },
                    e('i', { className: `fas ${copiedId === m.id ? 'fa-check text-emerald-400' : 'fa-copy'}` }),
                    copiedId === m.id ? 'Copied' : 'Copy'
                  ),
                  e(
                    'button',
                    {
                      onClick: () => speakReply(m.content, m.detectedLanguage?.lang),
                      className: 'hover:text-slate-200 flex items-center gap-1 transition'
                    },
                    e('i', { className: 'fas fa-volume-high' }),
                    'Speak'
                  ),
                  m.dbMessageId && e(
                    'button',
                    {
                      onClick: () => sendFeedback(m, 'up'),
                      className: `hover:text-emerald-400 transition ${m.feedback === 'up' ? 'text-emerald-400' : ''}`,
                      title: 'Helpful'
                    },
                    e('i', { className: 'fas fa-thumbs-up' })
                  ),
                  m.dbMessageId && e(
                    'button',
                    {
                      onClick: () => sendFeedback(m, 'down'),
                      className: `hover:text-rose-400 transition ${m.feedback === 'down' ? 'text-rose-400' : ''}`,
                      title: 'Needs improvement'
                    },
                    e('i', { className: 'fas fa-thumbs-down' })
                  ),
                  (m.topPredictions && m.topPredictions.length > 0 || m.introspection) && e(
                    'button',
                    {
                      onClick: () => setExpandedDetails((prev) => ({ ...prev, [m.id]: !prev[m.id] })),
                      className: 'hover:text-indigo-300 transition ml-auto'
                    },
                    expandedDetails[m.id] ? 'Hide CNN & Introspection' : 'CNN + Introspection'
                  )
                ),
                expandedDetails[m.id] && (m.topPredictions || m.introspection) && e(
                  'div',
                  { className: 'p-2.5 rounded-xl bg-slate-950 border border-slate-800 space-y-2 text-[11px]' },
                  m.topPredictions && m.topPredictions.length > 0 && e(
                    'div',
                    { className: 'space-y-1' },
                    e('div', { className: 'text-[10px] uppercase tracking-wider text-slate-500 font-semibold' }, 'TextCNN Domain Predictions'),
                    m.topPredictions.map((p) =>
                      e(
                        'div',
                        { key: p.tag, className: 'flex items-center justify-between gap-2' },
                        e('span', { className: 'font-mono text-slate-300 truncate' }, p.tag),
                        e('span', { className: 'text-indigo-400 font-mono' }, `${(p.probability * 100).toFixed(1)}%`)
                      )
                    )
                  ),
                  m.introspection && m.introspection.steps && e(
                    'div',
                    { className: 'pt-2 border-t border-slate-800 space-y-1' },
                    e('div', { className: 'text-[10px] uppercase tracking-wider text-emerald-400 font-semibold' }, `Recursive Introspection Trace (Q=${Math.round((m.introspection.finalSufficiencyQ || 0) * 100)}% • ${m.introspection.totalWebSources || 0} Web Sources)`),
                    m.introspection.steps.map((st) =>
                      e(
                        'div',
                        { key: st.step, className: 'flex flex-wrap items-center justify-between gap-2 text-slate-400 font-mono text-[10px]' },
                        e('span', null, `Step ${st.step}: ${st.action} -> ${st.nextAction}`),
                        e('span', { className: 'text-cyan-300' }, `Q=${Math.round((st.sufficiencyQ || 0) * 100)}% | Web=${st.webSources || 0}`)
                      )
                    )
                  )
                ),
                // Follow ups
                isBot && m.followUps && m.followUps.length > 0 && e(
                  'div',
                  { className: 'flex flex-wrap gap-1.5 pt-1' },
                  m.followUps.map((fu, fi) =>
                    e(
                      'button',
                      {
                        key: fi,
                        onClick: () => sendMessage(fu, 'text'),
                        className: 'px-2.5 py-1 rounded-lg text-[11px] bg-slate-950/70 hover:bg-slate-800 text-indigo-300 border border-indigo-500/20 transition'
                      },
                      `💡 ${fu}`
                    )
                  )
                )
              )
            )
          }),
          isSending && e(
            'div',
            { className: 'flex gap-3 items-center text-xs text-slate-400' },
            e('div', { className: 'w-8 h-8 rounded-lg bg-indigo-600/50 text-white flex items-center justify-center' }, e('i', { className: 'fas fa-spinner fa-spin' })),
            e('span', null, 'Browsing web sources & running TextCNN + Recursive Introspection...')
          ),
          interimTranscript && e(
            'div',
            { className: 'flex gap-3 items-start' },
            e('div', { className: 'w-8 h-8 rounded-lg bg-indigo-600/50 text-white flex items-center justify-center shrink-0' }, e('i', { className: 'fas fa-microphone animate-pulse' })),
            e('div', { className: 'p-3 rounded-2xl bg-indigo-950/40 border border-indigo-500/30 text-indigo-200 text-xs italic animate-pulse' }, interimTranscript)
          ),
          e('div', { ref: chatBottomRef })
        ),
        // Live Audio Spectrum Visualizer
        isListening && e(
          'div',
          { className: 'px-4 py-2 bg-slate-950/80 border-t border-slate-800 flex items-center justify-center gap-1' },
          audioBars.map((h, i) =>
            e('div', {
              key: i,
              style: { height: `${h}px` },
              className: 'w-1.5 bg-gradient-to-t from-indigo-500 to-emerald-400 rounded-full transition-all duration-75'
            })
          )
        ),
        // Input Box
        e(
          'div',
          { className: 'p-3 bg-slate-950/90 border-t border-slate-800' },
          e(
            'form',
            {
              onSubmit: (ev) => {
                ev.preventDefault()
                sendMessage(inputText, 'text')
              },
              className: 'flex items-center gap-2'
            },
            e('input', {
              type: 'text',
              dir: currentLangMeta.rtl ? 'rtl' : 'auto',
              value: inputText,
              onChange: (ev) => setInputText(ev.target.value),
              placeholder: `Ask any CS/Math question in ${asrLang} or click mic to speak...`,
              className: 'flex-1 bg-slate-900 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs sm:text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500'
            }),
            e(
              'button',
              {
                type: 'button',
                onClick: toggleMicrophone,
                disabled: !speechSupported,
                title: 'Toggle Voice Microphone',
                className: `p-2.5 rounded-xl border text-sm transition ${
                  isListening
                    ? 'bg-rose-600 text-white border-rose-500 animate-pulse'
                    : 'bg-slate-900 hover:bg-slate-800 text-slate-300 border-slate-800'
                }`
              },
              e('i', { className: `fas ${isListening ? 'fa-microphone-slash' : 'fa-microphone'}` })
            ),
            e(
              'button',
              {
                type: 'submit',
                disabled: isSending || !inputText.trim(),
                className: 'px-4 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-xl text-xs sm:text-sm font-medium transition flex items-center gap-2'
              },
              e('span', null, 'Send'),
              e('i', { className: 'fas fa-paper-plane' })
            )
          )
        )
      )
    )

    // ==================== TAB 2: KNOWLEDGE POOL ====================
    const knowledgeView = e(
      'div',
      { className: 'max-w-7xl mx-auto p-4 lg:p-6 space-y-6 w-full' },
      e(
        'div',
        { className: 'bg-gradient-to-r from-indigo-900/40 via-purple-900/30 to-slate-900 border border-indigo-500/20 rounded-3xl p-6 sm:p-8 space-y-4' },
        e(
          'div',
          { className: 'flex flex-wrap items-center justify-between gap-4' },
          e(
            'div',
            { className: 'space-y-1' },
            e('span', { className: 'text-xs font-semibold px-2.5 py-1 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30' }, 'Live BM25 Index & Automated Harvester'),
            e('h2', { className: 'text-xl sm:text-2xl font-bold text-white' }, 'Dynamic CS & Mathematical Knowledge Pool'),
            e('p', { className: 'text-xs sm:text-sm text-slate-300 max-w-3xl' }, 'Self-updating knowledge engine combining sharded Wikipedia encyclopedic articles, ArXiv preprints, and instant real-time web learning persisted in the database.')
          ),
          e(
            'button',
            {
              onClick: triggerAutoUpgrade,
              disabled: isUpgrading,
              className: `px-5 py-3 rounded-xl font-medium text-xs sm:text-sm flex items-center gap-2.5 transition shadow-lg ${
                isUpgrading
                  ? 'bg-slate-800 text-slate-400 cursor-not-allowed'
                  : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-600/30'
              }`
            },
            e('i', { className: `fas ${isUpgrading ? 'fa-spinner fa-spin' : 'fa-arrows-rotate'}` }),
            isUpgrading ? 'Harvesting Live...' : 'Trigger ArXiv & Wiki Harvest'
          )
        ),
        upgradeMessage && e(
          'div',
          { className: 'p-3 rounded-xl bg-slate-950/80 border border-indigo-500/30 text-xs text-indigo-300 flex items-center gap-2' },
          e('i', { className: 'fas fa-info-circle text-indigo-400' }),
          upgradeMessage
        )
      ),

      // Stat Cards
      e(
        'div',
        { className: 'grid grid-cols-2 lg:grid-cols-4 gap-4' },
        [
          { label: 'Certified Languages', val: `${languages.length} Languages`, desc: 'ISO-639 / BCP-47 + RTL', icon: 'fa-globe' },
          { label: 'Indexed Articles', val: `${knowledgeStatus?.total_scraped_articles || 0}`, desc: `${knowledgeStatus?.learned_articles || 0} self-learned live`, icon: 'fa-book-open' },
          { label: 'BM25 Search Terms', val: `${knowledgeStatus?.kb_terms || 0}`, desc: 'Sub-millisecond extractive QA', icon: 'fa-magnifying-glass' },
          { label: 'TF-IDF Vocab Features', val: `${modelInfo?.vocab_size || 5000}`, desc: 'Word 1-2g + Char 3-4g + CJK', icon: 'fa-brain' }
        ].map((c, i) =>
          e(
            'div',
            { key: i, className: 'bg-slate-900/80 border border-slate-800 rounded-2xl p-4 sm:p-5 space-y-1' },
            e('div', { className: 'text-xs text-slate-400 flex items-center justify-between' }, c.label, e('i', { className: `fas ${c.icon} text-indigo-400` })),
            e('div', { className: 'text-xl sm:text-2xl font-bold text-slate-100' }, c.val),
            e('div', { className: 'text-[11px] text-slate-500' }, c.desc)
          )
        )
      ),

      // On-Demand Topic Ingestion + Live Knowledge Explorer Search
      e(
        'div',
        { className: 'grid grid-cols-1 lg:grid-cols-2 gap-5' },
        e(
          'div',
          { className: 'bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-3' },
          e('h3', { className: 'text-sm font-semibold text-slate-200' }, 'On-Demand Topic Scraping & Learning'),
          e('p', { className: 'text-xs text-slate-400' }, 'Fetch any Wikipedia or ArXiv topic immediately into the live Knowledge Pool:'),
          e(
            'div',
            { className: 'flex flex-wrap sm:flex-nowrap gap-2.5' },
            e('input', {
              type: 'text',
              value: customScrapeTopic,
              onChange: (ev) => setCustomScrapeTopic(ev.target.value),
              onKeyDown: (ev) => { if (ev.key === 'Enter') triggerOnDemandScrape() },
              placeholder: 'e.g. FlashAttention, Kalman filter, Elliptic curve...',
              className: 'flex-1 bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs sm:text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500'
            }),
            e(
              'select',
              {
                value: customScrapeLang,
                onChange: (ev) => setCustomScrapeLang(ev.target.value),
                className: 'bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-300'
              },
              e('option', { value: 'en' }, 'en'),
              e('option', { value: 'es' }, 'es'),
              e('option', { value: 'fr' }, 'fr'),
              e('option', { value: 'de' }, 'de'),
              e('option', { value: 'hi' }, 'hi'),
              e('option', { value: 'zh' }, 'zh')
            ),
            e(
              'button',
              {
                onClick: triggerOnDemandScrape,
                disabled: isUpgrading || !customScrapeTopic.trim(),
                className: 'px-4 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-xl text-xs sm:text-sm font-medium transition shrink-0'
              },
              'Scrape & Learn'
            )
          )
        ),
        e(
          'div',
          { className: 'bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-3' },
          e('h3', { className: 'text-sm font-semibold text-slate-200' }, 'Search Indexed Knowledge Pool'),
          e('p', { className: 'text-xs text-slate-400' }, 'Search articles currently indexed in the in-memory BM25 engine:'),
          e(
            'div',
            { className: 'flex gap-2.5' },
            e('input', {
              type: 'text',
              value: kbSearchQuery,
              onChange: (ev) => {
                setKbSearchQuery(ev.target.value)
                searchKnowledge(ev.target.value)
              },
              placeholder: 'Search SVD, Dijkstra, Transformer, Raft, RSA...',
              className: 'flex-1 bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs sm:text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500'
            }),
            kbSearching && e('span', { className: 'self-center text-xs text-indigo-400' }, e('i', { className: 'fas fa-spinner fa-spin' }))
          )
        )
      ),

      // Search Results Grid
      kbSearchResults.length > 0 && e(
        'div',
        { className: 'space-y-3' },
        e('h3', { className: 'text-xs font-semibold uppercase tracking-wider text-slate-400' }, kbSearchQuery ? `Matching Knowledge Articles (${kbSearchResults.length})` : 'Featured Knowledge Articles'),
        e(
          'div',
          { className: 'grid grid-cols-1 md:grid-cols-3 gap-4' },
          kbSearchResults.map((art, idx) =>
            e(
              'div',
              { key: `${art.title}-${idx}`, className: 'bg-slate-900/80 border border-slate-800 rounded-2xl p-4 flex flex-col justify-between space-y-2' },
              e(
                'div',
                { className: 'space-y-1.5' },
                e(
                  'div',
                  { className: 'flex items-center justify-between gap-2' },
                  e('span', { className: 'text-xs font-bold text-indigo-300 truncate' }, art.title),
                  art.learned && e('span', { className: 'text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 shrink-0' }, 'Learned')
                ),
                e('p', { className: 'text-xs text-slate-400 line-clamp-3 leading-relaxed' }, art.summary)
              ),
              e(
                'div',
                { className: 'flex items-center justify-between pt-2 border-t border-slate-800/80' },
                e('span', { className: 'text-[10px] text-slate-500 truncate' }, art.source),
                e(
                  'button',
                  {
                    onClick: () => {
                      setActiveTab('chat')
                      sendMessage(`What is ${art.title}?`, 'text')
                    },
                    className: 'text-[11px] text-indigo-400 hover:text-indigo-300 font-medium'
                  },
                  'Ask AI →'
                )
              )
            )
          )
        )
      ),

      // 12 Domains Explorer
      e(
        'div',
        { className: 'space-y-3' },
        e('h3', { className: 'text-sm font-semibold uppercase tracking-wider text-slate-400' }, '12 Core Computer Science & Mathematical Computing Domains'),
        e(
          'div',
          { className: 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4' },
          CS_MATH_PILLARS.map((p) =>
            e(
              'div',
              { key: p.id, className: 'bg-slate-900/80 hover:bg-slate-800/80 border border-slate-800 hover:border-indigo-500/30 rounded-2xl p-4 space-y-2 transition flex flex-col justify-between' },
              e(
                'div',
                { className: 'space-y-1' },
                e('div', { className: 'flex items-center gap-2 text-indigo-400 text-sm font-semibold' }, e('i', { className: `fas ${p.icon}` }), p.title),
                e('p', { className: 'text-xs text-slate-400 leading-relaxed' }, p.desc)
              ),
              e(
                'button',
                {
                  onClick: () => {
                    setActiveTab('chat')
                    sendMessage(p.query, 'text')
                  },
                  className: 'w-full py-1.5 px-3 rounded-lg bg-indigo-600/10 hover:bg-indigo-600 text-indigo-300 hover:text-white border border-indigo-500/20 text-xs font-medium transition text-left truncate flex items-center justify-between'
                },
                e('span', { className: 'truncate' }, `Ask: "${p.query}"`),
                e('i', { className: 'fas fa-arrow-right text-[10px]' })
              )
            )
          )
        )
      )
    )

    // ==================== TAB 3: WER LAB ====================
    const werView = e(
      'div',
      { className: 'max-w-5xl mx-auto p-4 lg:p-6 space-y-6 w-full' },
      e(
        'div',
        { className: 'bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4' },
        e(
          'div',
          { className: 'flex flex-wrap items-center justify-between gap-3' },
          e(
            'div',
            null,
            e('h2', { className: 'text-base font-bold text-white' }, 'Word Error Rate (WER) & CER Evaluator Lab'),
            e('p', { className: 'text-xs text-slate-400' }, 'Calculate speech recognition accuracy using dynamic programming Levenshtein alignment across Substitutions (S), Deletions (D), and Insertions (I).')
          ),
          e(
            'button',
            {
              onClick: toggleWerMicrophone,
              disabled: !speechSupported,
              className: `px-3.5 py-2 rounded-xl text-xs font-medium flex items-center gap-2 transition ${
                isListeningWer
                  ? 'bg-rose-600 text-white mic-recording-pulse'
                  : 'bg-indigo-600 hover:bg-indigo-500 text-white'
              }`
            },
            e('i', { className: `fas ${isListeningWer ? 'fa-microphone-slash' : 'fa-microphone'}` }),
            isListeningWer ? 'Listening Hypothesis...' : `Dictate Hypothesis (${asrLang})`
          )
        ),
        e(
          'div',
          { className: 'flex flex-wrap gap-2' },
          PRESET_WER_EXAMPLES.map((ex, idx) =>
            e(
              'button',
              {
                key: idx,
                onClick: () => {
                  setWerRef(ex.ref)
                  setWerHyp(ex.hyp)
                  runWerEvaluation(ex.ref, ex.hyp, true)
                },
                className: 'px-2.5 py-1.5 rounded-lg bg-slate-950 hover:bg-slate-800 border border-slate-800 text-xs text-slate-300 transition'
              },
              ex.label
            )
          )
        ),
        e(
          'div',
          { className: 'grid grid-cols-1 md:grid-cols-2 gap-4' },
          e(
            'div',
            { className: 'space-y-1' },
            e('label', { className: 'text-xs text-slate-400 block' }, 'Ground-Truth Reference:'),
            e('textarea', {
              rows: 3,
              value: werRef,
              onChange: (ev) => {
                setWerRef(ev.target.value)
                runWerEvaluation(ev.target.value, werHyp, false)
              },
              className: 'w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-indigo-500'
            })
          ),
          e(
            'div',
            { className: 'space-y-1' },
            e('label', { className: 'text-xs text-slate-400 block' }, 'ASR Spoken Hypothesis:'),
            e('textarea', {
              rows: 3,
              value: werHyp,
              onChange: (ev) => {
                setWerHyp(ev.target.value)
                runWerEvaluation(werRef, ev.target.value, false)
              },
              className: 'w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-indigo-500'
            })
          )
        ),
        e(
          'div',
          { className: 'flex justify-end' },
          e(
            'button',
            {
              onClick: () => runWerEvaluation(werRef, werHyp, true),
              className: 'px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition'
            },
            'Save Benchmark Run'
          )
        ),
        werResult && e(
          'div',
          { className: 'space-y-4 pt-2' },
          e(
            'div',
            { className: 'grid grid-cols-2 sm:grid-cols-4 gap-3' },
            [
              { label: 'Word Error Rate (WER)', val: `${werResult.wer}%`, color: werResult.wer === 0 ? 'text-emerald-400' : 'text-amber-400' },
              { label: 'Word Accuracy', val: `${werResult.wordAccuracy}%`, color: 'text-indigo-400' },
              { label: 'Character Error Rate (CER)', val: `${werResult.cer}%`, color: 'text-purple-400' },
              { label: 'Operations Breakdown', val: `S:${werResult.substitutions} D:${werResult.deletions} I:${werResult.insertions}`, color: 'text-slate-300' }
            ].map((m, i) =>
              e(
                'div',
                { key: i, className: 'bg-slate-950 border border-slate-800 rounded-xl p-3 text-center' },
                e('div', { className: `text-lg font-bold ${m.color}` }, m.val),
                e('div', { className: 'text-[11px] text-slate-500 mt-0.5' }, m.label)
              )
            )
          ),
          e(
            'div',
            { className: 'bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-2' },
            e('div', { className: 'text-xs font-semibold text-slate-400' }, 'Word-Level Alignment Backtrace:'),
            e(
              'div',
              { className: 'flex flex-wrap gap-1.5' },
              (werResult.alignment || []).map((step, idx) => {
                let badgeClass = 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                let label = step.ref || step.hyp
                if (step.type === 'substitution') {
                  badgeClass = 'bg-amber-500/20 text-amber-300 border-amber-500/30'
                  label = `${step.ref} → ${step.hyp}`
                } else if (step.type === 'deletion') {
                  badgeClass = 'bg-rose-500/20 text-rose-300 border-rose-500/30'
                  label = `-${step.ref}`
                } else if (step.type === 'insertion') {
                  badgeClass = 'bg-blue-500/20 text-blue-300 border-blue-500/30'
                  label = `+${step.hyp}`
                }
                return e(
                  'span',
                  { key: idx, className: `px-2 py-0.5 rounded text-xs border font-mono ${badgeClass}` },
                  label
                )
              })
            )
          )
        )
      ),
      // WER Benchmark History
      werHistory.length > 0 && e(
        'div',
        { className: 'bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-3' },
        e(
          'div',
          { className: 'flex items-center justify-between' },
          e('h3', { className: 'text-sm font-semibold text-slate-200' }, 'Recent Saved WER Benchmark Runs'),
          e(
            'button',
            { onClick: clearWerLog, className: 'text-xs text-rose-400 hover:text-rose-300' },
            'Clear History'
          )
        ),
        e(
          'div',
          { className: 'space-y-2' },
          werHistory.map((item) =>
            e(
              'div',
              { key: item.id, className: 'p-3 rounded-xl bg-slate-950 border border-slate-800 flex flex-wrap items-center justify-between gap-2 text-xs' },
              e('div', { className: 'space-y-0.5 max-w-2xl' },
                e('div', { className: 'text-slate-300 truncate' }, `Ref: ${item.reference_text}`),
                e('div', { className: 'text-slate-500 truncate' }, `Hyp: ${item.hypothesis_text}`)
              ),
              e('div', { className: 'flex items-center gap-3 font-mono' },
                e('span', { className: 'text-amber-400' }, `WER ${item.wer}%`),
                e('span', { className: 'text-purple-400' }, `CER ${item.cer}%`)
              )
            )
          )
        )
      )
    )

    // ==================== TAB 4: ANALYTICS & MODEL INSPECTOR ====================
    const modelView = e(
      'div',
      { className: 'max-w-6xl mx-auto p-4 lg:p-6 space-y-6 w-full' },
      analytics && e(
        'div',
        { className: 'bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4' },
        e(
          'div',
          { className: 'flex items-center justify-between' },
          e('h2', { className: 'text-base font-bold text-white' }, 'Live Commercial Usage & Telemetry'),
          e('span', { className: 'text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 font-mono' }, `DB: ${analytics.databaseBackend}`)
        ),
        e(
          'div',
          { className: 'grid grid-cols-2 sm:grid-cols-4 gap-3' },
          [
            { label: 'Total Sessions', val: analytics.totalSessions ?? 0 },
            { label: 'Total Queries (Voice / Text)', val: `${analytics.total_queries ?? 0} (${analytics.voice_queries ?? 0}🎤 / ${analytics.text_queries ?? 0}⌨️)` },
            { label: 'Avg Inference Latency', val: analytics.avg_latency_ms ? `${analytics.avg_latency_ms} ms` : 'N/A' },
            { label: 'Feedback (👍 / 👎)', val: `${analytics.feedback?.positive ?? 0} / ${analytics.feedback?.negative ?? 0}` }
          ].map((m, i) =>
            e(
              'div',
              { key: i, className: 'bg-slate-950 border border-slate-800 rounded-xl p-3 text-center' },
              e('div', { className: 'text-lg font-bold text-emerald-400' }, m.val),
              e('div', { className: 'text-[11px] text-slate-500 mt-0.5' }, m.label)
            )
          )
        )
      ),
      modelInfo && e(
        'div',
        { className: 'bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4' },
        e('h2', { className: 'text-base font-bold text-white' }, 'Multi-Scale 1D TextCNN & 3-Step Recursive Introspection Architecture'),
        e(
          'div',
          { className: 'grid grid-cols-2 sm:grid-cols-4 gap-3' },
          [
            { label: 'TextCNN Train / Val Acc', val: `${((modelInfo.train_accuracy || 0) * 100).toFixed(1)}% / ${((modelInfo.cnn_val_accuracy || modelInfo.cv_accuracy_mean || 0) * 100).toFixed(1)}%` },
            { label: 'Introspection Policy / Critic', val: `${((modelInfo.introspection_policy_acc || 0) * 100).toFixed(1)}% / ${((modelInfo.introspection_critic_cal || 0) * 100).toFixed(1)}%` },
            { label: 'CNN + TF-IDF Vocabulary', val: `${modelInfo.cnn_vocab_size || 0} + ${modelInfo.vocab_size || 0}` },
            { label: 'Training Samples & Domains', val: `${modelInfo.augmented_samples || 0} (${modelInfo.num_classes || 0} classes)` }
          ].map((m, i) =>
            e(
              'div',
              { key: i, className: 'bg-slate-950 border border-slate-800 rounded-xl p-3 text-center' },
              e('div', { className: 'text-lg font-bold text-indigo-400' }, m.val),
              e('div', { className: 'text-[11px] text-slate-500 mt-0.5' }, m.label)
            )
          )
        ),
        e(
          'div',
          { className: 'space-y-3' },
          e('h3', { className: 'text-xs font-semibold uppercase tracking-wider text-slate-400' }, 'All Trained Multilingual Intent Classes'),
          e(
            'div',
            { className: 'grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3' },
            (modelInfo.intents || []).map((it) =>
              e(
                'div',
                {
                  key: it.tag,
                  onClick: () => {
                    setActiveTab('chat')
                    if (it.sample_follow_ups && it.sample_follow_ups[0]) {
                      sendMessage(it.sample_follow_ups[0], 'text')
                    }
                  },
                  className: 'p-3 rounded-xl bg-slate-950 hover:bg-slate-800 border border-slate-800 hover:border-indigo-500/40 cursor-pointer transition'
                },
                e(
                  'div',
                  { className: 'flex items-center justify-between' },
                  e('span', { className: 'font-mono text-xs text-indigo-400' }, it.tag),
                  e('span', { className: 'text-[10px] text-slate-500' }, `${it.pattern_count} patterns`)
                ),
                e('div', { className: 'text-xs font-semibold text-slate-200 mt-1' }, it.title)
              )
            )
          )
        )
      )
    )

    return e(
      'div',
      { className: 'min-h-screen flex flex-col bg-slate-950 text-slate-100' },
      headerNode,
      activeTab === 'chat'
        ? chatView
        : activeTab === 'knowledge'
        ? knowledgeView
        : activeTab === 'wer'
        ? werView
        : modelView
    )
  }

  const root = ReactDOM.createRoot(document.getElementById('root'))
  root.render(React.createElement(App))
})()
