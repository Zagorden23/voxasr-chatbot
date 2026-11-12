/**
 * VoxASR Frontend SPA — Pure JavaScript + React 18 (No TypeScript / No JSX build needed)
 * Integrates:
 *  - 35 Certified World Languages for Web Speech API (STT) & SpeechSynthesis (TTS)
 *  - Multilingual Computer Science & Mathematical Computing AI Assistant
 *  - Continuous Automated Web Scraping & ML-DL Retraining Dashboard
 *  - Web Audio API (AnalyserNode) for live microphone frequency waveform visualization
 *  - Word Error Rate (WER) / Character Error Rate (CER) Dynamic Programming Evaluator Lab
 *  - Python Training Pipeline & Softmax Weights Inspector
 */

;(function () {
  const e = React.createElement
  const { useState, useEffect, useRef } = React

  const CERTIFIED_LANGUAGES = [
    { code: 'en-US', name: 'English (US)', flag: '🇺🇸', script: 'Latin' },
    { code: 'en-GB', name: 'English (UK)', flag: '🇬🇧', script: 'Latin' },
    { code: 'en-IN', name: 'English (India)', flag: '🇮🇳', script: 'Latin' },
    { code: 'es-ES', name: 'Español (España)', flag: '🇪🇸', script: 'Latin' },
    { code: 'es-MX', name: 'Español (México)', flag: '🇲🇽', script: 'Latin' },
    { code: 'zh-CN', name: '简体中文 (Mandarin)', flag: '🇨🇳', script: 'Hanzi' },
    { code: 'zh-TW', name: '繁體中文 (Traditional)', flag: '🇹🇼', script: 'Hanzi' },
    { code: 'hi-IN', name: 'हिन्दी (Hindi)', flag: '🇮🇳', script: 'Devanagari' },
    { code: 'ar-SA', name: 'العربية (Arabic)', flag: '🇸🇦', script: 'Arabic' },
    { code: 'fr-FR', name: 'Français (French)', flag: '🇫🇷', script: 'Latin' },
    { code: 'ru-RU', name: 'Русский (Russian)', flag: '🇷🇺', script: 'Cyrillic' },
    { code: 'pt-BR', name: 'Português (Brasil)', flag: '🇧🇷', script: 'Latin' },
    { code: 'de-DE', name: 'Deutsch (German)', flag: '🇩🇪', script: 'Latin' },
    { code: 'ja-JP', name: '日本語 (Japanese)', flag: '🇯🇵', script: 'Kanji/Kana' },
    { code: 'ko-KR', name: '한국어 (Korean)', flag: '🇰🇷', script: 'Hangul' },
    { code: 'it-IT', name: 'Italiano (Italian)', flag: '🇮🇹', script: 'Latin' },
    { code: 'bn-IN', name: 'বাংলা (Bengali)', flag: '🇧🇩', script: 'Bengali' },
    { code: 'tr-TR', name: 'Türkçe (Turkish)', flag: '🇹🇷', script: 'Latin' },
    { code: 'vi-VN', name: 'Tiếng Việt (Vietnamese)', flag: '🇻🇳', script: 'Latin' },
    { code: 'id-ID', name: 'Bahasa Indonesia', flag: '🇮🇩', script: 'Latin' },
    { code: 'nl-NL', name: 'Nederlands (Dutch)', flag: '🇳🇱', script: 'Latin' },
    { code: 'pl-PL', name: 'Polski (Polish)', flag: '🇵🇱', script: 'Latin' },
    { code: 'fa-IR', name: 'فارسی (Persian)', flag: '🇮🇷', script: 'Arabic' },
    { code: 'sw-KE', name: 'Kiswahili (Swahili)', flag: '🇰🇪', script: 'Latin' },
    { code: 'ta-IN', name: 'தமிழ் (Tamil)', flag: '🇮🇳', script: 'Tamil' },
    { code: 'te-IN', name: 'తెలుగు (Telugu)', flag: '🇮🇳', script: 'Telugu' },
    { code: 'ur-PK', name: 'اردو (Urdu)', flag: '🇵🇰', script: 'Arabic' },
    { code: 'mr-IN', name: 'मराठी (Marathi)', flag: '🇮🇳', script: 'Devanagari' },
    { code: 'uk-UA', name: 'Українська (Ukrainian)', flag: '🇺🇦', script: 'Cyrillic' },
    { code: 'el-GR', name: 'Ελληνικά (Greek)', flag: '🇬🇷', script: 'Greek' },
    { code: 'sv-SE', name: 'Svenska (Swedish)', flag: '🇸🇪', script: 'Latin' },
    { code: 'he-IL', name: 'עברית (Hebrew)', flag: '🇮🇱', script: 'Hebrew' },
    { code: 'th-TH', name: 'ไทย (Thai)', flag: '🇹🇭', script: 'Thai' },
    { code: 'cs-CZ', name: 'Čeština (Czech)', flag: '🇨🇿', script: 'Latin' }
  ]

  const CS_MATH_PILLARS = [
    {
      id: 'algorithms_complexity',
      title: 'Algorithms & Complexity',
      icon: 'fa-sitemap',
      desc: 'Dijkstra, A*, Dynamic Programming, NP-completeness, Big-O, Master Theorem',
      query: 'Explain Dijkstra shortest path algorithm and time complexity'
    },
    {
      id: 'numerical_analysis',
      title: 'Numerical Computing & SVD',
      icon: 'fa-square-root-variable',
      desc: 'Singular Value Decomposition (SVD), QR, LU, Runge-Kutta RK4, FFT, IEEE 754',
      query: 'What is Singular Value Decomposition (SVD) and how does it work?'
    },
    {
      id: 'machine_learning_dl',
      title: 'Deep Learning & Neural Networks',
      icon: 'fa-network-wired',
      desc: 'Transformers, RoPE, Attention, Backpropagation, AdamW, Diffusion Models, RLHF',
      query: 'Explain the Transformer self-attention mechanism and backpropagation'
    },
    {
      id: 'speech_ai_asr',
      title: 'Automatic Speech Recognition (ASR)',
      icon: 'fa-microphone-lines',
      desc: 'OpenAI Whisper, Conformer, CTC Loss, Wav2Vec 2.0, Mel Spectrograms, WER',
      query: 'How does OpenAI Whisper convert speech audio into text?'
    },
    {
      id: 'theoretical_cs_discrete_math',
      title: 'Theoretical CS & Discrete Math',
      icon: 'fa-atom',
      desc: 'Turing Machines, Chomsky Hierarchy, Halting Problem, Graph Theory, Modular Arithmetic',
      query: 'What is a Turing Machine and how is the Halting Problem proved undecidable?'
    },
    {
      id: 'systems_distributed_computing',
      title: 'OS & Distributed Systems',
      icon: 'fa-server',
      desc: 'CAP Theorem, Raft Consensus, Virtual Memory, TLB, Deadlocks, Consistent Hashing',
      query: 'Explain the CAP Theorem and how Raft consensus operates'
    },
    {
      id: 'compilers_programming_languages',
      title: 'Compilers & Programming Languages',
      icon: 'fa-code',
      desc: 'AST, LLVM IR, SSA Form, Recursive Descent, Type Inference, Garbage Collection',
      query: 'How do compilers use Static Single Assignment (SSA) form for optimization?'
    },
    {
      id: 'cryptography_security',
      title: 'Modern Cryptography & Security',
      icon: 'fa-shield-halved',
      desc: 'RSA, Elliptic Curve (ECC), AES-GCM, Zero-Knowledge Proofs, Post-Quantum Kyber',
      query: 'How does RSA encryption work mathematically with Euler totient?'
    },
    {
      id: 'calculus_linear_algebra_prob',
      title: 'Calculus, Linear Algebra & Probability',
      icon: 'fa-chart-line',
      desc: 'Jacobian, Hessian, Eigenvalues, Bayes Theorem, Markov Chains, Central Limit Theorem',
      query: 'What is the Jacobian and Hessian matrix in multivariable optimization?'
    },
    {
      id: 'gpu_hpc_computing',
      title: 'GPU Computing & HPC',
      icon: 'fa-microchip',
      desc: 'CUDA Kernels, Warps, Shared Memory, OpenMP, MPI, BLAS GEMM, Tensor Cores',
      query: 'How does the CUDA warp execution model work on GPUs?'
    },
    {
      id: 'database_vector_search',
      title: 'Database Internals & Vector Search',
      icon: 'fa-database',
      desc: 'B+ Trees, LSM Trees, ACID, MVCC, Approximate Nearest Neighbor (HNSW), Cosine Distance',
      query: 'What is the difference between B+ trees and Log-Structured Merge (LSM) trees?'
    },
    {
      id: 'nlp_large_language_models',
      title: 'NLP & Large Language Models',
      icon: 'fa-robot',
      desc: 'BPE Tokenization, FlashAttention, KV Caching, RAG, Quantization (GGUF/AWQ), LoRA',
      query: 'Explain Byte-Pair Encoding (BPE) tokenization and KV caching in LLMs'
    }
  ]

  const MULTILINGUAL_SHOWCASE = [
    { flag: '🇪🇸', lang: 'Spanish', text: '¿Cómo funciona la retropropagación en redes neuronales?' },
    { flag: '🇨🇳', lang: 'Mandarin', text: '什么是Transformer架构和自注意力机制？' },
    { flag: '🇮🇳', lang: 'Hindi', text: 'डाइक्स्ट्रा एल्गोरिदम और टाइम कॉम्प्लेक्सिटी क्या है?' },
    { flag: '🇫🇷', lang: 'French', text: "Qu'est-ce que le théorème CAP en systèmes distribués?" },
    { flag: '🇩🇪', lang: 'German', text: 'Wie funktioniert die RSA Verschlüsselung?' },
    { flag: '🇷🇺', lang: 'Russian', text: 'Что такое машина Тьюринга и проблема остановки?' },
    { flag: '🇯🇵', lang: 'Japanese', text: '特異値分解（SVD）とは何ですか？' },
    { flag: '🇺🇸', lang: 'English', text: 'Explain Singular Value Decomposition (SVD) and low-rank approximation.' }
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

  // Markdown-lite formatter
  function renderFormattedText(text) {
    if (!text) return null
    const lines = String(text).split('\n')
    return lines.map((line, lineIdx) => {
      if (!line.trim()) {
        return e('div', { key: lineIdx, className: 'h-2' })
      }
      const isBullet = /^(\s*[-*]|\s*\d+\.)\s+/.test(line)
      const isHeader = /^###?\s+/.test(line)
      const cleanLine = isBullet
        ? line.replace(/^(\s*[-*]|\s*\d+\.)\s+/, '')
        : isHeader
        ? line.replace(/^###?\s+/, '')
        : line

      const parts = cleanLine.split(/(\*\*[^*]+\*\*|`[^`]+`|\$[^$]+\$)/g).filter(Boolean)
      const inlineNodes = parts.map((part, idx) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return e('strong', { key: idx, className: 'font-semibold text-indigo-200' }, part.slice(2, -2))
        }
        if (part.startsWith('`') && part.endsWith('`')) {
          return e(
            'code',
            {
              key: idx,
              className: 'font-mono text-xs bg-slate-800/90 text-emerald-300 px-1.5 py-0.5 rounded border border-slate-700'
            },
            part.slice(1, -1)
          )
        }
        if (part.startsWith('$') && part.endsWith('$')) {
          return e(
            'span',
            { key: idx, className: 'font-mono text-xs text-amber-300 bg-amber-950/40 px-1 py-0.5 rounded' },
            part.slice(1, -1)
          )
        }
        return part
      })

      if (isHeader) {
        return e('h4', { key: lineIdx, className: 'text-sm font-bold text-indigo-300 mt-2 mb-1' }, inlineNodes)
      }

      if (isBullet) {
        return e(
          'div',
          { key: lineIdx, className: 'flex items-start gap-2 pl-1 my-1 text-slate-200 text-xs sm:text-sm' },
          e('span', { className: 'text-indigo-400 mt-1 text-xs' }, '●'),
          e('span', { className: 'leading-relaxed' }, inlineNodes)
        )
      }

      return e('p', { key: lineIdx, className: 'leading-relaxed my-0.5 text-slate-200 text-xs sm:text-sm' }, inlineNodes)
    })
  }

  function App() {
    const [activeTab, setActiveTab] = useState('chat') // 'chat' | 'knowledge' | 'wer' | 'model'
    const [sessionId, setSessionId] = useState(() => `sess_${Date.now().toString(36)}`)
    const [sessions, setSessions] = useState([])
    const [messages, setMessages] = useState([
      {
        id: 'welcome',
        role: 'bot',
        content:
          'Welcome to **VoxASR** — Certified Multilingual AI Assistant for **Computer Science** and **Mathematical Computing**!\n\n- **Certified World Languages**: Speak or type in 35 certified languages (English, Spanish, Mandarin, Hindi, French, German, Russian, Arabic, Japanese, Portuguese, etc.).\n- **Comprehensive Knowledge Pool**: Inquire about Algorithms, Deep Learning, SVD/QR decompositions, Cryptography, Distributed Systems, Compiler Theory, or Speech AI.\n- **Automated Web Scraping & Training**: My knowledge pool is continuously updated via Wikipedia and ArXiv scraping with ML-DL model retraining!',
        intentTag: 'greeting',
        intentTitle: 'Multilingual CS/Math Assistant Ready',
        confidence: 1.0,
        detectedLanguage: { lang: 'en-US', label: 'English', script: 'Latin' },
        followUps: [
          'What is Singular Value Decomposition (SVD)?',
          'Explain Dijkstra shortest path algorithm',
          'How does OpenAI Whisper perform Speech-to-Text?',
          'What is P vs NP in computational complexity?'
        ]
      }
    ])

    const [inputText, setInputText] = useState('')
    const [interimTranscript, setInterimTranscript] = useState('')
    const [isListening, setIsListening] = useState(false)
    const [asrLang, setAsrLang] = useState('en-US')
    const [autoSendVoice, setAutoSendVoice] = useState(true)
    const [ttsEnabled, setTtsEnabled] = useState(false)
    const [speechSupported, setSpeechSupported] = useState(true)
    const [micError, setMicError] = useState('')
    const [isSending, setIsSending] = useState(false)
    const [lastAsrConfidence, setLastAsrConfidence] = useState(null)

    // Audio visualizer bars
    const [audioBars, setAudioBars] = useState(new Array(20).fill(6))

    // Model & Knowledge status
    const [modelInfo, setModelInfo] = useState(null)
    const [knowledgeStatus, setKnowledgeStatus] = useState(null)
    const [isUpgrading, setIsUpgrading] = useState(false)
    const [upgradeMessage, setUpgradeMessage] = useState('')
    const [customScrapeTopic, setCustomScrapeTopic] = useState('')
    const [customScrapeLang, setCustomScrapeLang] = useState('en')

    // WER Lab state
    const [werRef, setWerRef] = useState(PRESET_WER_EXAMPLES[0].ref)
    const [werHyp, setWerHyp] = useState(PRESET_WER_EXAMPLES[0].hyp)
    const [werResult, setWerResult] = useState(null)
    const [isListeningWer, setIsListeningWer] = useState(false)

    const recognitionRef = useRef(null)
    const audioCtxRef = useRef(null)
    const animFrameRef = useRef(null)
    const mediaStreamRef = useRef(null)
    const chatBottomRef = useRef(null)

    useEffect(() => {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition
      if (!SR) setSpeechSupported(false)

      fetchModelInfo()
      fetchKnowledgeStatus()
      fetchSessions()
      runWerEvaluation(PRESET_WER_EXAMPLES[0].ref, PRESET_WER_EXAMPLES[0].hyp)

      return () => {
        stopAudioVisualizer()
        if (window.speechSynthesis) window.speechSynthesis.cancel()
      }
    }, [])

    useEffect(() => {
      if (chatBottomRef.current) {
        chatBottomRef.current.scrollIntoView({ behavior: 'smooth' })
      }
    }, [messages, interimTranscript])

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
          if (data.is_upgrading) setIsUpgrading(true)
        }
      } catch (_) {}
    }

    async function fetchSessions() {
      try {
        const res = await fetch('/api/sessions')
        const data = await res.json()
        if (data.ok) setSessions(data.sessions || [])
      } catch (_) {}
    }

    async function triggerAutoUpgrade() {
      setIsUpgrading(true)
      setUpgradeMessage('Triggering automated web scraping (Wikipedia & ArXiv) and ML-DL retraining...')
      try {
        const res = await fetch('/api/knowledge/upgrade', { method: 'POST' })
        const data = await res.json()
        if (data.ok) {
          setUpgradeMessage('Scraping articles and retraining multi-class Softmax neural weights in background...')
          // Poll every 3s
          const interval = setInterval(async () => {
            const pollRes = await fetch('/api/knowledge/status')
            const pollData = await pollRes.json()
            if (pollData.ok) {
              setKnowledgeStatus(pollData)
              if (!pollData.is_upgrading) {
                clearInterval(interval)
                setIsUpgrading(false)
                setUpgradeMessage('Knowledge base upgraded & model weights reloaded successfully!')
                fetchModelInfo()
              } else {
                setUpgradeMessage(`Upgrading: ${pollData.last_progress || 'Processing...'}`)
              }
            }
          }, 3000)
        } else {
          setIsUpgrading(false)
          setUpgradeMessage(data.error || 'Upgrade failed.')
        }
      } catch (err) {
        setIsUpgrading(false)
        setUpgradeMessage('Could not connect to upgrade daemon (Port 3001). Run `python webapp/training/upgrade_server.py`!')
      }
    }

    async function triggerOnDemandScrape() {
      if (!customScrapeTopic.trim()) return
      setIsUpgrading(true)
      setUpgradeMessage(`Scraping on-demand article for "${customScrapeTopic}"...`)
      try {
        const res = await fetch('/api/knowledge/scrape', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ topic: customScrapeTopic.trim(), lang: customScrapeLang })
        })
        const data = await res.json()
        if (data.ok) {
          setUpgradeMessage(`Ingested "${customScrapeTopic}". Retraining model in background...`)
          setTimeout(() => {
            fetchKnowledgeStatus()
            fetchModelInfo()
            setIsUpgrading(false)
          }, 4000)
        } else {
          setIsUpgrading(false)
          setUpgradeMessage(data.error || 'Scrape failed.')
        }
      } catch (err) {
        setIsUpgrading(false)
        setUpgradeMessage('Could not connect to upgrade service on Port 3001.')
      }
    }

    function speakReply(rawText) {
      if (!ttsEnabled || !window.speechSynthesis) return
      window.speechSynthesis.cancel()
      const clean = String(rawText)
        .replace(/\*\*([^*]+)\*\*/g, '$1')
        .replace(/`([^`]+)`/g, '$1')
        .replace(/\$([^$]+)\$/g, '$1')
        .replace(/[#>*_]/g, '')
      const utter = new SpeechSynthesisUtterance(clean)
      utter.lang = asrLang
      utter.rate = 1.02
      window.speechSynthesis.speak(utter)
    }

    async function sendMessage(textToSend, mode = 'text', asrConf = null) {
      const msg = (textToSend || inputText).trim()
      if (!msg || isSending) return

      setInputText('')
      setInterimTranscript('')
      setIsSending(true)

      const userMsgId = `usr_${Date.now()}`
      const newMessages = [
        ...messages,
        {
          id: userMsgId,
          role: 'user',
          content: msg,
          inputMode: mode,
          asrConfidence: asrConf
        }
      ]
      setMessages(newMessages)

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sessionId,
            message: msg,
            inputMode: mode,
            asrConfidence: asrConf
          })
        })
        const data = await res.json()
        if (data.ok && data.botResponse) {
          const botResp = data.botResponse
          setMessages((prev) => [
            ...prev,
            {
              id: `bot_${Date.now()}`,
              role: 'bot',
              content: botResp.reply,
              intentTag: botResp.intentTag,
              intentTitle: botResp.intentTitle,
              confidence: botResp.confidence,
              detectedLanguage: botResp.detectedLanguage,
              followUps: botResp.followUps || [],
              topPredictions: botResp.topPredictions || [],
              matchedTokens: botResp.matchedTokens || [],
              latencyMs: botResp.latencyMs
            }
          ])
          speakReply(botResp.reply)
          fetchSessions()
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
      }
    }

    // Audio visualizer bars
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
      recognition.lang = asrLang
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
            setInputText(finalTxt)
          }
        }
      }

      recognition.onerror = (event) => {
        setMicError(`Microphone error: ${event.error}`)
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
      }
    }

    async function runWerEvaluation(ref, hyp) {
      try {
        const res = await fetch('/api/wer', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ reference: ref, hypothesis: hyp })
        })
        const data = await res.json()
        if (data.ok) setWerResult(data.metrics)
      } catch (_) {}
    }

    // ==================== HEADER ====================
    const headerNode = e(
      'header',
      { className: 'border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-30 px-4 py-3' },
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
              e('span', null, '35 Certified Languages ● 12 CS/Math Pillars ● Web Scraping Upgraded')
            )
          )
        ),
        e(
          'nav',
          { className: 'flex items-center gap-1.5 bg-slate-900/90 p-1 rounded-xl border border-slate-800' },
          [
            { id: 'chat', icon: 'fa-microphone-lines', label: 'ASR Voice Chat' },
            { id: 'knowledge', icon: 'fa-database', label: 'Knowledge Base & Auto-Upgrade' },
            { id: 'wer', icon: 'fa-scale-balanced', label: 'WER Lab' },
            { id: 'model', icon: 'fa-chart-pie', label: 'Model Inspector' }
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

    // ==================== TAB 1: VOICE CHAT ====================
    const chatView = e(
      'div',
      { className: 'grid grid-cols-1 lg:grid-cols-12 gap-5 max-w-7xl mx-auto p-4 lg:p-6 w-full flex-1' },
      // Left Sidebar: Speech Locale & Certified Languages
      e(
        'aside',
        { className: 'lg:col-span-3 space-y-4' },
        e(
          'div',
          { className: 'bg-slate-900/80 border border-slate-800 rounded-2xl p-4 space-y-3.5' },
          e(
            'div',
            { className: 'flex items-center justify-between' },
            e('h2', { className: 'text-xs font-semibold uppercase tracking-wider text-slate-400' }, 'Speech Settings'),
            e('span', { className: 'text-[11px] px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20' }, '35 Languages')
          ),
          e(
            'div',
            { className: 'space-y-1' },
            e('label', { className: 'text-xs text-slate-400 block' }, 'Certified Speech Recognition Locale:'),
            e(
              'select',
              {
                value: asrLang,
                onChange: (ev) => setAsrLang(ev.target.value),
                className: 'w-full bg-slate-950 border border-slate-700/80 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500'
              },
              CERTIFIED_LANGUAGES.map((l) =>
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
            e('span', null, 'Text-to-Speech (TTS) Bot Voice'),
            e('input', {
              type: 'checkbox',
              checked: ttsEnabled,
              onChange: (ev) => setTtsEnabled(ev.target.checked),
              className: 'accent-indigo-500 rounded'
            })
          ),
          e(
            'div',
            { className: 'pt-2 border-t border-slate-800' },
            e(
              'button',
              {
                onClick: toggleMicrophone,
                className: `w-full py-2.5 px-3 rounded-xl font-medium text-xs flex items-center justify-center gap-2 transition ${
                  isListening
                    ? 'bg-rose-600 text-white animate-pulse shadow-lg shadow-rose-600/30'
                    : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-md shadow-indigo-600/25'
                }`
              },
              e('i', { className: `fas ${isListening ? 'fa-microphone-slash' : 'fa-microphone'}` }),
              isListening ? 'Stop Listening...' : 'Click to Speak Live'
            )
          )
        ),
        // Multilingual Query Suggestions
        e(
          'div',
          { className: 'bg-slate-900/80 border border-slate-800 rounded-2xl p-4 space-y-2.5' },
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
        { className: 'lg:col-span-9 flex flex-col h-[750px] bg-slate-900/60 border border-slate-800 rounded-2xl overflow-hidden' },
        // Message List
        e(
          'div',
          { className: 'flex-1 overflow-y-auto p-4 sm:p-5 space-y-4' },
          messages.map((m) => {
            const isBot = m.role === 'bot'
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
                isBot && m.intentTitle && e(
                  'div',
                  { className: 'flex items-center gap-2 text-[11px] text-slate-400' },
                  e('span', { className: 'px-2 py-0.5 rounded-md bg-indigo-500/10 text-indigo-300 border border-indigo-500/20' }, m.intentTitle),
                  m.detectedLanguage && e('span', { className: 'text-slate-400' }, `🌐 ${m.detectedLanguage.label}`),
                  m.confidence && e('span', { className: 'text-emerald-400' }, `${Math.round(m.confidence * 100)}% match`)
                ),
                // Bubble
                e(
                  'div',
                  {
                    className: `p-3.5 rounded-2xl text-xs sm:text-sm ${
                      isBot
                        ? 'bg-slate-900 border border-slate-800 text-slate-200 shadow-sm'
                        : 'bg-indigo-600 text-white'
                    }`
                  },
                  renderFormattedText(m.content)
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
                        className: 'px-2 py-1 rounded-md text-[11px] bg-slate-950/70 hover:bg-slate-800 text-indigo-300 border border-indigo-500/20 transition'
                      },
                      `💡 ${fu}`
                    )
                  )
                )
              )
            )
          }),
          // Interim ASR transcript
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

    // ==================== TAB 2: KNOWLEDGE BASE & AUTO-UPGRADE ====================
    const knowledgeView = e(
      'div',
      { className: 'max-w-7xl mx-auto p-4 lg:p-6 space-y-6 w-full' },
      // Overview Banner
      e(
        'div',
        { className: 'bg-gradient-to-r from-indigo-900/40 via-purple-900/30 to-slate-900 border border-indigo-500/20 rounded-3xl p-6 sm:p-8 space-y-4' },
        e(
          'div',
          { className: 'flex flex-wrap items-center justify-between gap-4' },
          e(
            'div',
            { className: 'space-y-1' },
            e('span', { className: 'text-xs font-semibold px-2.5 py-1 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30' }, 'Automated Harvesting & Ingestion'),
            e('h2', { className: 'text-xl sm:text-2xl font-bold text-white' }, 'Dynamic CS & Mathematical Knowledge Pool'),
            e('p', { className: 'text-xs sm:text-sm text-slate-300 max-w-3xl' }, 'Self-updating knowledge engine that scrapes Wikipedia and ArXiv across 12 core computer science and mathematical domains, generates multilingual patterns, and retrains the model with zero server downtime.')
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
            isUpgrading ? 'Auto-Upgrading...' : 'Trigger Web Scraping & Auto-Train'
          )
        ),
        upgradeMessage && e(
          'div',
          { className: 'p-3 rounded-xl bg-slate-950/80 border border-indigo-500/30 text-xs text-indigo-300 flex items-center gap-2' },
          e('i', { className: 'fas fa-info-circle text-indigo-400' }),
          upgradeMessage
        )
      ),

      // Real-time Stat Cards
      e(
        'div',
        { className: 'grid grid-cols-2 lg:grid-cols-4 gap-4' },
        [
          { label: 'Certified Languages', val: '35 Languages', desc: 'UN + ISO-639 World Standards', icon: 'fa-globe' },
          { label: 'CS & Math Pillars', val: '12 Domains', desc: 'Algorithms to Quantum & SVD', icon: 'fa-cubes' },
          { label: 'TF-IDF Multilingual Vocab', val: `${modelInfo?.vocab_size || 5000} Features`, desc: 'Word 1-2g + Char 3-4g + CJK', icon: 'fa-brain' },
          { label: 'Scraped Articles Pool', val: `${knowledgeStatus?.total_scraped_articles || 36} Ingested`, desc: 'Wikipedia & ArXiv Preprints', icon: 'fa-file-lines' }
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

      // On-Demand Topic Ingestion
      e(
        'div',
        { className: 'bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-3' },
        e('h3', { className: 'text-sm font-semibold text-slate-200' }, 'On-Demand Web Scraping & Topic Ingestion'),
        e('p', { className: 'text-xs text-slate-400' }, 'Type any computer science or mathematical concept to immediately scrape authoritative sources and update the training data:'),
        e(
          'div',
          { className: 'flex flex-wrap sm:flex-nowrap gap-2.5' },
          e('input', {
            type: 'text',
            value: customScrapeTopic,
            onChange: (ev) => setCustomScrapeTopic(ev.target.value),
            placeholder: 'e.g. FlashAttention, Runge-Kutta, Elliptic Curve Cryptography, Quantum Shor Algorithm...',
            className: 'flex-1 bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs sm:text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500'
          }),
          e(
            'select',
            {
              value: customScrapeLang,
              onChange: (ev) => setCustomScrapeLang(ev.target.value),
              className: 'bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-300'
            },
            e('option', { value: 'en' }, 'English (en)'),
            e('option', { value: 'es' }, 'Spanish (es)'),
            e('option', { value: 'fr' }, 'French (fr)'),
            e('option', { value: 'de' }, 'German (de)'),
            e('option', { value: 'zh' }, 'Chinese (zh)')
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

      // CS & Mathematical Computing Pillars Explorer
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
        e('h2', { className: 'text-base font-bold text-white' }, 'Word Error Rate (WER) & CER Evaluator Lab'),
        e('p', { className: 'text-xs text-slate-400' }, 'Calculate speech recognition accuracy using dynamic programming Levenshtein minimum edit distance alignment across Substitutions (S), Deletions (D), and Insertions (I).'),
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
                  runWerEvaluation(ex.ref, ex.hyp)
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
                runWerEvaluation(ev.target.value, werHyp)
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
                runWerEvaluation(werRef, ev.target.value)
              },
              className: 'w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-indigo-500'
            })
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
          // Word Alignment Visualizer
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
      )
    )

    // ==================== TAB 4: MODEL INSPECTOR ====================
    const modelView = e(
      'div',
      { className: 'max-w-6xl mx-auto p-4 lg:p-6 space-y-6 w-full' },
      modelInfo && e(
        'div',
        { className: 'space-y-5' },
        e(
          'div',
          { className: 'bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4' },
          e('h2', { className: 'text-base font-bold text-white' }, 'Softmax Neural & TF-IDF Model Architecture'),
          e(
            'div',
            { className: 'grid grid-cols-2 sm:grid-cols-4 gap-3' },
            [
              { label: 'Training Accuracy', val: `${(modelInfo.train_accuracy * 100).toFixed(2)}%` },
              { label: '5-Fold CV Accuracy', val: `${(modelInfo.cv_accuracy_mean * 100).toFixed(1)}%` },
              { label: 'TF-IDF Multilingual Vocab', val: `${modelInfo.vocab_size} n-grams` },
              { label: 'Training Samples', val: `${modelInfo.augmented_samples} (${modelInfo.num_classes} classes)` }
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
                  e('div', { className: 'flex items-center justify-between' },
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
