/**
 * VoxASR Frontend SPA — Pure JavaScript + React 18 (No TypeScript / No JSX build needed)
 * Integrates:
 *  - Web Speech API (SpeechRecognition) for real-time Speech-to-Text (ASR)
 *  - Web Audio API (AnalyserNode) for live microphone frequency waveform visualization
 *  - SpeechSynthesis API for Text-to-Speech (TTS) spoken chatbot replies
 *  - WER / CER Levenshtein Alignment Evaluator Lab
 *  - Python Training Pipeline & Softmax Weights Inspector
 */

;(function () {
  const e = React.createElement
  const { useState, useEffect, useRef } = React

  // Simple Markdown-lite formatter for bold, inline code, and bullet lists
  function renderFormattedText(text) {
    if (!text) return null
    const lines = String(text).split('\n')
    return lines.map((line, lineIdx) => {
      if (!line.trim()) {
        return e('div', { key: lineIdx, className: 'h-2' })
      }
      const isBullet = /^(\s*[-*]|\s*\d+\.)\s+/.test(line)
      const cleanLine = isBullet ? line.replace(/^(\s*[-*]|\s*\d+\.)\s+/, '') : line

      // Parse **bold** and `code`
      const parts = cleanLine.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean)
      const inlineNodes = parts.map((part, idx) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return e(
            'strong',
            { key: idx, className: 'font-semibold text-indigo-200' },
            part.slice(2, -2)
          )
        }
        if (part.startsWith('`') && part.endsWith('`')) {
          return e(
            'code',
            {
              key: idx,
              className:
                'font-mono-code text-xs bg-slate-800/90 text-emerald-300 px-1.5 py-0.5 rounded border border-slate-700'
            },
            part.slice(1, -1)
          )
        }
        return part
      })

      if (isBullet) {
        return e(
          'div',
          { key: lineIdx, className: 'flex items-start gap-2 pl-1 my-1' },
          e('span', { className: 'text-indigo-400 mt-1 text-xs' }, '●'),
          e('span', { className: 'leading-relaxed' }, inlineNodes)
        )
      }

      return e('p', { key: lineIdx, className: 'leading-relaxed my-0.5' }, inlineNodes)
    })
  }

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
      label: 'Noisy Far-Field Insertion & Deletion',
      ref: 'please activate voice activity detection before running the decoder',
      hyp: 'uh please activate voice detection before running running decoder'
    }
  ]

  function App() {
    const [activeTab, setActiveTab] = useState('chat') // 'chat' | 'wer' | 'model'
    const [sessionId, setSessionId] = useState(() => `sess_${Date.now().toString(36)}`)
    const [sessions, setSessions] = useState([])
    const [messages, setMessages] = useState([
      {
        id: 'welcome',
        role: 'bot',
        content:
          'Welcome to **VoxASR**! My intent neural classifier was trained in **Python** (`training/train_chatbot.py`) and runs natively in **JavaScript**.\n\n- Click the **Microphone** button to speak live via Automatic Speech Recognition (`Web Speech API`).\n- Or click a quick topic below to explore **Whisper**, **Conformer/CTC**, **MFCCs**, **VAD**, and **Word Error Rate (WER)**!',
        intentTag: 'greeting',
        intentTitle: 'System Ready',
        confidence: 1.0,
        followUps: [
          'What is Automatic Speech Recognition?',
          'How does OpenAI Whisper work?',
          'Compare CTC, RNN-T, and Attention models',
          'How do you calculate Word Error Rate?'
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

    // Model info state
    const [modelInfo, setModelInfo] = useState(null)

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
      if (!SR) {
        setSpeechSupported(false)
      }
      fetchModelInfo()
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

    async function fetchSessions() {
      try {
        const res = await fetch('/api/sessions')
        const data = await res.json()
        if (data.ok) setSessions(data.sessions || [])
      } catch (_) {}
    }

    async function loadSession(id) {
      try {
        const res = await fetch(`/api/sessions/${id}`)
        const data = await res.json()
        if (data.ok && data.messages && data.messages.length > 0) {
          setSessionId(id)
          setMessages(
            data.messages.map((m) => ({
              id: m.id,
              role: m.role,
              content: m.content,
              inputMode: m.input_mode,
              asrConfidence: m.asr_confidence,
              intentTag: m.intent_tag,
              confidence: m.intent_confidence
            }))
          )
          setActiveTab('chat')
        }
      } catch (_) {}
    }

    function startNewSession() {
      const freshId = `sess_${Date.now().toString(36)}`
      setSessionId(freshId)
      setMessages([
        {
          id: 'welcome_' + freshId,
          role: 'bot',
          content:
            'Started a fresh ASR session! Click the **Microphone** to dictate speech or type an ASR question below.',
          intentTag: 'greeting',
          intentTitle: 'New Session',
          confidence: 1.0,
          followUps: [
            'What are MFCCs and Log-Mel Spectrograms?',
            'How does Wav2Vec 2.0 learn from raw audio?',
            'What is Voice Activity Detection (VAD)?'
          ]
        }
      ])
    }

    async function deleteSession(id, ev) {
      ev.stopPropagation()
      try {
        await fetch(`/api/sessions/${id}`, { method: 'DELETE' })
        fetchSessions()
        if (id === sessionId) startNewSession()
      } catch (_) {}
    }

    // Speak bot response using browser TTS
    function speakReply(rawText) {
      if (!ttsEnabled || !window.speechSynthesis) return
      window.speechSynthesis.cancel()
      const clean = String(rawText)
        .replace(/\*\*([^*]+)\*\*/g, '$1')
        .replace(/`([^`]+)`/g, '$1')
        .replace(/[#>*_]/g, '')
      const utter = new SpeechSynthesisUtterance(clean)
      utter.lang = asrLang
      utter.rate = 1.04
      window.speechSynthesis.speak(utter)
    }

    // Live Web Audio API frequency bars
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
      } catch (_) {
        // Fallback animated bars if getUserMedia stream is held exclusively by SpeechRecognition
      }
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

    // Toggle Live Speech-to-Text for Chat
    function toggleMicrophone() {
      setMicError('')
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition
      if (!SR) {
        setMicError(
          'Your browser does not support Web Speech API (try Chrome or Edge). You can still type queries!'
        )
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
      recognition.maxAlternatives = 1

      let finalCaptured = ''
      let capturedConf = null

      recognition.onstart = () => {
        setIsListening(true)
        setInterimTranscript('')
        startAudioVisualizer()
      }

      recognition.onresult = (event) => {
        let interim = ''
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript
          if (event.results[i].isFinal) {
            finalCaptured += transcript + ' '
            capturedConf = event.results[i][0].confidence || 0.94
          } else {
            interim += transcript
          }
        }
        setInterimTranscript(interim)
        if (finalCaptured.trim()) {
          setInputText(finalCaptured.trim())
          setLastAsrConfidence(capturedConf)
        }
      }

      recognition.onerror = (event) => {
        if (event.error !== 'no-speech') {
          setMicError(`Microphone ASR notice: ${event.error}`)
        }
        setIsListening(false)
        stopAudioVisualizer()
      }

      recognition.onend = () => {
        setIsListening(false)
        setInterimTranscript('')
        stopAudioVisualizer()
        const cleanFinal = finalCaptured.trim()
        if (cleanFinal && autoSendVoice) {
          sendMessage(cleanFinal, 'voice', capturedConf)
          setInputText('')
        }
      }

      recognitionRef.current = recognition
      recognition.start()
    }

    // Dictate directly into WER Hypothesis box to test live browser ASR accuracy!
    function dictateIntoWerHypothesis() {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition
      if (!SR) {
        setMicError('Web Speech API requires Chrome or Edge.')
        return
      }
      if (isListeningWer && recognitionRef.current) {
        recognitionRef.current.stop()
        setIsListeningWer(false)
        return
      }
      const rec = new SR()
      rec.lang = asrLang
      rec.interimResults = true
      rec.continuous = false
      let finalHyp = ''

      rec.onstart = () => setIsListeningWer(true)
      rec.onresult = (ev) => {
        let interim = ''
        for (let i = ev.resultIndex; i < ev.results.length; i++) {
          const t = ev.results[i][0].transcript
          if (ev.results[i].isFinal) finalHyp += t + ' '
          else interim += t
        }
        setWerHyp((finalHyp + interim).trim())
      }
      rec.onend = () => {
        setIsListeningWer(false)
        const spoken = finalHyp.trim()
        if (spoken) {
          setWerHyp(spoken)
          runWerEvaluation(werRef, spoken)
        }
      }
      rec.onerror = () => setIsListeningWer(false)
      recognitionRef.current = rec
      rec.start()
    }

    async function sendMessage(overrideText, mode = 'text', asrConf = null) {
      const textToSend = (overrideText !== undefined ? overrideText : inputText).trim()
      if (!textToSend || isSending) return

      const effectiveMode =
        overrideText !== undefined ? mode : lastAsrConfidence ? 'voice' : 'text'
      const effectiveConf =
        asrConf !== null ? asrConf : effectiveMode === 'voice' ? lastAsrConfidence : null

      setInputText('')
      setLastAsrConfidence(null)
      setIsSending(true)

      const userMsg = {
        id: `u_${Date.now()}`,
        role: 'user',
        content: textToSend,
        inputMode: effectiveMode,
        asrConfidence: effectiveConf
      }
      setMessages((prev) => [...prev, userMsg])

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sessionId,
            message: textToSend,
            inputMode: effectiveMode,
            asrConfidence: effectiveConf
          })
        })
        const data = await res.json()
        if (data.ok && data.botResponse) {
          const bot = data.botResponse
          setMessages((prev) => [
            ...prev,
            {
              id: `b_${Date.now()}`,
              role: 'bot',
              content: bot.reply,
              intentTag: bot.intentTag,
              intentTitle: bot.intentTitle,
              confidence: bot.confidence,
              topPredictions: bot.topPredictions,
              matchedTokens: bot.matchedTokens,
              latencyMs: bot.latencyMs,
              followUps: bot.followUps
            }
          ])
          speakReply(bot.reply)
          fetchSessions()
        }
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          {
            id: `err_${Date.now()}`,
            role: 'bot',
            content: `Connection error: ${err.message}`,
            intentTag: 'error',
            confidence: 0
          }
        ])
      } finally {
        setIsSending(false)
      }
    }

    async function runWerEvaluation(refStr, hypStr) {
      try {
        const res = await fetch('/api/wer', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            reference: refStr !== undefined ? refStr : werRef,
            hypothesis: hypStr !== undefined ? hypStr : werHyp
          })
        })
        const data = await res.json()
        if (data.ok) setWerResult(data.metrics)
      } catch (_) {}
    }

    // ==================== HEADER ====================
    const headerNode = e(
      'header',
      {
        id: 'app-header',
        className:
          'glass-panel sticky top-0 z-30 border-b border-slate-800/80 px-4 lg:px-6 py-3 flex flex-wrap items-center justify-between gap-3'
      },
      e(
        'div',
        { className: 'flex items-center gap-3' },
        e(
          'div',
          {
            className:
              'w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-500/25'
          },
          e('i', { className: 'fas fa-wave-square text-white text-lg' })
        ),
        e(
          'div',
          null,
          e(
            'div',
            { className: 'flex items-center gap-2' },
            e('h1', { className: 'text-lg font-bold tracking-tight text-white' }, 'VoxASR'),
            e(
              'span',
              {
                className:
                  'text-[11px] font-medium px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/30'
              },
              'Python Trained • Pure JS'
            )
          ),
          e(
            'p',
            { className: 'text-xs text-slate-400' },
            'Automatic Speech Recognition Voice Chatbot & WER Lab'
          )
        )
      ),
      e(
        'nav',
        { id: 'main-tabs', className: 'flex items-center gap-1.5 bg-slate-900/90 p-1 rounded-xl border border-slate-800' },
        [
          { id: 'chat', icon: 'fa-microphone-lines', label: 'ASR Voice Chat' },
          { id: 'wer', icon: 'fa-scale-balanced', label: 'WER Evaluator' },
          { id: 'model', icon: 'fa-brain', label: 'Python Model' }
        ].map((tab) =>
          e(
            'button',
            {
              key: tab.id,
              id: `tab-${tab.id}`,
              onClick: () => setActiveTab(tab.id),
              className: `px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2 ${
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

    // ==================== TAB 1: VOICE CHAT ====================
    const chatView = e(
      'div',
      { className: 'grid grid-cols-1 lg:grid-cols-12 gap-5 max-w-7xl mx-auto p-4 lg:p-6' },
      // Left Sidebar: ASR Controls & Session History
      e(
        'aside',
        { id: 'asr-sidebar', className: 'lg:col-span-3 space-y-4' },
        // Live ASR Settings Card
        e(
          'section',
          { className: 'glass-panel rounded-2xl p-4 space-y-3.5' },
          e(
            'div',
            { className: 'flex items-center justify-between' },
            e(
              'h2',
              { className: 'text-xs font-semibold uppercase tracking-wider text-slate-400' },
              'Speech Controls'
            ),
            e(
              'span',
              {
                className: `text-[11px] px-2 py-0.5 rounded-full ${
                  speechSupported
                    ? 'bg-indigo-500/15 text-indigo-300'
                    : 'bg-amber-500/15 text-amber-300'
                }`
              },
              speechSupported ? 'Web Speech Ready' : 'Text Mode'
            )
          ),
          e(
            'div',
            { className: 'space-y-1' },
            e('label', { className: 'text-xs text-slate-400 block' }, 'Recognition Locale'),
            e(
              'select',
              {
                id: 'asr-lang-select',
                value: asrLang,
                onChange: (ev) => setAsrLang(ev.target.value),
                className:
                  'w-full bg-slate-900 border border-slate-700/80 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500'
              },
              e('option', { value: 'en-US' }, 'English (US) — en-US'),
              e('option', { value: 'en-GB' }, 'English (UK) — en-GB'),
              e('option', { value: 'en-IN' }, 'English (India) — en-IN')
            )
          ),
          e(
            'label',
            { className: 'flex items-center justify-between text-xs text-slate-300 cursor-pointer pt-1' },
            e('span', null, 'Auto-send after speaking'),
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
            e('span', null, 'Speak bot replies (TTS)'),
            e('input', {
              type: 'checkbox',
              checked: ttsEnabled,
              onChange: (ev) => {
                const next = ev.target.checked
                setTtsEnabled(next)
                if (!next && window.speechSynthesis) window.speechSynthesis.cancel()
              },
              className: 'accent-indigo-500 rounded'
            })
          ),
          // Live Audio Spectrum Bars
          e(
            'div',
            { className: 'pt-2 border-t border-slate-800/80' },
            e(
              'div',
              { className: 'flex items-center justify-between text-[11px] text-slate-400 mb-1.5' },
              e('span', null, isListening ? 'Listening to Mic...' : 'Mic Spectrum Idle'),
              isListening &&
                e('span', { className: 'text-rose-400 font-medium animate-pulse' }, '● LIVE')
            ),
            e(
              'div',
              {
                className:
                  'h-12 bg-slate-900/90 rounded-xl border border-slate-800 px-3 flex items-center justify-between gap-1'
              },
              audioBars.map((h, i) =>
                e('div', {
                  key: i,
                  style: { height: `${h}px` },
                  className: `w-1.5 rounded-full transition-all duration-75 ${
                    isListening
                      ? 'bg-gradient-to-t from-indigo-500 to-rose-400'
                      : 'bg-slate-700/60'
                  }`
                })
              )
            )
          )
        ),

        // Saved D1 Sessions Card
        e(
          'section',
          { className: 'glass-panel rounded-2xl p-4 space-y-3' },
          e(
            'div',
            { className: 'flex items-center justify-between' },
            e(
              'h2',
              { className: 'text-xs font-semibold uppercase tracking-wider text-slate-400' },
              'Chat History'
            ),
            e(
              'button',
              {
                id: 'new-session-btn',
                onClick: startNewSession,
                className:
                  'text-xs text-indigo-400 hover:text-indigo-300 font-medium flex items-center gap-1'
              },
              e('i', { className: 'fas fa-plus text-[10px]' }),
              'New'
            )
          ),
          sessions.length === 0
            ? e(
                'p',
                { className: 'text-xs text-slate-500 py-2' },
                'Ask a question to save your first ASR session.'
              )
            : e(
                'div',
                { className: 'space-y-1.5 max-h-56 overflow-y-auto pr-1' },
                sessions.map((s) =>
                  e(
                    'div',
                    {
                      key: s.id,
                      onClick: () => loadSession(s.id),
                      className: `group flex items-center justify-between px-2.5 py-2 rounded-lg text-xs cursor-pointer transition ${
                        s.id === sessionId
                          ? 'bg-indigo-600/20 border border-indigo-500/40 text-indigo-200'
                          : 'hover:bg-slate-800/70 text-slate-300'
                      }`
                    },
                    e(
                      'span',
                      { className: 'truncate pr-2' },
                      s.title || 'Voice ASR Session'
                    ),
                    e(
                      'button',
                      {
                        onClick: (ev) => deleteSession(s.id, ev),
                        title: 'Delete session',
                        className:
                          'opacity-0 group-hover:opacity-100 text-slate-500 hover:text-rose-400 transition'
                      },
                      e('i', { className: 'fas fa-trash-can text-[11px]' })
                    )
                  )
                )
              )
        )
      ),

      // Right Main Chat Panel
      e(
        'section',
        {
          id: 'chat-panel',
          className:
            'lg:col-span-9 glass-panel rounded-2xl flex flex-col h-[calc(100vh-8.5rem)] min-h-[540px]'
        },
        // Messages Stream
        e(
          'div',
          { className: 'flex-1 overflow-y-auto p-4 lg:p-6 space-y-5' },
          messages.map((msg) => {
            const isUser = msg.role === 'user'
            return e(
              'article',
              {
                key: msg.id,
                className: `flex flex-col ${isUser ? 'items-end' : 'items-start'}`
              },
              e(
                'div',
                {
                  className: `max-w-2xl rounded-2xl px-4 py-3 text-sm shadow-sm ${
                    isUser
                      ? 'bg-indigo-600 text-white rounded-br-sm'
                      : 'bg-slate-900/95 border border-slate-800 text-slate-200 rounded-bl-sm'
                  }`
                },
                // Top metadata badge for voice input or bot Python intent classification
                !isUser &&
                  msg.intentTag &&
                  e(
                    'div',
                    {
                      className:
                        'flex flex-wrap items-center gap-2 mb-2 pb-1.5 border-b border-slate-800/90 text-[11px]'
                    },
                    e(
                      'span',
                      { className: 'font-mono-code text-indigo-400 font-medium' },
                      `intent: ${msg.intentTag}`
                    ),
                    typeof msg.confidence === 'number' &&
                      e(
                        'span',
                        {
                          className:
                            'px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/20'
                        },
                        `${Math.round(msg.confidence * 100)}% conf`
                      ),
                    typeof msg.latencyMs === 'number' &&
                      e('span', { className: 'text-slate-500' }, `${msg.latencyMs} ms`)
                  ),
                isUser &&
                  msg.inputMode === 'voice' &&
                  e(
                    'div',
                    {
                      className:
                        'flex items-center gap-1.5 text-[11px] text-indigo-200 mb-1 opacity-90'
                    },
                    e('i', { className: 'fas fa-microphone text-[10px]' }),
                    e('span', null, 'Transcribed via ASR'),
                    msg.asrConfidence &&
                      e('span', null, `(${Math.round(msg.asrConfidence * 100)}%)`)
                  ),
                e('div', { className: 'space-y-1' }, renderFormattedText(msg.content))
              ),
              // Suggested Follow-up Chips
              !isUser &&
                msg.followUps &&
                msg.followUps.length > 0 &&
                e(
                  'div',
                  { className: 'flex flex-wrap gap-1.5 mt-2 max-w-2xl' },
                  msg.followUps.map((q, idx) =>
                    e(
                      'button',
                      {
                        key: idx,
                        onClick: () => sendMessage(q, 'text'),
                        className:
                          'text-xs bg-slate-900 hover:bg-indigo-950/70 text-slate-300 hover:text-indigo-200 border border-slate-800 hover:border-indigo-500/40 rounded-full px-3 py-1 transition'
                      },
                      q
                    )
                  )
                )
            )
          }),
          // Interim live speech bubble
          interimTranscript &&
            e(
              'div',
              { className: 'flex justify-end' },
              e(
                'div',
                {
                  className:
                    'bg-rose-500/15 border border-rose-500/40 text-rose-200 rounded-2xl px-4 py-2.5 text-sm italic flex items-center gap-2'
                },
                e('i', { className: 'fas fa-Assistive-listening-systems fas fa-ear-listen animate-pulse' }),
                `${interimTranscript}...`
              )
            ),
          isSending &&
            e(
              'div',
              { className: 'text-xs text-slate-400 flex items-center gap-2 pl-1' },
              e('i', { className: 'fas fa-circle-notch fa-spin text-indigo-400' }),
              'Running Softmax intent inference...'
            ),
          e('div', { ref: chatBottomRef })
        ),

        // Error Banner if Mic blocked
        micError &&
          e(
            'div',
            {
              className:
                'mx-4 mb-2 px-3 py-2 rounded-lg bg-amber-500/15 border border-amber-500/30 text-amber-200 text-xs flex items-center justify-between'
            },
            e('span', null, micError),
            e(
              'button',
              { onClick: () => setMicError(''), className: 'text-amber-300 hover:text-white' },
              '✕'
            )
          ),

        // Input Composer with Mic + Text + Send
        e(
          'form',
          {
            id: 'chat-composer',
            onSubmit: (ev) => {
              ev.preventDefault()
              sendMessage()
            },
            className: 'p-3.5 lg:p-4 border-t border-slate-800/90 flex items-center gap-2.5'
          },
          e(
            'button',
            {
              id: 'mic-toggle-btn',
              type: 'button',
              onClick: toggleMicrophone,
              title: isListening ? 'Stop Microphone' : 'Start Live Speech Recognition',
              className: `w-11 h-11 rounded-xl flex items-center justify-center transition-all shrink-0 ${
                isListening
                  ? 'bg-rose-600 text-white mic-recording-pulse'
                  : 'bg-slate-800 hover:bg-slate-700 text-indigo-400 border border-slate-700'
              }`
            },
            e('i', { className: `fas ${isListening ? 'fa-stop' : 'fa-microphone'} text-base` })
          ),
          e('input', {
            id: 'chat-input',
            type: 'text',
            value: inputText,
            onChange: (ev) => setInputText(ev.target.value),
            placeholder: isListening
              ? 'Speak now... live ASR transcript appears automatically'
              : 'Speak with the mic or ask about Whisper, CTC, Conformer, MFCC, VAD, WER...',
            className:
              'flex-1 bg-slate-900 border border-slate-700/90 rounded-xl px-4 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500'
          }),
          e(
            'button',
            {
              id: 'send-message-btn',
              type: 'submit',
              disabled: !inputText.trim() || isSending,
              className:
                'px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:pointer-events-none text-white text-sm font-medium transition flex items-center gap-2 shrink-0'
            },
            e('span', { className: 'hidden sm:inline' }, 'Send'),
            e('i', { className: 'fas fa-paper-plane text-xs' })
          )
        )
      )
    )

    // ==================== TAB 2: WER EVALUATOR LAB ====================
    const werView = e(
      'main',
      { id: 'wer-lab-section', className: 'max-w-5xl mx-auto p-4 lg:p-6 space-y-6' },
      e(
        'section',
        { className: 'glass-panel rounded-2xl p-5 lg:p-6 space-y-5' },
        e(
          'div',
          { className: 'flex flex-wrap items-center justify-between gap-3' },
          e(
            'div',
            null,
            e(
              'h2',
              { className: 'text-lg font-bold text-white' },
              'Word Error Rate (WER) & CER Alignment Lab'
            ),
            e(
              'p',
              { className: 'text-xs text-slate-400 mt-0.5' },
              'Benchmark ASR output against ground-truth text—or read the reference aloud with your microphone!'
            )
          ),
          e(
            'div',
            { className: 'flex flex-wrap gap-1.5' },
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
                  className:
                    'text-xs px-2.5 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 transition'
                },
                ex.label
              )
            )
          )
        ),

        e(
          'div',
          { className: 'grid grid-cols-1 md:grid-cols-2 gap-4' },
          e(
            'div',
            { className: 'space-y-1.5' },
            e(
              'label',
              { className: 'text-xs font-medium text-slate-300' },
              'Ground-Truth Reference Transcript'
            ),
            e('textarea', {
              id: 'wer-reference-input',
              rows: 3,
              value: werRef,
              onChange: (ev) => setWerRef(ev.target.value),
              className:
                'w-full bg-slate-900 border border-slate-700 rounded-xl p-3 text-sm text-slate-100 focus:outline-none focus:border-indigo-500'
            })
          ),
          e(
            'div',
            { className: 'space-y-1.5' },
            e(
              'div',
              { className: 'flex items-center justify-between' },
              e(
                'label',
                { className: 'text-xs font-medium text-slate-300' },
                'ASR Hypothesis Transcript'
              ),
              e(
                'button',
                {
                  type: 'button',
                  onClick: dictateIntoWerHypothesis,
                  className: `text-xs px-2.5 py-1 rounded-lg flex items-center gap-1.5 transition ${
                    isListeningWer
                      ? 'bg-rose-600 text-white animate-pulse'
                      : 'bg-indigo-600/20 text-indigo-300 hover:bg-indigo-600/30 border border-indigo-500/30'
                  }`
                },
                e('i', { className: `fas ${isListeningWer ? 'fa-stop' : 'fa-microphone'}` }),
                isListeningWer ? 'Listening...' : 'Read Reference Aloud'
              )
            ),
            e('textarea', {
              id: 'wer-hypothesis-input',
              rows: 3,
              value: werHyp,
              onChange: (ev) => setWerHyp(ev.target.value),
              className:
                'w-full bg-slate-900 border border-slate-700 rounded-xl p-3 text-sm text-slate-100 focus:outline-none focus:border-indigo-500'
            })
          )
        ),

        e(
          'button',
          {
            id: 'compute-wer-btn',
            onClick: () => runWerEvaluation(werRef, werHyp),
            className:
              'px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium transition shadow-md shadow-indigo-600/20'
          },
          'Compute Levenshtein WER & CER'
        ),

        werResult &&
          e(
            'div',
            { className: 'space-y-5 pt-3 border-t border-slate-800' },
            // KPI Cards
            e(
              'div',
              { className: 'grid grid-cols-2 sm:grid-cols-5 gap-3' },
              [
                { label: 'Word Error Rate', val: `${werResult.wer}%`, tone: 'text-indigo-300' },
                { label: 'Char Error Rate', val: `${werResult.cer}%`, tone: 'text-violet-300' },
                { label: 'Substitutions (S)', val: werResult.substitutions, tone: 'text-amber-300' },
                { label: 'Deletions (D)', val: werResult.deletions, tone: 'text-rose-300' },
                { label: 'Insertions (I)', val: werResult.insertions, tone: 'text-sky-300' }
              ].map((stat, i) =>
                e(
                  'div',
                  {
                    key: i,
                    className: 'bg-slate-900/90 border border-slate-800 rounded-xl p-3.5 text-center'
                  },
                  e('div', { className: `text-xl font-bold ${stat.tone}` }, stat.val),
                  e('div', { className: 'text-[11px] text-slate-400 mt-0.5' }, stat.label)
                )
              )
            ),
            // Token-level visual diff
            e(
              'div',
              { className: 'space-y-2' },
              e(
                'div',
                { className: 'flex flex-wrap items-center justify-between text-xs text-slate-400' },
                e('span', { className: 'font-medium text-slate-300' }, 'Word-Level Alignment'),
                e(
                  'div',
                  { className: 'flex items-center gap-3 text-[11px]' },
                  e('span', { className: 'text-emerald-300' }, '■ Match'),
                  e('span', { className: 'text-amber-300' }, '■ Substitution'),
                  e('span', { className: 'text-rose-300' }, '■ Deletion'),
                  e('span', { className: 'text-sky-300' }, '■ Insertion')
                )
              ),
              e(
                'div',
                {
                  className:
                    'bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex flex-wrap gap-2'
                },
                werResult.alignment.map((item, idx) => {
                  if (item.type === 'correct') {
                    return e(
                      'span',
                      {
                        key: idx,
                        className:
                          'px-2.5 py-1 rounded-lg text-xs font-mono-code bg-emerald-500/10 text-emerald-300 border border-emerald-500/25'
                      },
                      item.hyp
                    )
                  }
                  if (item.type === 'substitution') {
                    return e(
                      'span',
                      {
                        key: idx,
                        className:
                          'px-2.5 py-1 rounded-lg text-xs font-mono-code bg-amber-500/15 text-amber-200 border border-amber-500/35'
                      },
                      `${item.ref} → ${item.hyp}`
                    )
                  }
                  if (item.type === 'deletion') {
                    return e(
                      'span',
                      {
                        key: idx,
                        className:
                          'px-2.5 py-1 rounded-lg text-xs font-mono-code bg-rose-500/15 text-rose-300 border border-rose-500/35 line-through'
                      },
                      item.ref
                    )
                  }
                  return e(
                    'span',
                    {
                      key: idx,
                      className:
                        'px-2.5 py-1 rounded-lg text-xs font-mono-code bg-sky-500/15 text-sky-300 border border-sky-500/35'
                    },
                    `+${item.hyp}`
                  )
                })
              )
            )
          )
      )
    )

    // ==================== TAB 3: PYTHON MODEL INSPECTOR ====================
    const modelView = e(
      'main',
      { id: 'python-model-section', className: 'max-w-5xl mx-auto p-4 lg:p-6 space-y-6' },
      e(
        'section',
        { className: 'glass-panel rounded-2xl p-5 lg:p-6 space-y-5' },
        e(
          'div',
          null,
          e(
            'h2',
            { className: 'text-lg font-bold text-white' },
            'Python Chatbot Training Pipeline & Exported Model'
          ),
          e(
            'p',
            { className: 'text-xs text-slate-400 mt-1' },
            'Trained offline in Python (`training/train_chatbot.py`) with scikit-learn & NumPy, exported to `src/model/chatbot_model.json`, and executed in pure JavaScript (`src/inference.js`).'
          )
        ),
        modelInfo
          ? e(
              'div',
              { className: 'space-y-5' },
              e(
                'div',
                { className: 'grid grid-cols-2 sm:grid-cols-4 gap-3' },
                [
                  {
                    label: 'Training Accuracy',
                    val: `${(modelInfo.train_accuracy * 100).toFixed(2)}%`
                  },
                  {
                    label: '5-Fold CV Accuracy',
                    val: `${(modelInfo.cv_accuracy_mean * 100).toFixed(1)}%`
                  },
                  { label: 'TF-IDF Vocabulary', val: `${modelInfo.vocab_size} n-grams` },
                  {
                    label: 'Training Samples',
                    val: `${modelInfo.augmented_samples} (${modelInfo.num_classes} classes)`
                  }
                ].map((m, i) =>
                  e(
                    'div',
                    {
                      key: i,
                      className: 'bg-slate-900 border border-slate-800 rounded-xl p-3.5'
                    },
                    e('div', { className: 'text-lg font-bold text-indigo-300' }, m.val),
                    e('div', { className: 'text-xs text-slate-400 mt-0.5' }, m.label)
                  )
                )
              ),
              e(
                'div',
                { className: 'bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-2' },
                e(
                  'div',
                  { className: 'text-xs font-semibold text-slate-300' },
                  'Retrain Command (Python 3):'
                ),
                e(
                  'pre',
                  {
                    className:
                      'font-mono-code text-xs text-emerald-300 bg-slate-950 p-3 rounded-lg border border-slate-800 overflow-x-auto'
                  },
                  '# Edit training/intents.json, then retrain & rebuild:\nnpm run train   # runs python3 training/train_chatbot.py\nnpm run build   # bundles updated chatbot_model.json into pure JS worker'
                )
              ),
              e(
                'div',
                { className: 'space-y-2' },
                e(
                  'h3',
                  { className: 'text-xs font-semibold uppercase tracking-wider text-slate-400' },
                  'Trained ASR Intent Classes'
                ),
                e(
                  'div',
                  { className: 'grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5' },
                  modelInfo.intents.map((it) =>
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
                        className:
                          'bg-slate-900/90 hover:bg-slate-800/90 border border-slate-800 hover:border-indigo-500/40 rounded-xl p-3 cursor-pointer transition'
                      },
                      e(
                        'div',
                        { className: 'flex items-center justify-between' },
                        e(
                          'span',
                          { className: 'font-mono-code text-xs text-indigo-400' },
                          it.tag
                        ),
                        e(
                          'span',
                          { className: 'text-[11px] text-slate-500' },
                          `${it.pattern_count} patterns`
                        )
                      ),
                      e('div', { className: 'text-sm font-medium text-slate-200 mt-1' }, it.title)
                    )
                  )
                )
              )
            )
          : e('p', { className: 'text-sm text-slate-400' }, 'Loading model summary...')
      )
    )

    return e(
      'div',
      { className: 'min-h-screen flex flex-col' },
      headerNode,
      activeTab === 'chat' ? chatView : activeTab === 'wer' ? werView : modelView
    )
  }

  const root = ReactDOM.createRoot(document.getElementById('root'))
  root.render(React.createElement(App))
})()
