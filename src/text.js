/**
 * Shared text utilities (pure JavaScript, no dependencies):
 * Unicode normalisation, retrieval tokenizer, language detection,
 * sentence splitting and Wikipedia-extract cleanup.
 */

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

const CJK_RE = /[぀-ヿ㐀-䶿一-鿿가-힯]/

// Words that never carry the topic of a question.
const STOPWORDS = {
  en: 'a an the is are was were be been being am of in on at to for from by with about as into onto over under and or but if then than that this these those it its i me my mine you your we our us they them their he she his her him do does did doing done can could would should will shall may might must have has had having what which who whom whose when where why how there here so also just very much many more most not no yes some any each every all both either neither such own same too only out up down off again once s t ll re ve d m'
    .split(' '),
  es: 'qué que es son el la los las un una unos unas de del al en por para con sin sobre y o pero como cómo cuál cuáles cuándo dónde quién quiénes porqué se su sus lo le me mi tu te nos funciona explica explícame dime cuéntame significa hola favor puedes'
    .split(' '),
  fr: 'qu est ce que qui quoi quel quelle quels quelles le la les un une des du de d l en et ou mais dans sur pour par avec sans comment pourquoi où quand je tu il elle nous vous ils me te se mon ma mes ton ta son sa fonctionne explique expliquez moi dis parle signifie bonjour sont c s il'
    .split(' '),
  de: 'was ist sind der die das den dem des ein eine einen einem einer und oder aber in im an am auf für von vom mit ohne über zu zum zur wie warum wo wann wer wen wem ich du er sie es wir ihr mir mich dir dich funktioniert erkläre erklären erklär bedeutet bitte hallo man kann'
    .split(' '),
  pt: 'o a os as um uma uns umas de do da dos das em no na nos nas por para com sem sobre e ou mas que qual quais como quando onde quem porque é são se seu sua me te você funciona explique explica significa olá favor pode'
    .split(' '),
  it: 'il lo la i gli le un una uno di del della dei delle a al alla in nel nella su per con senza e o ma che chi cosa come quando dove perché è sono si mi ti funziona spiega spiegami significa ciao favore può cos'
    .split(' '),
  nl: 'de het een van in op aan voor met zonder over en of maar wat wie hoe waarom waar wanneer is zijn ik je jij u hij zij we werkt uitleg leg betekent hallo kun'
    .split(' '),
  ru: 'что такое это как почему где когда кто в на по для с без о об и или но а не же ли бы я ты он она мы вы они работает объясни объясните расскажи расскажите значит привет пожалуйста чем какой какая какие'
    .split(' '),
  hi: 'क्या है हैं था थी थे का की के को में पर से और या लेकिन यह वह ये वे कैसे क्यों कहाँ कहां कब कौन मैं तुम आप हम काम करता करती करते समझाइए समझाओ बताइए बताओ मतलब नमस्ते कृपया एक होता होती होते'
    .split(' '),
  tr: 've veya ama bir bu şu ne nedir nasıl neden nerede kim için ile de da mi mı mu mü çalışır açıkla anlat merhaba lütfen'
    .split(' '),
  id: 'apa itu adalah yang dan atau tetapi di ke dari untuk dengan tanpa bagaimana mengapa kenapa dimana kapan siapa saya kamu anda cara kerja jelaskan artinya halo tolong'
    .split(' '),
  pl: 'co to jest są i lub ale w na do z od dla o jak dlaczego gdzie kiedy kto ja ty on ona działa wyjaśnij wytłumacz oznacza cześć proszę czym'
    .split(' '),
  sv: 'vad är en ett och eller men i på till från för med utan om hur varför var när vem jag du han hon fungerar förklara betyder hej snälla'
    .split(' '),
  ro: 'ce este sunt un o și sau dar în pe la cu fără despre cum de ce unde când cine funcționează explică salut te rog'
    .split(' ')
}

const STOP_SETS = Object.fromEntries(Object.entries(STOPWORDS).map(([k, v]) => [k, new Set(v)]))
const EN_STOP = STOP_SETS.en

// Polite / conversational filler that should not be required of an answer.
export const FILLER = new Set(
  'please pls kindly tell explain describe define definition meaning mean means meant work works working worked know want need like let give show teach help understand basically exactly actually briefly brief short quick simple simply easy terms detail details detailed overview introduction intro concept idea something thing things stuff regarding concerning say said tell talk bit little info information fact facts question answer wanna gonna really literally example examples called named known term word hey hi hello ok okay thanks thank'
    .split(' ')
)

/** Light plural stemmer (ASCII words only) so "trees" ~ "tree", "queries" ~ "query". */
export function stem(word) {
  if (word.length <= 3 || !/^[a-z0-9]+$/.test(word)) return word
  if (word.endsWith('ies') && word.length > 4) return word.slice(0, -3) + 'y'
  if (word.endsWith('sses')) return word.slice(0, -2)
  if (word.endsWith('s') && !/(ss|us|is|as|os)$/.test(word)) return word.slice(0, -1)
  return word
}

/**
 * Tokenizer used for both indexing and querying.
 * Drops stopwords (English plus the detected language), stems plurals and
 * emits character bigrams for unsegmented CJK runs.
 */
export function tokenize(rawText, lang = 'en', keepFiller = true) {
  const norm = normalizeText(String(rawText).replace(/(\p{L})['’]s\b/gu, '$1'))
  if (!norm) return []
  const extra = lang !== 'en' ? STOP_SETS[lang] : null
  const out = []
  const words = norm.split(' ')
  for (let i = 0; i < words.length; i++) {
    const w = words[i]
    if (CJK_RE.test(w)) {
      const chars = Array.from(w).filter(isCjkChar)
      if (chars.length === 1) out.push(chars[0])
      for (let j = 0; j < chars.length - 1; j++) out.push(chars[j] + chars[j + 1])
      continue
    }
    if (EN_STOP.has(w) || (extra && extra.has(w))) continue
    if (!keepFiller && FILLER.has(w)) continue
    out.push(stem(w))
  }
  return out
}

/** True when two tokens are the same word up to a suffix ("invented" ~ "inventor"). */
export function looseMatch(a, b) {
  if (a === b) return true
  const short = a.length <= b.length ? a : b
  const long = a.length <= b.length ? b : a
  if (short.length < 4) return false
  let p = 0
  while (p < short.length && short[p] === long[p]) p++
  return p >= 4 && p >= short.length - 2 && p >= Math.ceil(long.length * 0.6)
}

// ---------------------------------------------------------------------------
// Language detection
// ---------------------------------------------------------------------------

const SCRIPT_LANGS = [
  [/[ऀ-ॿ]/, { script: 'Devanagari', lang: 'hi-IN', label: 'Hindi' }],
  [/[぀-ゟ゠-ヿ]/, { script: 'Kana', lang: 'ja-JP', label: 'Japanese' }],
  [/[가-힯]/, { script: 'Hangul', lang: 'ko-KR', label: 'Korean' }],
  [/[一-鿿]/, { script: 'Hanzi', lang: 'zh-CN', label: 'Mandarin Chinese' }],
  [/[Ѐ-ӿ]/, { script: 'Cyrillic', lang: 'ru-RU', label: 'Russian' }],
  [/[؀-ۿ]/, { script: 'Arabic', lang: 'ar-SA', label: 'Arabic' }],
  [/[ঀ-৿]/, { script: 'Bengali', lang: 'bn-IN', label: 'Bengali' }],
  [/[஀-௿]/, { script: 'Tamil', lang: 'ta-IN', label: 'Tamil' }],
  [/[ఀ-౿]/, { script: 'Telugu', lang: 'te-IN', label: 'Telugu' }],
  [/[Ͱ-Ͽ]/, { script: 'Greek', lang: 'el-GR', label: 'Greek' }],
  [/[֐-׿]/, { script: 'Hebrew', lang: 'he-IL', label: 'Hebrew' }],
  [/[฀-๿]/, { script: 'Thai', lang: 'th-TH', label: 'Thai' }]
]

// Same-script languages the speech-locale hint may refine.
const SCRIPT_VARIANTS = {
  Devanagari: { 'mr-IN': 'Marathi' },
  Arabic: { 'fa-IR': 'Persian', 'ur-PK': 'Urdu' },
  Cyrillic: { 'uk-UA': 'Ukrainian' },
  Hanzi: { 'zh-TW': 'Traditional Chinese', 'ja-JP': 'Japanese' }
}

const LATIN_MARKERS = {
  'es-ES': ['Spanish', 'qué cómo cuál dónde quién es el los las una del por para con funciona explica hola gracias cuándo está son', /[ñ¿¡]/],
  'fr-FR': ['French', "qu est-ce est ce que comment pourquoi le les des une du dans avec fonctionne expliquez bonjour merci quelle quel sont où", /[çœ]|è|ê|à /],
  'de-DE': ['German', 'was ist wie warum der die das ein eine und funktioniert erkläre hallo danke nicht mit für sind wer wo', /[äöüß]/],
  'pt-BR': ['Portuguese', 'o que é como qual quais onde você uma do da dos das não funciona explique olá obrigado são em para', /[ãõ]|ç/],
  'it-IT': ['Italian', "cos è che cosa come quale perché il gli una della delle funziona spiega ciao grazie sono dove chi", /è |ò|ù/],
  'nl-NL': ['Dutch', 'wat is hoe waarom het een van en werkt uitleg hallo dank niet wie waar zijn', null],
  'tr-TR': ['Turkish', 'nedir nasıl neden ve bir bu için ile çalışır merhaba teşekkürler', /[ğışİ]/],
  'id-ID': ['Indonesian', 'apa itu adalah yang dan bagaimana mengapa untuk dengan jelaskan halo terima kasih cara kerja', null],
  'vi-VN': ['Vietnamese', 'là gì như thế nào tại sao và của cho không xin chào cảm ơn', /[ăơưđ]|[ạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]/],
  'pl-PL': ['Polish', 'co to jest jak dlaczego czym i w na działa wyjaśnij cześć dziękuję', /[ąęłńśźż]/],
  'sv-SE': ['Swedish', 'vad är hur varför och en ett fungerar förklara hej tack', /å/],
  'sw-KE': ['Swahili', 'nini ni jinsi kwa nini na ya wa habari asante inafanyaje', null],
  'cs-CZ': ['Czech', 'co je jak proč a v na funguje vysvětli ahoj děkuji', /[ěščřžůť]/],
  'ro-RO': ['Romanian', 'ce este cum de ce funcționează explică salut mulțumesc', /[ăâîșț]/]
}
const EN_MARKERS = new Set(
  'what is are how why who when where which the a an of in on for to and does do did can explain tell me about with from this that it be was were has have you your'.split(' ')
)
const LATIN_SETS = Object.entries(LATIN_MARKERS).map(([code, [label, words, charRe]]) => ({
  code,
  label,
  words: new Set(words.split(' ')),
  charRe
}))

/**
 * Detects script + language. `hint` is the speech locale selected in the UI;
 * it only breaks ties (same script, or Latin text with no evidence either way).
 */
export function detectLanguage(rawText = '', hint = '') {
  const text = String(rawText)
  for (const [re, meta] of SCRIPT_LANGS) {
    if (re.test(text)) {
      const variant = SCRIPT_VARIANTS[meta.script] && SCRIPT_VARIANTS[meta.script][hint]
      return variant ? { script: meta.script, lang: hint, label: variant } : meta
    }
  }

  const lower = text.toLowerCase()
  const words = lower.replace(/[^\p{L}\p{M}\s-]/gu, ' ').split(/\s+/).filter(Boolean)
  let enScore = 0
  for (const w of words) if (EN_MARKERS.has(w)) enScore++

  let best = null
  let bestScore = 0
  for (const cand of LATIN_SETS) {
    let score = 0
    const seen = new Set()
    for (const w of words) {
      if (cand.words.has(w) && !seen.has(w)) {
        seen.add(w)
        score++
      }
    }
    if (cand.charRe && cand.charRe.test(lower)) score += 1.5
    if (score > bestScore || (score === bestScore && best && cand.code === hint)) {
      best = cand
      bestScore = score
    }
  }

  if (best && bestScore >= 2 && bestScore > enScore) {
    return { script: 'Latin', lang: best.code, label: best.label }
  }
  if (best && bestScore >= 1 && enScore === 0 && words.length <= 3) {
    return { script: 'Latin', lang: best.code, label: best.label }
  }
  return { script: 'Latin', lang: 'en-US', label: 'English' }
}

/** BCP-47 locale -> Wikipedia language subdomain. */
export function wikiLang(locale = 'en-US') {
  const iso = String(locale).split('-')[0].toLowerCase()
  return iso || 'en'
}

// ---------------------------------------------------------------------------
// Sentences & extract cleanup
// ---------------------------------------------------------------------------

const ABBREVIATIONS = /(?:\b(?:e\.g|i\.e|etc|vs|cf|approx|ca|dr|mr|mrs|ms|prof|st|jr|sr|inc|ltd|co|fig|no|vol|al|u\.s|u\.k)|\b[A-Z])\.$/i

/** Splits prose into sentences (Latin, CJK, Devanagari and Arabic punctuation). */
export function splitSentences(text = '') {
  const flat = String(text).replace(/\s+/g, ' ').trim()
  if (!flat) return []
  const rough = flat.split(/(?<=[.!?])\s+(?=[\p{Lu}\p{N}"'“‘(\[$¿¡])|(?<=[。！？।؟])\s*/u)
  const out = []
  for (const part of rough) {
    const piece = part.trim()
    if (!piece) continue
    if (out.length && ABBREVIATIONS.test(out[out.length - 1])) {
      out[out.length - 1] += ' ' + piece
    } else {
      out.push(piece)
    }
  }
  return out
}

export function truncateAtSentence(text = '', limit = 1000) {
  if (text.length <= limit) return text
  const sentences = splitSentences(text)
  let out = ''
  for (const s of sentences) {
    if ((out + ' ' + s).trim().length > limit) break
    out = (out + ' ' + s).trim()
  }
  if (out.length >= limit * 0.4) return out
  return text.slice(0, limit).replace(/\s+\S*$/, '').replace(/[,;:\s]+$/, '') + '…'
}

const MATH_BLOCK = /(?:\n[ \t]+[^\n{]*)*\n[ \t]+\{\\displaystyle ([^\n]*)\}[ \t]*\n?[ \t]*/g
const INLINE_MATH = /\{\\displaystyle ([^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*)\}/g

/** Turns a raw Wikipedia plain-text extract into tidy prose with $inline math$. */
export function cleanExtract(raw = '') {
  if (!raw) return ''
  let text = '\n' + String(raw).replace(/\r/g, '').replace(/[⁠​]/g, '')
  text = text.replace(MATH_BLOCK, (_, m) => ` $${m.trim()}$ `)
  text = text.replace(INLINE_MATH, (_, m) => ` $${m.trim()}$ `)
  text = text.trim().replace(/(?<=[^\s.!?:;,$])\n+(?=\S)/g, '. ')
  text = text.replace(/\[\d+\]/g, '').replace(/\s+/g, ' ').trim()
  text = text.replace(/\s*\((?:[\s,;:–-]|listen|or|and)*\)/g, '')
  text = text.replace(/\s+([,.;:!?)])/g, '$1').replace(/\(\s+/g, '(')
  return text.trim()
}

const ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', ndash: '–', mdash: '—', hellip: '…', copy: '©', reg: '®', laquo: '«', raquo: '»', rsquo: '’', lsquo: '‘', rdquo: '”', ldquo: '“', times: '×', middot: '·', deg: '°' }

export function decodeEntities(str = '') {
  return String(str).replace(/&(#x?[0-9a-f]+|[a-z]+);/gi, (whole, code) => {
    if (code[0] === '#') {
      const cp = code[1] === 'x' || code[1] === 'X' ? parseInt(code.slice(2), 16) : parseInt(code.slice(1), 10)
      return Number.isFinite(cp) && cp > 0 && cp < 0x110000 ? String.fromCodePoint(cp) : ' '
    }
    const named = ENTITIES[code.toLowerCase()]
    return named !== undefined ? named : whole
  })
}
