"""
Certified World Languages Registry & Multilingual Pattern Generators
Covers UN official languages + ISO-639 / BCP-47 certified global languages.
Provides native names, BCP-47 codes, scripts, and multilingual training patterns
for ASR speech recognition, text normalization, and intent classification.
"""

CERTIFIED_LANGUAGES = [
    {
        "code": "en-US",
        "iso": "en",
        "name": "English (US)",
        "nativeName": "English (US)",
        "flag": "🇺🇸",
        "script": "Latin",
        "greetings": ["hello", "hi", "hey there", "good morning", "good evening", "how are you", "can you hear me"],
        "check": ["mic check one two", "testing microphone", "voice recognition test"]
    },
    {
        "code": "en-GB",
        "iso": "en",
        "name": "English (UK)",
        "nativeName": "English (UK)",
        "flag": "🇬🇧",
        "script": "Latin",
        "greetings": ["cheerio", "hello assistant", "good day", "greetings"],
        "check": ["is the microphone working", "audio check"]
    },
    {
        "code": "en-IN",
        "iso": "en",
        "name": "English (India)",
        "nativeName": "English (India)",
        "flag": "🇮🇳",
        "script": "Latin",
        "greetings": ["namaste", "hello assistant", "hey voxasr", "good morning"],
        "check": ["mic check", "can you hear my voice"]
    },
    {
        "code": "es-ES",
        "iso": "es",
        "name": "Spanish (Spain)",
        "nativeName": "Español (España)",
        "flag": "🇪🇸",
        "script": "Latin",
        "greetings": ["hola", "buenos días", "buenas tardes", "saludos", "hola asistente"],
        "check": ["prueba de micrófono", "puedes oírme", "comprobando micrófono"]
    },
    {
        "code": "es-MX",
        "iso": "es",
        "name": "Spanish (Mexico)",
        "nativeName": "Español (México)",
        "flag": "🇲🇽",
        "script": "Latin",
        "greetings": ["hola qué tal", "qué onda", "buen día", "hola chatbot"],
        "check": ["probando audio", "me escuchas"]
    },
    {
        "code": "zh-CN",
        "iso": "zh",
        "name": "Chinese (Mandarin - Simplified)",
        "nativeName": "简体中文",
        "flag": "🇨🇳",
        "script": "Hanzi",
        "greetings": ["你好", "您好", "早上好", "晚上好", "助手你好", "你好聊天机器人"],
        "check": ["测试麦克风", "你能听到我说话吗", "语音识别测试", "麦克风测试"]
    },
    {
        "code": "zh-TW",
        "iso": "zh",
        "name": "Chinese (Traditional)",
        "nativeName": "繁體中文",
        "flag": "🇹🇼",
        "script": "Hanzi",
        "greetings": ["您好", "早安", "晚安", "哈囉", "機器人你好"],
        "check": ["測試麥克風", "聽得到我的聲音嗎", "語音識別測試"]
    },
    {
        "code": "hi-IN",
        "iso": "hi",
        "name": "Hindi",
        "nativeName": "हिन्दी",
        "flag": "🇮🇳",
        "script": "Devanagari",
        "greetings": ["नमस्ते", "नमस्कार", "सुप्रभात", "शुभ संध्या", "हेलो असिस्टेंट", "क्या आप सुन रहे हैं"],
        "check": ["माइक परीक्षण", "क्या आप मेरी आवाज़ सुन सकते हैं", "माइक्रोफ़ोन चेक", "वॉइस टेस्ट"]
    },
    {
        "code": "ar-SA",
        "iso": "ar",
        "name": "Arabic",
        "nativeName": "العربية",
        "flag": "🇸🇦",
        "script": "Arabic",
        "greetings": ["مرحبا", "أهلا وسهلا", "صباح الخير", "مساء الخير", "السلام عليكم", "تحياتي"],
        "check": ["فحص الميكروفون", "هل تسمعني", "اختبار الصوت", "تجربة الميكروفون"]
    },
    {
        "code": "fr-FR",
        "iso": "fr",
        "name": "French (France)",
        "nativeName": "Français",
        "flag": "🇫🇷",
        "script": "Latin",
        "greetings": ["bonjour", "salut", "bonsoir", "coucou", "salutations assistant"],
        "check": ["test du microphone", "m'entendez-vous", "test vocal", "vérification micro"]
    },
    {
        "code": "ru-RU",
        "iso": "ru",
        "name": "Russian",
        "nativeName": "Русский",
        "flag": "🇷🇺",
        "script": "Cyrillic",
        "greetings": ["здравствуйте", "привет", "доброе утро", "добрый вечер", "привет ассистент"],
        "check": ["проверка микрофона", "ты меня слышишь", "тест распознавания речи"]
    },
    {
        "code": "pt-BR",
        "iso": "pt",
        "name": "Portuguese (Brazil)",
        "nativeName": "Português (Brasil)",
        "flag": "🇧🇷",
        "script": "Latin",
        "greetings": ["olá", "oi", "bom dia", "boa tarde", "saudações assistente"],
        "check": ["teste de microfone", "você pode me ouvir", "teste de voz"]
    },
    {
        "code": "de-DE",
        "iso": "de",
        "name": "German",
        "nativeName": "Deutsch",
        "flag": "🇩🇪",
        "script": "Latin",
        "greetings": ["hallo", "guten morgen", "guten tag", "guten abend", "grüße"],
        "check": ["mikrofontest", "kannst du mich hören", "spracherkennungstest"]
    },
    {
        "code": "ja-JP",
        "iso": "ja",
        "name": "Japanese",
        "nativeName": "日本語",
        "flag": "🇯🇵",
        "script": "Kanji/Kana",
        "greetings": ["こんにちは", "おはようございます", "こんばんは", "はじめまして", "ハロー"],
        "check": ["マイクテスト", "聞こえますか", "音声認識テスト", "マイクの確認"]
    },
    {
        "code": "ko-KR",
        "iso": "ko",
        "name": "Korean",
        "nativeName": "한국어",
        "flag": "🇰🇷",
        "script": "Hangul",
        "greetings": ["안녕하세요", "좋은 아침입니다", "반갑습니다", "안녕 챗봇"],
        "check": ["마이크 테스트", "제 목소리가 들리나요", "음성 인식 테스트"]
    },
    {
        "code": "it-IT",
        "iso": "it",
        "name": "Italian",
        "nativeName": "Italiano",
        "flag": "🇮🇹",
        "script": "Latin",
        "greetings": ["ciao", "buongiorno", "buonasera", "saluti assistente"],
        "check": ["prova microfono", "mi senti", "test vocale"]
    },
    {
        "code": "bn-IN",
        "iso": "bn",
        "name": "Bengali",
        "nativeName": "বাংলা",
        "flag": "🇧🇩",
        "script": "Bengali",
        "greetings": ["নমস্কার", "হ্যালো", "শুভ সকাল", "শুভ সন্ধ্যা", "কেমন আছেন"],
        "check": ["মাইক পরীক্ষা", "আপনি কি আমার কথা শুনতে পাচ্ছেন", "ভয়েস টেস্ট"]
    },
    {
        "code": "tr-TR",
        "iso": "tr",
        "name": "Turkish",
        "nativeName": "Türkçe",
        "flag": "🇹🇷",
        "script": "Latin",
        "greetings": ["merhaba", "selam", "günaydın", "iyi akşamlar", "merhaba asistan"],
        "check": ["mikrofon testi", "beni duyabiliyor musun", "ses tanıma testi"]
    },
    {
        "code": "vi-VN",
        "iso": "vi",
        "name": "Vietnamese",
        "nativeName": "Tiếng Việt",
        "flag": "🇻🇳",
        "script": "Latin",
        "greetings": ["xin chào", "chào bạn", "chào buổi sáng", "xin chào trợ lý"],
        "check": ["kiểm tra micrô", "bạn có nghe thấy tôi không", "thử giọng nói"]
    },
    {
        "code": "id-ID",
        "iso": "id",
        "name": "Indonesian",
        "nativeName": "Bahasa Indonesia",
        "flag": "🇮🇩",
        "script": "Latin",
        "greetings": ["halo", "hai", "selamat pagi", "selamat siang", "selamat malam"],
        "check": ["tes mikrofon", "apakah Anda bisa mendengar saya", "uji coba suara"]
    },
    {
        "code": "nl-NL",
        "iso": "nl",
        "name": "Dutch",
        "nativeName": "Nederlands",
        "flag": "🇳🇱",
        "script": "Latin",
        "greetings": ["hallo", "goedemorgen", "goedemiddag", "goedenavond", "hoi"],
        "check": ["microfoontest", "kun je me horen", "spraakherkenning test"]
    },
    {
        "code": "pl-PL",
        "iso": "pl",
        "name": "Polish",
        "nativeName": "Polski",
        "flag": "🇵🇱",
        "script": "Latin",
        "greetings": ["dzień dobry", "cześć", "witaj", "dobry wieczór"],
        "check": ["test mikrofonu", "czy mnie słyszysz", "próba mikrofonu"]
    },
    {
        "code": "fa-IR",
        "iso": "fa",
        "name": "Persian",
        "nativeName": "فارسی",
        "flag": "🇮🇷",
        "script": "Arabic",
        "greetings": ["سلام", "درود", "صبح بخیر", "عصر بخیر", "سلام دستیار"],
        "check": ["آزمایش میکروفون", "صدای من را می شنوید", "تست صوتی"]
    },
    {
        "code": "sw-KE",
        "iso": "sw",
        "name": "Swahili",
        "nativeName": "Kiswahili",
        "flag": "🇰🇪",
        "script": "Latin",
        "greetings": ["jambo", "habari", "habari za asubuhi", "hujambo"],
        "check": ["majaribio ya kipaza sauti", "unaweza kunisikia", "jaribio la sauti"]
    },
    {
        "code": "ta-IN",
        "iso": "ta",
        "name": "Tamil",
        "nativeName": "தமிழ்",
        "flag": "🇮🇳",
        "script": "Tamil",
        "greetings": ["வணக்கம்", "காலை வணக்கம்", "மாலை வணக்கம்", "வணக்கம் உதவியாளர்"],
        "check": ["மைக் சோதனை", "என் குரல் கேட்கிறதா", "மைக்ரோஃபோன் சரிபார்ப்பு"]
    },
    {
        "code": "te-IN",
        "iso": "te",
        "name": "Telugu",
        "nativeName": "తెలుగు",
        "flag": "🇮🇳",
        "script": "Telugu",
        "greetings": ["నమస్కారం", "శుభోదయం", "శుభ సాయంత్రం", "హలో అసిస్టెంట్"],
        "check": ["మైక్ టెస్ట్", "నా వాయిస్ వినపడుతోందా", "మైక్రోఫోన్ తనిఖీ"]
    },
    {
        "code": "ur-PK",
        "iso": "ur",
        "name": "Urdu",
        "nativeName": "اردو",
        "flag": "🇵🇰",
        "script": "Arabic",
        "greetings": ["السلام علیکم", "ہیلو", "صبح بخیر", "شام بخیر", "آداب"],
        "check": ["مائیک چیک", "کیا آپ میری آواز سن سکتے ہیں", "مائیکروفون ٹیسٹ"]
    },
    {
        "code": "mr-IN",
        "iso": "mr",
        "name": "Marathi",
        "nativeName": "मराठी",
        "flag": "🇮🇳",
        "script": "Devanagari",
        "greetings": ["नमस्कार", "शुभ प्रभात", "शुभ संध्याकाळ", "हॅलो सहाय्यक"],
        "check": ["माईक चाचणी", "तुम्हाला माझा आवाज येतोय का", "मायक्रोफोन टेस्ट"]
    },
    {
        "code": "uk-UA",
        "iso": "uk",
        "name": "Ukrainian",
        "nativeName": "Українська",
        "flag": "🇺🇦",
        "script": "Cyrillic",
        "greetings": ["добрий день", "привіт", "доброго ранку", "добрий вечір", "вітаю"],
        "check": ["перевірка мікрофона", "ти мене чуєш", "тест мікрофона"]
    },
    {
        "code": "el-GR",
        "iso": "el",
        "name": "Greek",
        "nativeName": "Ελληνικά",
        "flag": "🇬🇷",
        "script": "Greek",
        "greetings": ["γεια σας", "καλημέρα", "καλησπέρα", "γεια σου βοηθέ"],
        "check": ["έλεγχος μικροφώνου", "με ακούτε", "δοκιμή μικροφώνου"]
    },
    {
        "code": "sv-SE",
        "iso": "sv",
        "name": "Swedish",
        "nativeName": "Svenska",
        "flag": "🇸🇪",
        "script": "Latin",
        "greetings": ["hej", "hallå", "god morgon", "god kväll", "hejsan"],
        "check": ["mikrofontest", "hör du mig", "kontrollera mikrofon"]
    },
    {
        "code": "he-IL",
        "iso": "he",
        "name": "Hebrew",
        "nativeName": "עברית",
        "flag": "🇮🇱",
        "script": "Hebrew",
        "greetings": ["שלום", "בוקר טוב", "ערב טוב", "היי עוזר"],
        "check": ["בדיקת מיקרופון", "האם אתה שומע אותי", "בדיקת שמע"]
    },
    {
        "code": "th-TH",
        "iso": "th",
        "name": "Thai",
        "nativeName": "ไทย",
        "flag": "🇹🇭",
        "script": "Thai",
        "greetings": ["สวัสดี", "สวัสดีครับ", "สวัสดีค่ะ", "อรุณสวัสดิ์"],
        "check": ["ทดสอบไมโครโฟน", "คุณได้ยินฉันไหม", "ทดสอบเสียง"]
    },
    {
        "code": "cs-CZ",
        "iso": "cs",
        "name": "Czech",
        "nativeName": "Čeština",
        "flag": "🇨🇿",
        "script": "Latin",
        "greetings": ["ahoj", "dobrý den", "dobré ráno", "dobrý večer"],
        "check": ["test mikrofonu", "slyšíš mě", "zkouška mikrofonu"]
    }
]

def get_all_greetings():
    res = []
    for lang in CERTIFIED_LANGUAGES:
        res.extend(lang["greetings"])
        res.extend(lang["check"])
    return list(dict.fromkeys(res))

def get_language_summary():
    return [
        {
            "code": l["code"],
            "iso": l["iso"],
            "name": l["name"],
            "nativeName": l["nativeName"],
            "flag": l["flag"],
            "script": l["script"]
        }
        for l in CERTIFIED_LANGUAGES
    ]
