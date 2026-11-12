#!/usr/bin/env python3
"""
Advanced Python NLP Intent Classifier Training Pipeline for VoxASR
-------------------------------------------------------------------
Trains a Multi-Class Softmax Neural / Logistic Classifier across:
- Certified World Languages (English, Spanish, Mandarin, Hindi, French, German, Russian, Arabic, Japanese, Portuguese, etc.)
- All Core Pillars of Computer Science & Mathematical Computing (Algorithms, Data Structures, Numerical Analysis, ML/DL, Theory, Systems, Compilers, Cryptography, GPU/HPC, Databases, NLP/LLMs, Speech AI).
- Ingests both `training/intents.json` and dynamic web-scraped articles from `training/dynamic_knowledge_pool.json`.
- Exports portable JSON weights (`src/model/chatbot_model.json`) for <1 ms pure JavaScript edge inference.
"""

import json
import logging
import math
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainChatbot")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from training import kb_store  # noqa: E402

INTENTS_PATH = ROOT_DIR / "training" / "intents.json"
DYNAMIC_POOL_PATH = ROOT_DIR / "training" / "dynamic_knowledge_pool.json"
MODEL_OUT_PATH = ROOT_DIR / "src" / "model" / "chatbot_model.json"
HISTORY_PATH = ROOT_DIR / "src" / "model" / "training_history.json"


def normalize_text(text: str) -> str:
    """
    Unicode-aware text normalization identical in Python and JavaScript inference.
    Preserves letters (L), combining marks (M), and numbers (N) across all global scripts.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text).lower())
    chars = []
    for c in text:
        cat = unicodedata.category(c)
        if cat[0] in ("L", "M", "N") or c.isspace():
            chars.append(c)
        else:
            chars.append(" ")
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def is_cjk_char(ch: str) -> bool:
    """Checks whether a character belongs to CJK (Chinese, Japanese, Korean) scripts."""
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF or   # CJK Unified Ideographs
        0x3400 <= cp <= 0x4DBF or   # CJK Extension A
        0x3040 <= cp <= 0x309F or   # Hiragana
        0x30A0 <= cp <= 0x30FF or   # Katakana
        0xAC00 <= cp <= 0xD7AF      # Hangul Syllables
    )


def extract_word_ngrams(norm_text: str):
    """Extract word 1-grams and 2-grams."""
    words = norm_text.split() if norm_text else []
    tokens = list(words)
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]}_{words[i + 1]}")
    return tokens


def extract_char_ngrams(norm_text: str):
    """
    Extract per-word character 3-grams & 4-grams, plus CJK unigrams and bigrams.
    Robust to ASR homophones, speech typos, and unsegmented Asian languages.
    """
    tokens = []
    # CJK unigrams and bigrams
    cjk_seq = [c for c in norm_text if is_cjk_char(c)]
    for i, c in enumerate(cjk_seq):
        tokens.append(f"c:{c}")
        if i < len(cjk_seq) - 1:
            tokens.append(f"c:{c}{cjk_seq[i+1]}")

    # Standard per-word char n-grams for alphabetic and abugida scripts
    for word in norm_text.split():
        if len(word) < 2 or any(is_cjk_char(c) for c in word):
            continue
        padded = f"<{word}>"
        for n in (3, 4):
            for i in range(max(0, len(padded) - n + 1)):
                tokens.append(f"c:{padded[i:i + n]}")
    return tokens


def combined_analyzer(raw_text: str):
    norm = normalize_text(raw_text)
    return extract_word_ngrams(norm) + extract_char_ngrams(norm)


def load_all_training_data():
    """
    Loads base intents from `intents.json` and enriches them with:
    1. Sharded knowledge pool from `public/static/kb/` (via `kb_store.load_kb()`)
    2. Legacy dynamic knowledge pool (`dynamic_knowledge_pool.json`) for ArXiv/fallback insights.
    """
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        base_dataset = json.load(f)

    intents_dict = {}
    for item in base_dataset["intents"]:
        tag = item["tag"]
        intents_dict[tag] = {
            "tag": tag,
            "title": item["title"],
            "patterns": list(dict.fromkeys(item["patterns"])),
            "responses": item["responses"],
            "follow_ups": item.get("follow_ups", []),
            "scraped_insights": [],
        }

    scraped_articles_count = 0
    seen_titles = set()

    # 1. Load sharded KB articles (public/static/kb/)
    kb_articles = kb_store.load_kb()
    per_domain_patterns = {}
    for title, art in kb_articles.items():
        scraped_articles_count += 1
        seen_titles.add(title)
        domain_key = art.get("d")
        if domain_key and domain_key in intents_dict:
            count = per_domain_patterns.get(domain_key, 0)
            if count < 80:
                clean_title = re.sub(r"\s*\([^)]*\)$", "", title).strip().lower()
                if clean_title and len(clean_title) >= 2:
                    intents_dict[domain_key]["patterns"].append(clean_title)
                    intents_dict[domain_key]["patterns"].append(f"what is {clean_title}")
                    for alias in (art.get("a") or [])[:3]:
                        a_clean = str(alias).strip().lower()
                        if len(a_clean) >= 2:
                            intents_dict[domain_key]["patterns"].append(a_clean)
                    per_domain_patterns[domain_key] = count + 1
            if len(intents_dict[domain_key]["scraped_insights"]) < 4:
                lang = art.get("l", "en")
                src = art.get("s", "Wikipedia")
                url = art.get("u") or f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
                intents_dict[domain_key]["scraped_insights"].append({
                    "title": title,
                    "source": src,
                    "summary": art.get("x", "")[:240],
                    "url": url,
                })

    # 2. Enrich with dynamic_knowledge_pool.json (e.g. ArXiv preprints) without duplicating titles
    if DYNAMIC_POOL_PATH.exists():
        try:
            with open(DYNAMIC_POOL_PATH, "r", encoding="utf-8") as f:
                dyn_data = json.load(f)
            domains = dyn_data.get("domains", {})
            for domain_key, domain_info in domains.items():
                articles = domain_info.get("articles", [])
                target_tag = domain_key
                if target_tag not in intents_dict:
                    for k in intents_dict:
                        if domain_key in k or k in domain_key:
                            target_tag = k
                            break
                if target_tag in intents_dict:
                    for art in articles:
                        t = art.get("title", "")
                        if t in seen_titles:
                            continue
                        seen_titles.add(t)
                        scraped_articles_count += 1
                        # Only add concise domain-specific title/first-3 patterns to avoid cross-domain template noise
                        for p in art.get("patterns", [])[:5]:
                            intents_dict[target_tag]["patterns"].append(p)
                        if len(intents_dict[target_tag]["scraped_insights"]) < 4:
                            intents_dict[target_tag]["scraped_insights"].append({
                                "title": t,
                                "source": art.get("source", "ArXiv"),
                                "summary": art.get("summary", "")[:240],
                                "url": art.get("url", ""),
                            })
        except Exception as ex:
            logger.warning(f"Error loading dynamic knowledge pool: {ex}")

    # Deduplicate patterns per intent
    for tag in intents_dict:
        intents_dict[tag]["patterns"] = list(dict.fromkeys(intents_dict[tag]["patterns"]))

    final_intents = list(intents_dict.values())
    return final_intents, scraped_articles_count


def train_and_export():
    print("=" * 72)
    print("VoxASR Certified Multilingual CS & Math AI Training Pipeline")
    print("=" * 72)

    intents, scraped_count = load_all_training_data()
    print(f"Loaded {len(intents)} intent categories | Dynamic Scraped Articles: {scraped_count}")

    texts = []
    labels = []
    raw_texts = []
    raw_labels = []
    intent_meta = {}

    spoken_prefixes = ["", "please ", "can you explain ", "tell me about "]

    for idx, intent in enumerate(intents):
        tag = intent["tag"]
        intent_meta[tag] = {
            "index": idx,
            "tag": tag,
            "title": intent["title"],
            "responses": intent["responses"],
            "scraped_insights": intent.get("scraped_insights", []),
            "follow_ups": intent.get("follow_ups", []),
            "pattern_count": len(intent["patterns"])
        }
        for pattern in intent["patterns"]:
            raw_texts.append(pattern)
            raw_labels.append(idx)
            texts.append(pattern)
            labels.append(idx)
            # Conversational prefix augmentation for English multi-word queries
            for prefix in spoken_prefixes[1:]:
                if len(pattern.split()) >= 2 and tag not in ("greeting", "thanks_goodbye"):
                    texts.append(prefix + pattern)
                    labels.append(idx)

    y = np.array(labels)
    raw_y = np.array(raw_labels)

    print(f"Base Patterns: {len(raw_texts)} | Augmented Training Samples: {len(texts)}")

    # Fit TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        analyzer=combined_analyzer,
        sublinear_tf=True,
        norm="l2",
        max_features=8000,
        min_df=1
    )

    X = vectorizer.fit_transform(texts)
    vocab = vectorizer.vocabulary_
    idf = vectorizer.idf_
    print(f"Extracted Multilingual Vocabulary: {len(vocab)} features")

    # 5-Fold Stratified Cross Validation on base patterns
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_pipe = make_pipeline(
        TfidfVectorizer(analyzer=combined_analyzer, sublinear_tf=True, norm="l2", max_features=8000, min_df=1),
        LogisticRegression(C=15.0, max_iter=1000, class_weight="balanced", random_state=42)
    )
    cv_scores = cross_val_score(cv_pipe, raw_texts, raw_y, cv=cv, scoring="accuracy")

    # Train final classifier
    clf = LogisticRegression(C=15.0, max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X, y)

    train_preds = clf.predict(X)
    train_acc = accuracy_score(y, train_preds)
    train_f1 = f1_score(y, train_preds, average="macro")

    print(f"Training Accuracy: {train_acc * 100:.2f}% | Macro F1: {train_f1 * 100:.2f}%")
    print(f"5-Fold Stratified CV Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

    # Multi-language sanity checks
    multilingual_queries = [
        ("What is Singular Value Decomposition (SVD)?", "numerical_analysis"),
        ("Explain Dijkstra's shortest path algorithm", "algorithms_complexity"),
        ("¿Cómo funciona la retropropagación en redes neuronales?", "machine_learning_dl"),
        ("什么是Transformer架构和自注意力机制？", "machine_learning_dl"),
        ("डाइक्स्ट्रा एल्गोरिदम और टाइम कॉम्प्लेक्सिटी क्या है?", "algorithms_complexity"),
        ("Qu'est-ce que le théorème CAP en systèmes distribués?", "systems_distributed_computing"),
        ("Wie funktioniert die RSA Verschlüsselung?", "cryptography_security"),
        ("Что такое машина Тьюринга?", "theoretical_cs_discrete_math"),
        ("How does OpenAI Whisper perform Speech-to-Text?", "speech_ai_asr"),
        ("Calculate Word Error Rate (WER)", "wer_evaluation"),
        ("Bonjour assistant!", "greeting"),
        ("नमस्ते चैटबॉट", "greeting"),
        ("你好", "greeting"),
        ("Hola, ¿qué idiomas hablas?", "language_support")
    ]

    print("\nSanity-Checking Multi-Language & CS/Math Inference:")
    class_tags = [intent["tag"] for intent in intents]
    for q, expected in multilingual_queries:
        xq = vectorizer.transform([q])
        probs = clf.predict_proba(xq)[0]
        best_idx = int(np.argmax(probs))
        predicted = class_tags[best_idx]
        match = "✓" if predicted == expected else "✗"
        print(f"  [{match}] '{q[:48]}' -> [{predicted}] ({probs[best_idx] * 100:.1f}%)")

    # Format model JSON
    clean_vocab = {str(token): int(feat_idx) for token, feat_idx in vocab.items()}
    weights_matrix = [[round(float(w), 5) for w in row] for row in clf.coef_]
    biases = [round(float(b), 5) for b in clf.intercept_]
    idf_list = [round(float(v), 5) for v in idf]

    exported_model = {
        "model_info": {
            "name": "VoxASR-OmniKnowledge-v2",
            "version": "2.0.0",
            "trained_with": "Python 3.12 + scikit-learn + NumPy",
            "architecture": "Sublinear Multilingual TF-IDF (Word 1-2g + Char 3-4g + CJK) + L2 Softmax Classifier",
            "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "num_classes": len(intents),
            "base_patterns": len(raw_texts),
            "augmented_samples": len(texts),
            "vocab_size": len(clean_vocab),
            "scraped_articles_included": scraped_count,
            "train_accuracy": round(float(train_acc), 4),
            "macro_f1": round(float(train_f1), 4),
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4)
        },
        "classes": class_tags,
        "intents": [intent_meta[t] for t in class_tags],
        "vocabulary": clean_vocab,
        "idf": idf_list,
        "weights": weights_matrix,
        "biases": biases
    }

    MODEL_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(exported_model, f, separators=(",", ":"))

    kb_store.publish_model(MODEL_OUT_PATH, exported_model["model_info"]["trained_at"])

    size_kb = MODEL_OUT_PATH.stat().st_size / 1024
    print(f"\nExported trained model -> {MODEL_OUT_PATH} ({size_kb:.1f} KB)")

    # Record training history
    history_entry = {
        "timestamp": exported_model["model_info"]["trained_at"],
        "num_classes": len(intents),
        "patterns": len(raw_texts),
        "vocab_size": len(clean_vocab),
        "train_accuracy": train_acc,
        "macro_f1": train_f1,
        "scraped_articles": scraped_count
    }
    history = []
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass
    history.append(history_entry)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("=" * 72)
    return exported_model


if __name__ == "__main__":
    train_and_export()
