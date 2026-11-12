#!/usr/bin/env python3
"""
Python NLP Intent Classifier Training Script for VoxASR Chatbot
---------------------------------------------------------------
Trains a Multi-Class Softmax Neural / Logistic Classifier on ASR domain intents
using Word (1,2)-grams + Character (3,4)-grams TF-IDF features (robust to ASR
speech-to-text transcription variations & homophones) and exports portable JSON
weights (`src/model/chatbot_model.json`) for sub-millisecond pure JavaScript
inference on Cloudflare Workers.
"""

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score


ROOT_DIR = Path(__file__).resolve().parent.parent
INTENTS_PATH = ROOT_DIR / "training" / "intents.json"
MODEL_OUT_PATH = ROOT_DIR / "src" / "model" / "chatbot_model.json"


def normalize_text(text: str) -> str:
    """Identical normalization logic replicated in JavaScript inference."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_word_ngrams(text: str):
    """Extract word 1-grams and 2-grams."""
    words = text.split() if text else []
    tokens = list(words)
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]}_{words[i + 1]}")
    return tokens


def extract_char_ngrams(text: str):
    """Extract per-word character 3-grams and 4-grams (robust to ASR phonetic typos)."""
    chars = []
    for word in text.split():
        if len(word) < 2:
            continue
        padded = f"<{word}>"
        for n in (3, 4):
            for i in range(max(0, len(padded) - n + 1)):
                chars.append(f"c:{padded[i:i + n]}")
    return chars


def combined_analyzer(raw_text: str):
    norm = normalize_text(raw_text)
    return extract_word_ngrams(norm) + extract_char_ngrams(norm)


def train_and_export():
    print("=" * 68)
    print("VoxASR Python Chatbot Training Pipeline")
    print("=" * 68)

    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    intents = dataset["intents"]
    texts = []
    labels = []
    intent_meta = {}

    # Data augmentation: include slight conversational prefixes common in spoken ASR
    spoken_prefixes = ["", "please ", "hey can you ", "tell me "]

    raw_texts = []
    raw_labels = []

    for idx, intent in enumerate(intents):
        tag = intent["tag"]
        intent_meta[tag] = {
            "index": idx,
            "tag": tag,
            "title": intent["title"],
            "responses": intent["responses"],
            "follow_ups": intent.get("follow_ups", []),
            "pattern_count": len(intent["patterns"])
        }
        for pattern in intent["patterns"]:
            raw_texts.append(pattern)
            raw_labels.append(idx)
            for prefix in spoken_prefixes[1:]:
                if len(pattern.split()) >= 2 and tag not in ("greeting", "thanks_goodbye"):
                    texts.append(prefix + pattern)
                    labels.append(idx)
            texts.append(pattern)
            labels.append(idx)

    y = np.array(labels)
    raw_y = np.array(raw_labels)

    print(f"Loaded {len(intents)} intent classes")
    print(f"Base patterns: {len(raw_texts)} | Augmented training samples: {len(texts)}")

    # Fit TF-IDF Vectorizer using custom analyzer so JS can reproduce exact TF-IDF vectors
    vectorizer = TfidfVectorizer(
        analyzer=combined_analyzer,
        sublinear_tf=True,
        norm="l2",
        max_features=1600,
        min_df=1
    )

    X = vectorizer.fit_transform(texts)
    X_raw = vectorizer.transform(raw_texts)
    vocab = vectorizer.vocabulary_
    idf = vectorizer.idf_

    print(f"Extracted vocabulary size: {len(vocab)} features (word 1-2 grams + char 3-4 grams)")

    # Cross-validation on raw un-augmented patterns using a leak-free Pipeline
    from sklearn.pipeline import make_pipeline
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_pipe = make_pipeline(
        TfidfVectorizer(analyzer=combined_analyzer, sublinear_tf=True, norm="l2", max_features=1600, min_df=1),
        LogisticRegression(C=10.0, max_iter=1000, random_state=42)
    )
    cv_scores = cross_val_score(cv_pipe, raw_texts, raw_y, cv=cv, scoring="accuracy")

    # Train final Softmax classifier on full augmented set
    clf = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
    clf.fit(X, y)

    train_preds = clf.predict(X)
    train_acc = accuracy_score(y, train_preds)
    train_f1 = f1_score(y, train_preds, average="macro")

    print(f"Training Accuracy: {train_acc * 100:.2f}% | Macro F1: {train_f1 * 100:.2f}%")
    print(f"5-Fold Stratified CV Accuracy (base patterns): {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

    # Verify sample ASR queries (including slight ASR phonetic typos!)
    sample_queries = [
        "what is automatic speech recognition",
        "how does openai whisper work",
        "calculate word error rate wer",
        "explain conformer and ctc loss",
        "what are mfcc and log mel spectrograms",
        "voice activity detection silero vad",
        "wav2vec 2.0 self supervised audio",
        "web speech api microphone javascript"
    ]
    print("\n Sanity-Checking Inference on Sample Spoken Queries:")
    class_tags = [intent["tag"] for intent in intents]
    for q in sample_queries:
        xq = vectorizer.transform([q])
        probs = clf.predict_proba(xq)[0]
        best_idx = int(np.argmax(probs))
        print(f"  - '{q}' -> [{class_tags[best_idx]}] ({probs[best_idx] * 100:.1f}%)")

    # Convert numpy int64 indices to native Python int for JSON serialization
    clean_vocab = {str(token): int(feat_idx) for token, feat_idx in vocab.items()}

    # Round floats to 5 decimals to keep worker bundle compact (<150 KB)
    weights_matrix = [[round(float(w), 5) for w in row] for row in clf.coef_]
    biases = [round(float(b), 5) for b in clf.intercept_]
    idf_list = [round(float(v), 5) for v in idf]

    exported_model = {
        "model_info": {
            "name": "VoxASR-IntentNet-v1",
            "trained_with": "Python 3 + scikit-learn + NumPy",
            "architecture": "Sublinear TF-IDF (Word 1-2g + Char 3-4g) + L2 Softmax Classifier",
            "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "num_classes": len(intents),
            "base_patterns": len(raw_texts),
            "augmented_samples": len(texts),
            "vocab_size": len(clean_vocab),
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

    size_kb = MODEL_OUT_PATH.stat().st_size / 1024
    print(f"\nExported trained model -> {MODEL_OUT_PATH} ({size_kb:.1f} KB)")
    print("=" * 68)


if __name__ == "__main__":
    train_and_export()
