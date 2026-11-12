"""
Knowledge Base Store
--------------------
Single source of truth for the chatbot's knowledge pool.

The pool lives as compact JSON shards under `public/static/kb/` together with a
`manifest.json`. The running web server reads the manifest, loads the shards
into an in-memory search index, and re-reads them whenever the manifest version
changes — so harvesting or retraining never needs a rebuild or a restart.

Compact article format (short keys keep the shards small):
    t  title            x  text (lead extract)
    d  domain tag       a  aliases (redirect titles, acronyms)
    u  url              s  source          l  language
`u`, `s` and `l` are omitted for English Wikipedia articles (the URL is derived
from the title).
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
KB_DIR = ROOT_DIR / "public" / "static" / "kb"
DIST_KB_DIR = ROOT_DIR / "dist" / "static" / "kb"
MANIFEST_PATH = KB_DIR / "manifest.json"
LEARNED_PATH = ROOT_DIR / "training" / "learned_knowledge.json"

SHARD_SIZE = 1500
MODEL_FILE = "chatbot_model.json"


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
    os.replace(tmp, path)


def load_manifest() -> Dict:
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"version": 0, "total_articles": 0, "shards": [], "domains": {}, "model": None}


def load_kb() -> Dict[str, Dict]:
    """Loads every shard into an ordered {title: article} dict."""
    articles: Dict[str, Dict] = {}
    manifest = load_manifest()
    for shard in manifest.get("shards", []):
        path = KB_DIR / shard["file"]
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for art in json.load(f):
                    if art.get("t") and art.get("x"):
                        articles[art["t"]] = art
        except Exception:
            continue
    return articles


def mirror_to_dist(files: Optional[Iterable[str]] = None) -> None:
    """Copies the pool into `dist/` so an already-running server sees it at once."""
    if not DIST_KB_DIR.parent.parent.exists():
        return
    DIST_KB_DIR.mkdir(parents=True, exist_ok=True)
    names = list(files) if files is not None else [p.name for p in KB_DIR.glob("*.json")]
    # Manifest goes last: the server only switches once every shard is in place.
    for name in sorted(names, key=lambda n: n == "manifest.json"):
        src = KB_DIR / name
        if src.exists():
            tmp = DIST_KB_DIR / (name + ".tmp")
            shutil.copyfile(src, tmp)
            os.replace(tmp, DIST_KB_DIR / name)
    if files is None:
        keep = set(names)
        for stale in DIST_KB_DIR.glob("kb-*.json"):
            if stale.name not in keep:
                stale.unlink(missing_ok=True)


def save_kb(articles: Dict[str, Dict], note: str = "") -> Dict:
    """Writes the pool as shards plus a manifest with a bumped version."""
    KB_DIR.mkdir(parents=True, exist_ok=True)
    previous = load_manifest()
    items: List[Dict] = list(articles.values())

    shards = []
    domains: Dict[str, int] = {}
    sources: Dict[str, int] = {}
    for art in items:
        domains[art.get("d", "general")] = domains.get(art.get("d", "general"), 0) + 1
        src = art.get("s", "Wikipedia")
        sources[src] = sources.get(src, 0) + 1

    written = set()
    for idx in range(0, max(len(items), 1), SHARD_SIZE):
        chunk = items[idx: idx + SHARD_SIZE]
        name = f"kb-{idx // SHARD_SIZE:03d}.json"
        payload = json.dumps(chunk, ensure_ascii=False, separators=(",", ":"))
        _atomic_write(KB_DIR / name, payload)
        shards.append({"file": name, "count": len(chunk), "bytes": len(payload.encode("utf-8"))})
        written.add(name)

    for stale in KB_DIR.glob("kb-*.json"):
        if stale.name not in written:
            stale.unlink(missing_ok=True)

    manifest = {
        "version": int(time.time()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_articles": len(items),
        "total_bytes": sum(s["bytes"] for s in shards),
        "shards": shards,
        "domains": dict(sorted(domains.items(), key=lambda kv: -kv[1])),
        "sources": sources,
        "model": previous.get("model"),
        "note": note,
    }
    _atomic_write(MANIFEST_PATH, json.dumps(manifest, ensure_ascii=False, indent=2))
    mirror_to_dist()
    return manifest


def publish_model(model_path: Path, trained_at: str) -> None:
    """Publishes freshly trained classifier weights next to the pool for hot reload."""
    KB_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_path, KB_DIR / MODEL_FILE)
    manifest = load_manifest()
    manifest["model"] = {"file": MODEL_FILE, "trained_at": trained_at}
    manifest["version"] = max(int(time.time()), int(manifest.get("version", 0)) + 1)
    _atomic_write(MANIFEST_PATH, json.dumps(manifest, ensure_ascii=False, indent=2))
    mirror_to_dist([MODEL_FILE, "manifest.json"])


def load_learned() -> List[Dict]:
    if LEARNED_PATH.exists():
        try:
            with open(LEARNED_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            pass
    return []


def append_learned(doc: Dict) -> int:
    """Records an article the live server learned from the web. Returns the queue size."""
    title = str(doc.get("title") or "").strip()
    text = str(doc.get("text") or "").strip()
    if not title or len(text) < 40:
        return len(load_learned())
    learned = [d for d in load_learned() if d.get("title") != title]
    learned.append({
        "title": title[:200],
        "text": text[:6000],
        "url": str(doc.get("url") or "")[:500],
        "source": str(doc.get("source") or "Web")[:60],
        "lang": str(doc.get("lang") or "en")[:12],
        "aliases": [str(a)[:120] for a in (doc.get("aliases") or [])][:12],
        "learned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    _atomic_write(LEARNED_PATH, json.dumps(learned[-5000:], ensure_ascii=False, indent=1))
    return len(learned)
