#!/usr/bin/env python3
"""
VoxASR Knowledge Harvester & ML-DL Continuous Upgrade Daemon
------------------------------------------------------------
Lightweight HTTP daemon listening on 127.0.0.1:3001.
Receives upgrade requests from the webapp UI or API, triggers
real-time web scraping (Wikipedia & ArXiv) and ML-DL retraining,
rebuilds the web worker bundle, and reports real-time metrics.
"""

import json
import logging
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from training import kb_store
from training.auto_train_pipeline import run_pipeline
from training.certified_languages import CERTIFIED_LANGUAGES, get_language_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("UpgradeDaemon")

PORT = int(os.environ.get("UPGRADE_PORT", "3001"))
HOST = os.environ.get("UPGRADE_HOST", "127.0.0.1")

is_training_lock = threading.Lock()
training_status = {
    "is_upgrading": False,
    "last_progress": "Idle",
    "last_report": None
}


def rebuild_webapp():
    """Optional build hook; live server hot-reloads KB & model via manifest.json."""
    if os.environ.get("REBUILD_ON_UPGRADE") != "1":
        return True
    try:
        logger.info("Rebuilding webapp bundle...")
        npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
        res = subprocess.run([npm_cmd, "run", "build"], cwd=str(ROOT_DIR), capture_output=True, text=True, timeout=60)
        if res.returncode == 0:
            logger.info("Bundle rebuilt successfully!")
            return True
        logger.warning(f"Rebuild returned non-zero code: {res.stderr}")
    except Exception as ex:
        logger.warning(f"Failed to rebuild bundle: {ex}")
    return False


def get_current_stats():
    manifest = kb_store.load_manifest()
    model_path = ROOT_DIR / "src" / "model" / "chatbot_model.json"
    history_path = ROOT_DIR / "src" / "model" / "training_history.json"

    model_data = {}
    if model_path.exists():
        try:
            with open(model_path, "r", encoding="utf-8") as f:
                model_data = json.load(f)
        except Exception:
            pass

    history_data = []
    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history_data = json.load(f)
        except Exception:
            pass

    meta = model_data.get("model_info", {})
    kb_domains = manifest.get("domains", {})

    return {
        "ok": True,
        "is_upgrading": training_status["is_upgrading"],
        "last_progress": training_status["last_progress"],
        "model_version": meta.get("version", "2.0.0"),
        "trained_at": meta.get("trained_at", "N/A"),
        "train_accuracy": meta.get("train_accuracy", 0.0),
        "macro_f1": meta.get("macro_f1", 0.0),
        "vocab_size": meta.get("vocab_size", 0),
        "num_classes": meta.get("num_classes", len(model_data.get("classes", []))),
        "base_patterns": meta.get("base_patterns", 0),
        "total_scraped_articles": manifest.get("total_articles") or meta.get("scraped_articles_included", 0),
        "total_scraped_patterns": meta.get("augmented_samples", 0),
        "certified_languages_count": len(CERTIFIED_LANGUAGES),
        "domains_summary": [
            {
                "key": k,
                "title": k.replace("_", " ").title(),
                "article_count": count
            }
            for k, count in kb_domains.items()
        ],
        "history": history_data[-5:]
    }


def execute_upgrade_task(custom_topic=None, lang="en"):
    with is_training_lock:
        training_status["is_upgrading"] = True
        training_status["last_progress"] = "Scraping knowledge from Wikipedia & ArXiv..."
        try:
            logger.info("Executing background continuous upgrade...")
            report = run_pipeline(custom_topic=custom_topic, lang=lang, max_topics_per_domain=1)
            training_status["last_progress"] = "Rebuilding pure JavaScript distribution..."
            rebuild_webapp()
            training_status["last_report"] = report
            training_status["last_progress"] = "Upgrade Complete!"
        except Exception as ex:
            logger.error(f"Error in continuous upgrade: {ex}")
            training_status["last_progress"] = f"Failed: {ex}"
        finally:
            training_status["is_upgrading"] = False


class UpgradeServerHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path in ("/status", "/api/knowledge/status"):
            stats = get_current_stats()
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode("utf-8"))
        elif parsed.path in ("/languages", "/api/languages"):
            langs = get_language_summary()
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "languages": langs}).encode("utf-8"))
        else:
            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_len) if content_len > 0 else b"{}"

        try:
            body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception:
            body = {}

        if parsed.path in ("/upgrade", "/api/knowledge/upgrade"):
            if training_status["is_upgrading"]:
                self.send_response(409)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": "An upgrade job is already running."}).encode("utf-8"))
                return

            # Launch background upgrade thread
            th = threading.Thread(target=execute_upgrade_task, kwargs={"custom_topic": body.get("topic"), "lang": body.get("lang", "en")})
            th.daemon = True
            th.start()

            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({
                "ok": True,
                "message": "Automated web scraping and ML-DL training pipeline initiated in background.",
                "status": "started"
            }).encode("utf-8"))

        elif parsed.path in ("/scrape", "/api/knowledge/scrape"):
            topic = body.get("topic")
            if not topic:
                self.send_response(400)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": "Topic is required"}).encode("utf-8"))
                return

            th = threading.Thread(target=execute_upgrade_task, kwargs={"custom_topic": topic, "lang": body.get("lang", "en")})
            th.daemon = True
            th.start()

            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({
                "ok": True,
                "message": f"Targeted web scraping for topic '{topic}' initiated.",
                "status": "started"
            }).encode("utf-8"))
        else:
            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()


def start_server():
    server = HTTPServer((HOST, PORT), UpgradeServerHandler)
    logger.info(f"VoxASR Knowledge Harvester & Upgrade Server running on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down Upgrade Server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    start_server()
