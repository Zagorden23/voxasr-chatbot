#!/usr/bin/env python3
"""
Automated Upgradation & ML-DL Continuous Training Pipeline
----------------------------------------------------------
Orchestrates:
1. Web Scraping & Knowledge Harvesting (Wikipedia & ArXiv across 12 CS & Math domains)
2. Ingestion into `dynamic_knowledge_pool.json`
3. Multilingual TF-IDF + Softmax Classifier Retraining (`train_chatbot.py`)
4. Model Weights Export to `src/model/chatbot_model.json`
5. Returns JSON execution report for the Web API & CLI.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Add webapp directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from training.knowledge_harvester import harvest_all_domains, harvest_custom_topic
from training.train_chatbot import train_and_export

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AutoTrainPipeline")


def run_pipeline(custom_topic: str = None, lang: str = "en", max_topics_per_domain: int = 2):
    start_time = time.time()
    logger.info("=======================================================")
    logger.info("Starting Automated Knowledge Upgradation & Training...")
    logger.info("=======================================================")

    harvest_result = None
    if custom_topic:
        logger.info(f"Targeted on-demand harvest for topic: '{custom_topic}' (lang={lang})")
        harvest_result = harvest_custom_topic(custom_topic, lang=lang)
    else:
        logger.info(f"Full automated harvest across all 12 CS & Math domains (max_topics={max_topics_per_domain})...")
        harvest_result = harvest_all_domains(max_topics_per_domain=max_topics_per_domain)

    # Trigger ML-DL retraining
    logger.info("Triggering ML-DL Retraining with updated dynamic knowledge pool...")
    trained_model = train_and_export()

    elapsed = round(time.time() - start_time, 2)
    meta = trained_model["model_info"]

    report = {
        "ok": True,
        "status": "success",
        "duration_seconds": elapsed,
        "completed_at": meta["trained_at"],
        "model_version": meta["version"],
        "num_classes": meta["num_classes"],
        "total_patterns": meta["base_patterns"],
        "augmented_samples": meta["augmented_samples"],
        "vocab_size": meta["vocab_size"],
        "scraped_articles": meta["scraped_articles_included"],
        "train_accuracy": meta["train_accuracy"],
        "macro_f1": meta["macro_f1"],
        "cv_accuracy_mean": meta["cv_accuracy_mean"]
    }

    logger.info(f"Automated pipeline completed in {elapsed}s! Model upgraded successfully.")
    return report


if __name__ == "__main__":
    topic_arg = sys.argv[1] if len(sys.argv) > 1 else None
    res = run_pipeline(custom_topic=topic_arg, max_topics_per_domain=1)
    print(json.dumps(res, indent=2))
