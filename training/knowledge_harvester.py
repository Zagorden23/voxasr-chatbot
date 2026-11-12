"""
Automated Knowledge Harvester & Web Scraping Engine
---------------------------------------------------
Continuously ingests, scrapes, and synthesizes authoritative Computer Science,
Mathematical Computing, and AI/ML knowledge from Wikipedia REST API, ArXiv Atom API,
and curated educational repositories.

Maintains `training/dynamic_knowledge_pool.json`, auto-generating multilingual
training patterns and richly formatted mathematical/computational responses.
"""

import json
import logging
import os
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("KnowledgeHarvester")

ROOT_DIR = Path(__file__).resolve().parent.parent
DYNAMIC_POOL_PATH = ROOT_DIR / "training" / "dynamic_knowledge_pool.json"

USER_AGENT = "VoxASR-KnowledgeHarvester/2.0 (mailto:admin@voxasr.local; Educational & Research)"

# Curated CS & Mathematical Computing topics mapped to canonical sources
CORE_CS_MATH_DOMAINS = {
    "algorithms_complexity": {
        "title": "Algorithms & Computational Complexity",
        "wiki_topics": [
            "Dijkstra's_algorithm",
            "A*_search_algorithm",
            "Dynamic_programming",
            "Time_complexity",
            "NP-completeness",
            "Cook–Levin_theorem"
        ],
        "arxiv_query": "cat:cs.DS+OR+cat:cs.CC"
    },
    "data_structures": {
        "title": "Advanced Data Structures & Indexing",
        "wiki_topics": [
            "B-tree",
            "Red–black_tree",
            "Bloom_filter",
            "Disjoint-set_data_structure",
            "Segment_tree"
        ],
        "arxiv_query": "cat:cs.DS"
    },
    "numerical_analysis": {
        "title": "Numerical Analysis & Mathematical Computing",
        "wiki_topics": [
            "Singular_value_decomposition",
            "QR_decomposition",
            "Runge–Kutta_methods",
            "Fast_Fourier_transform",
            "Numerical_stability",
            "Newton's_method"
        ],
        "arxiv_query": "cat:math.NA"
    },
    "machine_learning_dl": {
        "title": "Machine Learning & Deep Learning",
        "wiki_topics": [
            "Transformer_(deep_learning_architecture)",
            "Backpropagation",
            "Stochastic_gradient_descent",
            "Convolutional_neural_network",
            "Diffusion_model",
            "Reinforcement_learning_from_human_feedback"
        ],
        "arxiv_query": "cat:cs.LG+OR+cat:cs.AI"
    },
    "speech_ai_asr": {
        "title": "Automatic Speech Recognition & Audio AI",
        "wiki_topics": [
            "Speech_recognition",
            "Mel-frequency_cepstrum",
            "Connectionist_temporal_classification",
            "Spectrogram",
            "Voice_activity_detection"
        ],
        "arxiv_query": "cat:cs.SD+OR+cat:cs.CL"
    },
    "theoretical_cs_discrete_math": {
        "title": "Theoretical CS & Discrete Mathematics",
        "wiki_topics": [
            "Turing_machine",
            "Chomsky_hierarchy",
            "Halting_problem",
            "Graph_theory",
            "Modular_arithmetic"
        ],
        "arxiv_query": "cat:math.CO+OR+cat:cs.DM"
    },
    "systems_distributed_computing": {
        "title": "Operating Systems & Distributed Systems",
        "wiki_topics": [
            "CAP_theorem",
            "Raft_(algorithm)",
            "Virtual_memory",
            "Deadlock",
            "Consistent_hashing"
        ],
        "arxiv_query": "cat:cs.DC+OR+cat:cs.OS"
    },
    "compilers_programming_languages": {
        "title": "Compilers & Programming Language Theory",
        "wiki_topics": [
            "LLVM",
            "Abstract_syntax_tree",
            "Static_single-assignment_form",
            "Garbage_collection_(computer_science)",
            "Type_inference"
        ],
        "arxiv_query": "cat:cs.PL"
    },
    "cryptography_security": {
        "title": "Modern Cryptography & Cyber Security",
        "wiki_topics": [
            "RSA_(cryptosystem)",
            "Elliptic-curve_cryptography",
            "Zero-knowledge_proof",
            "Advanced_Encryption_Standard",
            "Post-quantum_cryptography"
        ],
        "arxiv_query": "cat:cs.CR"
    },
    "calculus_linear_algebra": {
        "title": "Linear Algebra, Calculus & Probability for Computing",
        "wiki_topics": [
            "Eigenvalues_and_eigenvectors",
            "Jacobian_matrix_and_determinant",
            "Bayes'_theorem",
            "Markov_chain",
            "Central_limit_theorem"
        ],
        "arxiv_query": "cat:math.PR+OR+cat:math.OC"
    },
    "gpu_hpc_computing": {
        "title": "GPU Computing & High-Performance Computing",
        "wiki_topics": [
            "CUDA",
            "Message_Passing_Interface",
            "OpenMP",
            "Basic_Linear_Algebra_Subprograms",
            "Graphics_processing_unit"
        ],
        "arxiv_query": "cat:cs.DC"
    },
    "database_vector_search": {
        "title": "Database Internals & Vector Search",
        "wiki_topics": [
            "Log-structured_merge-tree",
            "ACID",
            "Multi-version_concurrency_control",
            "Nearest_neighbor_search",
            "Relational_database"
        ],
        "arxiv_query": "cat:cs.DB"
    }
}


def clean_scraped_text(text: str) -> str:
    """Removes reference tags, clean whitespace and formatting."""
    if not text:
        return ""
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_wikipedia_summary(topic: str, lang: str = "en") -> Optional[Dict]:
    """Fetches verified article summary from Wikipedia REST API."""
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{topic}"
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "title": data.get("title", topic.replace("_", " ")),
                "extract": clean_scraped_text(data.get("extract", "")),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", url),
                "lang": lang
            }
        else:
            logger.warning(f"Wikipedia API returned {resp.status_code} for topic {topic} ({lang})")
    except Exception as ex:
        logger.warning(f"Failed to fetch Wikipedia topic {topic}: {ex}")
    return None


def fetch_arxiv_papers(query: str, max_results: int = 3) -> List[Dict]:
    """Fetches recent preprints from ArXiv Atom API."""
    url = f"https://export.arxiv.org/api/query?search_query={query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    papers = []
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        if resp.status_code == 200:
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                title_elem = entry.find("atom:title", ns)
                summary_elem = entry.find("atom:summary", ns)
                id_elem = entry.find("atom:id", ns)
                if title_elem is not None and summary_elem is not None:
                    papers.append({
                        "title": clean_scraped_text(title_elem.text),
                        "summary": clean_scraped_text(summary_elem.text)[:450] + "...",
                        "url": id_elem.text.strip() if id_elem is not None else ""
                    })
    except Exception as ex:
        logger.warning(f"Failed to fetch ArXiv query {query}: {ex}")
    return papers


def generate_multilingual_patterns_for_topic(title: str, domain_key: str) -> List[str]:
    """Generates multilingual question variations for a given CS/Math topic."""
    norm_title = title.lower()
    patterns = [
        f"what is {norm_title}",
        f"explain {norm_title}",
        f"how does {norm_title} work",
        f"definition of {norm_title}",
        f"algorithm of {norm_title}",
        f"mathematical principles of {norm_title}",
        norm_title
    ]

    # Multilingual query templates for certified languages
    multilingual_templates = [
        # Spanish
        f"qué es {norm_title}",
        f"explica {norm_title}",
        f"cómo funciona {norm_title}",
        # French
        f"qu'est-ce que {norm_title}",
        f"expliquez {norm_title}",
        f"comment fonctionne {norm_title}",
        # German
        f"was ist {norm_title}",
        f"erkläre {norm_title}",
        f"wie funktioniert {norm_title}",
        # Chinese (Simplified)
        f"什么是{norm_title}",
        f"解释{norm_title}",
        f"{norm_title}的原理",
        # Hindi
        f"{norm_title} क्या है",
        f"{norm_title} कैसे काम करता है",
        f"{norm_title} समझाइए",
        # Russian
        f"что такое {norm_title}",
        f"объясните {norm_title}",
        # Arabic
        f"ما هو {norm_title}",
        f"اشرح {norm_title}",
        # Japanese
        f"{norm_title}とは",
        f"{norm_title}の仕組み",
        # Portuguese
        f"o que é {norm_title}",
        f"como funciona {norm_title}"
    ]
    patterns.extend(multilingual_templates)
    return list(dict.fromkeys(patterns))


def harvest_all_domains(max_topics_per_domain: int = 4) -> Dict:
    """Runs a full harvest across all CS & Mathematical Computing domains."""
    logger.info("Starting automated web scraping & knowledge ingestion pipeline...")
    harvested = {
        "metadata": {
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_domains": len(CORE_CS_MATH_DOMAINS),
            "harvester_version": "2.0.0"
        },
        "domains": {}
    }

    total_articles = 0
    total_patterns = 0

    for domain_key, info in CORE_CS_MATH_DOMAINS.items():
        logger.info(f"Scraping domain: {info['title']} ({domain_key})")
        domain_articles = []

        # Scrape Wikipedia topics
        topics_to_fetch = info["wiki_topics"][:max_topics_per_domain]
        for topic in topics_to_fetch:
            wiki_data = fetch_wikipedia_summary(topic, lang="en")
            if wiki_data and wiki_data["extract"]:
                patterns = generate_multilingual_patterns_for_topic(wiki_data["title"], domain_key)
                domain_articles.append({
                    "topic_id": f"{domain_key}_{topic.lower()}",
                    "title": wiki_data["title"],
                    "source": "Wikipedia",
                    "url": wiki_data["url"],
                    "summary": wiki_data["extract"],
                    "patterns": patterns
                })
                total_patterns += len(patterns)
                total_articles += 1
            time.sleep(0.3)  # Respectful crawling rate limit

        # Scrape ArXiv preprints
        if info.get("arxiv_query"):
            arxiv_papers = fetch_arxiv_papers(info["arxiv_query"], max_results=2)
            for paper in arxiv_papers:
                pats = generate_multilingual_patterns_for_topic(paper["title"][:40], domain_key)
                domain_articles.append({
                    "topic_id": f"arxiv_{domain_key}_{abs(hash(paper['title'])) % 100000}",
                    "title": paper["title"],
                    "source": "ArXiv",
                    "url": paper["url"],
                    "summary": paper["summary"],
                    "patterns": pats
                })
                total_patterns += len(pats)
                total_articles += 1

        harvested["domains"][domain_key] = {
            "title": info["title"],
            "article_count": len(domain_articles),
            "articles": domain_articles
        }

    harvested["metadata"]["total_articles"] = total_articles
    harvested["metadata"]["total_patterns"] = total_patterns

    # Save to dynamic knowledge pool
    DYNAMIC_POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DYNAMIC_POOL_PATH, "w", encoding="utf-8") as f:
        json.dump(harvested, f, indent=2, ensure_ascii=False)

    logger.info(f"Harvest complete! Ingested {total_articles} articles with {total_patterns} multilingual patterns -> {DYNAMIC_POOL_PATH}")
    return harvested


def harvest_custom_topic(topic: str, domain_key: str = "computer_science_general", lang: str = "en") -> Dict:
    """Scrapes an on-demand topic entered by the user or triggered by an API request."""
    logger.info(f"On-demand scraping for topic: {topic} (lang={lang})")
    formatted_topic = topic.strip().replace(" ", "_")
    wiki_data = fetch_wikipedia_summary(formatted_topic, lang=lang)

    if not wiki_data or not wiki_data["extract"]:
        # Fallback search if exact title did not hit
        search_url = f"https://{lang}.wikipedia.org/w/api.php?action=opensearch&search={topic}&limit=1&format=json"
        try:
            s_resp = requests.get(search_url, headers={"User-Agent": USER_AGENT}, timeout=8).json()
            if len(s_resp) >= 2 and s_resp[1]:
                best_match = s_resp[1][0]
                wiki_data = fetch_wikipedia_summary(best_match.replace(" ", "_"), lang=lang)
        except Exception:
            pass

    if not wiki_data or not wiki_data["extract"]:
        return {"ok": False, "error": f"Could not find authoritative article for '{topic}'"}

    # Load existing pool
    current_pool = {}
    if DYNAMIC_POOL_PATH.exists():
        try:
            with open(DYNAMIC_POOL_PATH, "r", encoding="utf-8") as f:
                current_pool = json.load(f)
        except Exception:
            pass

    if "domains" not in current_pool:
        current_pool["domains"] = {}

    if domain_key not in current_pool["domains"]:
        current_pool["domains"][domain_key] = {
            "title": domain_key.replace("_", " ").title(),
            "article_count": 0,
            "articles": []
        }

    patterns = generate_multilingual_patterns_for_topic(wiki_data["title"], domain_key)
    article_entry = {
        "topic_id": f"{domain_key}_{formatted_topic.lower()}",
        "title": wiki_data["title"],
        "source": "Wikipedia (On-Demand)",
        "url": wiki_data["url"],
        "summary": wiki_data["extract"],
        "patterns": patterns,
        "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    # Deduplicate
    articles = [a for a in current_pool["domains"][domain_key].get("articles", []) if a["title"] != wiki_data["title"]]
    articles.append(article_entry)
    current_pool["domains"][domain_key]["articles"] = articles
    current_pool["domains"][domain_key]["article_count"] = len(articles)

    with open(DYNAMIC_POOL_PATH, "w", encoding="utf-8") as f:
        json.dump(current_pool, f, indent=2, ensure_ascii=False)

    return {"ok": True, "article": article_entry}


if __name__ == "__main__":
    harvest_all_domains(max_topics_per_domain=3)
