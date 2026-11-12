#!/usr/bin/env python3
"""
Automated Knowledge Harvester & Web Scraping Engine (v3)
--------------------------------------------------------
Builds and continuously grows the chatbot's knowledge pool from Wikipedia:

  * crawls the category trees of every Computer Science / Mathematics domain,
  * pulls Wikipedia's "Vital articles" lists for broad general knowledge,
  * downloads lead extracts in batches of 20 over a small thread pool,
  * records redirect titles as aliases (so "SVD", "TCP", "CNN" resolve),
  * merges articles the live server learned by itself from the web,
  * writes everything as shards under `public/static/kb/` (see kb_store.py).

Usage:
    python training/knowledge_harvester.py                 # full harvest
    python training/knowledge_harvester.py --max 3000      # smaller pool
    python training/knowledge_harvester.py --topic "Raft"  # one topic
    python training/knowledge_harvester.py --expand 400    # grow by ~400 articles
"""

import argparse
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from training import kb_store  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("KnowledgeHarvester")

USER_AGENT = "KnowledgeHarvesterBot/3.0 (educational chatbot knowledge base; python-requests)"
MAX_TEXT_CHARS = 1000
BATCH_SIZE = 20
WORKERS = 3
SKIP_TITLE_PREFIXES = ("List of ", "Lists of ", "Outline of ", "Index of ", "Timeline of ", "Glossary of ", "Comparison of ")

# Domain tag -> curated seed topics + Wikipedia categories to crawl.
# The first 13 tags are also the classifier's intent classes.
DOMAINS: Dict[str, Dict] = {
    "algorithms_complexity": {
        "title": "Algorithms & Computational Complexity",
        "topics": ["Dijkstra's algorithm", "A* search algorithm", "Dynamic programming", "Time complexity",
                   "NP-completeness", "Cook–Levin theorem", "Big O notation", "Quicksort", "Merge sort",
                   "Binary search", "Breadth-first search", "Depth-first search", "Master theorem (analysis of algorithms)",
                   "P versus NP problem", "Greedy algorithm", "Divide-and-conquer algorithm", "Backtracking"],
        "categories": ["Algorithms", "Sorting algorithms", "Search algorithms", "Graph algorithms", "Dynamic programming",
                       "Computational complexity theory", "Analysis of algorithms", "Combinatorial optimization",
                       "Approximation algorithms", "Divide-and-conquer algorithms", "String matching algorithms",
                       "NP-complete problems", "Complexity classes", "Randomized algorithms", "Combinatorial algorithms",
                       "Optimization algorithms and methods", "Computational problems", "Algorithmic information theory"],
    },
    "data_structures": {
        "title": "Advanced Data Structures & Indexing",
        "topics": ["B-tree", "Red–black tree", "Bloom filter", "Disjoint-set data structure", "Segment tree",
                   "Queue (abstract data type)", "Stack (abstract data type)", "Linked list", "Hash table",
                   "Heap (data structure)", "Binary search tree", "Trie", "Graph (abstract data type)", "Array (data structure)",
                   "AVL tree", "Skip list", "Priority queue", "Data structure"],
        "categories": ["Data structures", "Trees (data structures)", "Hashing", "Heaps (data structures)",
                       "Graph data structures", "Linked lists", "Arrays", "Abstract data types", "Search trees",
                       "Binary trees", "Probabilistic data structures", "Priority queues", "Hash tables",
                       "String data structures", "Amortized data structures", "B-tree", "Data types"],
    },
    "numerical_analysis": {
        "title": "Numerical Analysis & Mathematical Computing",
        "topics": ["Singular value decomposition", "QR decomposition", "LU decomposition", "Runge–Kutta methods",
                   "Fast Fourier transform", "Numerical stability", "Newton's method", "Monte Carlo method",
                   "IEEE 754", "Gaussian elimination", "Numerical integration", "Finite element method",
                   "Conjugate gradient method", "Floating-point arithmetic", "Interpolation"],
        "categories": ["Numerical analysis", "Numerical linear algebra", "Numerical differential equations",
                       "Matrix decompositions", "Interpolation", "Root-finding algorithms", "Numerical integration (quadrature)",
                       "Finite element method", "Monte Carlo methods", "Fourier analysis", "Floating point",
                       "Computer arithmetic", "Iterative methods", "Approximation theory", "Computational science",
                       "Computational mathematics", "Mathematical software", "First order methods"],
    },
    "machine_learning_dl": {
        "title": "Machine Learning & Deep Learning",
        "topics": ["Transformer (deep learning architecture)", "Backpropagation", "Stochastic gradient descent",
                   "Convolutional neural network", "Diffusion model", "Reinforcement learning from human feedback",
                   "Machine learning", "Deep learning", "Neural network (machine learning)", "Attention (machine learning)",
                   "Recurrent neural network", "Long short-term memory", "Generative adversarial network",
                   "Support vector machine", "Random forest", "Gradient boosting", "Overfitting", "Reinforcement learning",
                   "Artificial intelligence", "Supervised learning", "Unsupervised learning", "K-means clustering"],
        "categories": ["Machine learning", "Machine learning algorithms", "Deep learning", "Artificial neural networks",
                       "Neural network architectures", "Supervised learning", "Unsupervised learning", "Reinforcement learning",
                       "Classification algorithms", "Cluster analysis algorithms", "Dimension reduction", "Artificial intelligence",
                       "Computer vision", "Generative artificial intelligence", "Statistical classification", "Ensemble learning",
                       "Kernel methods for machine learning", "Bayesian networks", "Loss functions", "Gradient methods",
                       "Machine learning models", "Artificial intelligence engineering", "Applications of artificial intelligence",
                       "Evolutionary algorithms", "Object recognition and categorization", "Image processing"],
    },
    "speech_ai_asr": {
        "title": "Automatic Speech Recognition & Audio AI",
        "topics": ["Speech recognition", "Mel-frequency cepstrum", "Connectionist temporal classification", "Spectrogram",
                   "Voice activity detection", "Whisper (speech recognition system)", "Speech synthesis", "Hidden Markov model",
                   "Word error rate", "Phoneme", "Wav2vec", "Speaker recognition", "Digital signal processing",
                   "Nyquist–Shannon sampling theorem", "Levenshtein distance"],
        "categories": ["Speech recognition", "Speech processing", "Speech synthesis", "Digital signal processing",
                       "Signal processing", "Audio codecs", "Phonetics", "Speech codecs", "Voice technology",
                       "Automatic identification and data capture", "Digital audio", "Audio engineering", "Acoustics",
                       "Statistical signal processing", "Signal processing filter"],
    },
    "theoretical_cs_discrete_math": {
        "title": "Theoretical CS & Discrete Mathematics",
        "topics": ["Turing machine", "Chomsky hierarchy", "Halting problem", "Graph theory", "Modular arithmetic",
                   "Finite-state machine", "Lambda calculus", "Regular expression", "Context-free grammar",
                   "Pushdown automaton", "Church–Turing thesis", "Fermat's little theorem", "Pigeonhole principle",
                   "Boolean algebra", "Combinatorics", "Prime number", "Quantum computing", "Information theory"],
        "categories": ["Theoretical computer science", "Automata (computation)", "Formal languages", "Computability theory",
                       "Models of computation", "Discrete mathematics", "Graph theory", "Combinatorics", "Mathematical logic",
                       "Number theory", "Set theory", "Theory of computation", "Information theory", "Coding theory",
                       "Quantum computing", "Boolean algebra", "Formal methods", "Logic in computer science",
                       "Graph theory objects", "Modular arithmetic", "Prime numbers", "Quantum algorithms",
                       "Quantum information science", "Proof theory", "Recursion"],
    },
    "systems_distributed_computing": {
        "title": "Operating Systems & Distributed Systems",
        "topics": ["CAP theorem", "Raft (algorithm)", "Virtual memory", "Deadlock (computer science)", "Consistent hashing",
                   "Operating system", "Process (computing)", "Thread (computing)", "Paxos (computer science)",
                   "Transmission Control Protocol", "Internet protocol suite", "HTTP", "Domain Name System",
                   "Kernel (operating system)", "Semaphore (programming)", "Mutual exclusion", "Paging",
                   "Scheduling (computing)", "MapReduce", "Microservices", "Load balancing (computing)", "Docker (software)",
                   "Kubernetes", "Linux", "OSI model", "Internet Protocol", "User Datagram Protocol"],
        "categories": ["Operating systems", "Operating system technology", "Distributed computing", "Distributed algorithms",
                       "Concurrency (computer science)", "Concurrency control", "Memory management", "Scheduling algorithms",
                       "Computer networking", "Internet protocols", "Network protocols", "Cloud computing",
                       "Distributed data storage", "Computer file systems", "Fault-tolerant computer systems",
                       "Inter-process communication", "Virtualization", "Operating system kernels", "Internet architecture",
                       "Application layer protocols", "Transport layer protocols", "Process (computing)", "Threads (computing)",
                       "Distributed computing architecture", "Computer networks", "Networking hardware", "Internet",
                       "World Wide Web", "Unix", "Linux", "Cloud infrastructure", "Network architecture", "Web technology",
                       "Hypertext Transfer Protocol", "Distributed computing problems", "Synchronization"],
    },
    "compilers_programming_languages": {
        "title": "Compilers & Programming Language Theory",
        "topics": ["LLVM", "Abstract syntax tree", "Static single-assignment form", "Garbage collection (computer science)",
                   "Type inference", "Compiler", "Interpreter (computing)", "Lexical analysis", "Parsing",
                   "Just-in-time compilation", "Python (programming language)", "JavaScript", "Java (programming language)",
                   "C (programming language)", "C++", "Rust (programming language)", "Object-oriented programming",
                   "Functional programming", "Recursion (computer science)", "Closure (computer programming)",
                   "Polymorphism (computer science)", "Programming language", "Software design pattern", "Git"],
        "categories": ["Compilers", "Compiler construction", "Compiler optimizations", "Programming language theory",
                       "Programming languages", "Programming language concepts", "Programming paradigms", "Type theory",
                       "Type systems", "Parsing", "Parsing algorithms", "Programming constructs", "Functional programming",
                       "Object-oriented programming", "Software design patterns", "Software engineering",
                       "Automatic memory management", "Control flow", "Software testing", "Version control",
                       "Programming language topics", "Source code", "Software development", "Web development",
                       "Programming language implementation", "Programming idioms", "Software architecture",
                       "Computer programming", "Programming principles", "Subroutines", "Variable (computer science)",
                       "Object-oriented programming languages", "Procedural programming languages", "Scripting languages",
                       "High-level programming languages", "Software development philosophies", "Software quality",
                       "Debugging", "Programming tools", "Markup languages", "JavaScript", "Web frameworks"],
    },
    "cryptography_security": {
        "title": "Modern Cryptography & Cyber Security",
        "topics": ["RSA cryptosystem", "Elliptic-curve cryptography", "Zero-knowledge proof", "Advanced Encryption Standard",
                   "Post-quantum cryptography", "Public-key cryptography", "Diffie–Hellman key exchange",
                   "Cryptographic hash function", "SHA-2", "Digital signature", "Transport Layer Security", "Blockchain",
                   "Encryption", "Man-in-the-middle attack", "SQL injection", "Cross-site scripting", "Firewall (computing)"],
        "categories": ["Cryptography", "Cryptographic algorithms", "Cryptographic protocols", "Public-key cryptography",
                       "Symmetric-key algorithms", "Cryptographic hash functions", "Block ciphers", "Computer security",
                       "Computer network security", "Cryptographic attacks", "Post-quantum cryptography",
                       "Authentication methods", "Malware", "Computer security exploits", "Digital signature schemes",
                       "Blockchains", "Hacking (computer security)", "Web security exploits", "Key management",
                       "Internet security", "Transport Layer Security", "Cryptographic primitives", "Stream ciphers",
                       "Cryptocurrencies", "Data security", "Access control", "Computer access control",
                       "Types of malware", "Cyberattacks", "Theory of cryptography", "Information privacy"],
    },
    "calculus_linear_algebra_prob": {
        "title": "Linear Algebra, Calculus & Probability for Computing",
        "topics": ["Eigenvalues and eigenvectors", "Jacobian matrix and determinant", "Bayes' theorem", "Markov chain",
                   "Central limit theorem", "Hessian matrix", "Matrix (mathematics)", "Derivative", "Integral",
                   "Gradient", "Determinant", "Vector space", "Normal distribution", "Probability", "Standard deviation",
                   "Taylor series", "Fourier transform", "Linear regression", "Pythagorean theorem", "Calculus",
                   "Linear algebra", "Chain rule", "Fundamental theorem of calculus", "Law of large numbers", "Variance"],
        "categories": ["Linear algebra", "Matrices (mathematics)", "Matrix theory", "Calculus", "Multivariable calculus",
                       "Differential calculus", "Integral calculus", "Probability theory", "Probability distributions",
                       "Statistics", "Statistical theory", "Mathematical analysis", "Differential equations",
                       "Vector calculus", "Mathematical optimization", "Convex optimization", "Stochastic processes",
                       "Markov processes", "Bayesian statistics", "Regression analysis", "Statistical inference",
                       "Real analysis", "Complex analysis", "Abstract algebra", "Geometry", "Trigonometry",
                       "Elementary algebra", "Functions and mappings", "Topology", "Elementary mathematics",
                       "Mathematical theorems", "Probability theorems", "Theorems in calculus", "Theorems in linear algebra",
                       "Continuous distributions", "Discrete distributions", "Summary statistics", "Statistical tests",
                       "Algebra", "Mathematical constants", "Elementary geometry", "Euclidean geometry", "Group theory",
                       "Ordinary differential equations", "Partial differential equations", "Sequences and series",
                       "Integer sequences", "Polynomials", "Elementary special functions", "Fields of mathematics",
                       "Mathematical concepts", "Arithmetic", "Elementary arithmetic", "Numbers", "Mathematical proofs"],
    },
    "gpu_hpc_computing": {
        "title": "GPU Computing & High-Performance Computing",
        "topics": ["CUDA", "Message Passing Interface", "OpenMP", "Basic Linear Algebra Subprograms",
                   "Graphics processing unit", "Parallel computing", "Supercomputer", "Amdahl's law",
                   "Central processing unit", "Single instruction, multiple data", "Cache (computing)", "CPU cache",
                   "Instruction pipelining", "Von Neumann architecture", "Moore's law", "Field-programmable gate array",
                   "Random-access memory", "Computer architecture", "Multi-core processor", "OpenCL"],
        "categories": ["Parallel computing", "GPGPU", "Graphics processing units", "Supercomputers", "Supercomputing",
                       "Concurrent programming languages", "Computer architecture", "Central processing unit",
                       "Computer hardware", "Graphics hardware", "Instruction processing", "Computer memory",
                       "Microprocessors", "SIMD computing", "Computer performance", "Classes of computers",
                       "Digital electronics", "Computer graphics", "3D computer graphics", "Application programming interfaces",
                       "Instruction set architectures", "Computer data storage", "Flynn's taxonomy", "Hardware acceleration",
                       "Computer buses", "Logic gates", "Semiconductor devices", "Computer engineering",
                       "Computer graphics algorithms", "Rendering (computer graphics)", "Embedded systems"],
    },
    "database_vector_search": {
        "title": "Database Internals & Vector Search",
        "topics": ["Log-structured merge-tree", "ACID", "Multiversion concurrency control", "Nearest neighbor search",
                   "Relational database", "SQL", "NoSQL", "Database index", "Database normalization",
                   "Database transaction", "Vector database", "Cosine similarity", "B+ tree", "Join (SQL)",
                   "Two-phase commit protocol", "Data warehouse", "Database", "PostgreSQL", "MongoDB", "Redis"],
        "categories": ["Databases", "Database management systems", "Database theory", "Data management", "SQL", "NoSQL",
                       "Database index techniques", "Transaction processing", "Data modeling", "Information retrieval",
                       "Information retrieval techniques", "Data mining", "Big data", "Data warehousing",
                       "Query languages", "Database algorithms", "Relational database management systems", "Data science",
                       "Data analysis", "Relational model", "Types of databases", "Database normalization",
                       "Information retrieval systems", "Search engine software", "Data engineering",
                       "Distributed data stores", "Structured storage", "Information science", "Data processing"],
    },
    "nlp_large_language_models": {
        "title": "Natural Language Processing & Large Language Models",
        "topics": ["Large language model", "Byte-pair encoding", "Retrieval-augmented generation", "Word embedding",
                   "Natural language processing", "BERT (language model)", "Generative pre-trained transformer",
                   "Word2vec", "Language model", "Prompt engineering", "Machine translation", "Tf–idf",
                   "Named-entity recognition", "Sentiment analysis", "Fine-tuning (deep learning)", "Chatbot",
                   "Tokenization (lexical analysis)", "Okapi BM25", "ChatGPT", "Hallucination (artificial intelligence)"],
        "categories": ["Natural language processing", "Computational linguistics", "Large language models",
                       "Language modeling", "Machine translation", "Tasks of natural language processing", "Chatbots",
                       "Generative pre-trained transformers", "Natural language processing software",
                       "Statistical natural language processing", "Text mining", "Information extraction",
                       "Natural language parsing", "Semantic Web", "Knowledge representation", "Word embedding",
                       "Linguistics", "Syntax", "Semantics", "Grammar", "Corpus linguistics", "Lexicography"],
    },
    # Wider STEM context (knowledge only — not classifier classes).
    "science_general": {
        "title": "Physics, Chemistry, Biology & Engineering",
        "topics": ["Physics", "Chemistry", "Biology", "Photosynthesis", "DNA", "Evolution", "General relativity",
                   "Quantum mechanics", "Thermodynamics", "Electromagnetism", "Periodic table", "Cell (biology)",
                   "Newton's laws of motion", "Entropy", "Electricity", "Semiconductor", "Transistor", "Robotics"],
        "categories": ["Concepts in physics", "Physical quantities", "Classical mechanics", "Thermodynamics",
                       "Electromagnetism", "Quantum mechanics", "Chemistry", "Chemical elements", "Biology",
                       "Molecular biology", "Genetics", "Astronomy", "Electrical engineering", "Electronics",
                       "Robotics", "Bioinformatics", "Subfields of physics", "Branches of biology", "Units of measurement",
                       "Scientific method", "Mechanical engineering", "Control theory", "Telecommunications",
                       "Wireless networking", "Mobile technology", "Emerging technologies", "Renewable energy"],
    },
}

VITAL_PAGES = [
    "Wikipedia:Vital articles/Level 3",
    "Wikipedia:Vital articles/Level 4/People",
    "Wikipedia:Vital articles/Level 4/History",
    "Wikipedia:Vital articles/Level 4/Geography",
    "Wikipedia:Vital articles/Level 4/Arts",
    "Wikipedia:Vital articles/Level 4/Philosophy and religion",
    "Wikipedia:Vital articles/Level 4/Everyday life",
    "Wikipedia:Vital articles/Level 4/Society and social sciences",
    "Wikipedia:Vital articles/Level 4/Biology and health sciences",
    "Wikipedia:Vital articles/Level 4/Physical sciences",
    "Wikipedia:Vital articles/Level 4/Technology",
    "Wikipedia:Vital articles/Level 4/Mathematics",
]

_thread_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
        _thread_local.session = s
    return _thread_local.session


class _Pacer:
    """Shared request pacer: spaces calls out across all threads and, when the
    server answers 429, pauses everyone and slows down for the rest of the run."""

    def __init__(self, interval: float = 0.34):
        self.interval = interval
        self.next_slot = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.time()
            slot = max(now, self.next_slot)
            self.next_slot = slot + self.interval
        delay = slot - time.time()
        if delay > 0:
            time.sleep(delay)

    def backoff(self, seconds: float) -> None:
        with self.lock:
            self.next_slot = max(self.next_slot, time.time() + seconds)
            self.interval = min(self.interval * 1.4, 2.0)


_pacer = _Pacer()


def api_get(params: Dict, lang: str = "en", retries: int = 7) -> Dict:
    """One MediaWiki API call, paced and retried politely (honours Retry-After)."""
    url = f"https://{lang}.wikipedia.org/w/api.php"
    query = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
    what = str(params.get("titles") or params.get("cmtitle") or params.get("srsearch") or "")[:60]
    for attempt in range(retries):
        try:
            _pacer.wait()
            resp = _session().get(url, params=query, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("error", {}).get("code") == "maxlag":
                    _pacer.backoff(3 + attempt * 2)
                    continue
                return data
            if resp.status_code in (429, 500, 502, 503, 504):
                try:
                    wait = float(resp.headers.get("Retry-After", ""))
                except ValueError:
                    wait = 3.0 * (attempt + 1)
                wait = min(wait + 1, 60)
                logger.info("Wikipedia API %s for '%s' — pausing %.0fs (pace now %.2fs/request)",
                            resp.status_code, what, wait, _pacer.interval)
                _pacer.backoff(wait)
                continue
            logger.warning("Wikipedia API %s for '%s'", resp.status_code, what)
            return {}
        except Exception as ex:  # network hiccup: back off and retry
            if attempt == retries - 1:
                logger.warning("Wikipedia API failed for '%s': %s", what, ex)
            time.sleep(1.5 * (attempt + 1))
    logger.warning("Wikipedia API gave up on '%s'", what)
    return {}


def api_query_all(params: Dict, lang: str = "en", max_rounds: int = 8) -> Tuple[Dict[int, Dict], List[Dict]]:
    """Runs a query following `continue`, merging page records. Returns (pages, redirect map)."""
    pages: Dict[int, Dict] = {}
    redirects: List[Dict] = []
    cont: Dict = {}
    for _ in range(max_rounds):
        data = api_get({**params, **cont}, lang=lang)
        query = data.get("query", {})
        redirects.extend(query.get("redirects", []))
        for page in query.get("pages", []):
            key = page.get("pageid", page.get("title"))
            if key in pages:
                for field, value in page.items():
                    if isinstance(value, list):
                        pages[key].setdefault(field, []).extend(value)
                    elif field not in pages[key] or not pages[key][field]:
                        pages[key][field] = value
            else:
                pages[key] = page
        if "continue" not in data:
            break
        cont = data["continue"]
    return pages, redirects


_MATH_BLOCK = re.compile(r"(?:\n[ \t]+[^\n{]*)*\n[ \t]+\{\\displaystyle ([^\n]*)\}[ \t]*\n?[ \t]*")
_INLINE_MATH = re.compile(r"\{\\displaystyle ([^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*)\}")
_EMPTY_PARENS = re.compile(r"\s*\((?:[\s,;:–-]|listen|or|and)*\)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[$])")


def clean_extract(raw: str) -> str:
    """Turns a TextExtracts lead into tidy single-spaced prose with $inline math$."""
    if not raw:
        return ""
    text = "\n" + raw.replace("\r", "").replace("⁠", "").replace("​", "")
    text = _MATH_BLOCK.sub(lambda m: f" ${m.group(1).strip()}$ ", text)
    text = _INLINE_MATH.sub(lambda m: f" ${m.group(1).strip()}$ ", text)
    # Bullet-list items arrive as bare lines: close them so they read as sentences.
    text = re.sub(r"(?<=[^\s.!?:;,$])\n+(?=\S)", ". ", text.strip())
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _EMPTY_PARENS.sub("", text)
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    return text.strip()


def truncate_at_sentence(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    ends = [m.start() for m in _SENTENCE_END.finditer(cut)]
    if ends and ends[-1] > limit * 0.45:
        return cut[: ends[-1]].strip()
    return cut.rsplit(" ", 1)[0].rstrip(",;: ") + "…"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


_ACRONYM = re.compile(r"\((?:or |also |abbreviated (?:as )?|commonly )?([A-Z][A-Za-z0-9+#&/.-]{1,9})[,;)]")


def pick_aliases(title: str, redirect_titles: Iterable[str], text: str = "", limit: int = 6) -> List[str]:
    """Keeps only trustworthy alternative names: acronyms introduced in the lead
    ("... decomposition (SVD) ..."), redirects that the lead itself uses, and
    spacing variants of the title ("Quick sort" for "Quicksort")."""
    base = _norm(title)
    bare = _norm(re.sub(r"\s*\([^)]*\)$", "", title))
    seen = {base, bare}
    padded = f" {_norm(text[:700])} "
    out: List[str] = []
    redirect_set = set(redirect_titles)
    redirect_norms = {_norm(r) for r in redirect_set}
    initials = "".join(w[0] for w in re.findall(r"[A-Za-z0-9]+", re.sub(r"\s*\([^)]*\)$", "", title))).lower()

    # An acronym in the lead counts only if it abbreviates this title or redirects here —
    # otherwise "(IP)" inside the TCP article would hijack questions about IP.
    for m in _ACRONYM.finditer(text[:400]):
        acr = m.group(1).rstrip(".")
        n = _norm(acr)
        if n in seen or sum(c.isupper() for c in acr) < 2:
            continue
        if n in redirect_norms or n.replace(" ", "") == initials:
            seen.add(n)
            out.append(acr)

    for r in sorted(redirect_set, key=len):
        if not r or len(r) > 48 or ":" in r or "(" in r:
            continue
        n = _norm(r)
        if len(n) < 2 or n in seen:
            continue
        spacing_variant = n.replace(" ", "") == bare.replace(" ", "")
        if spacing_variant or (f" {n} " in padded and (len(n) > 3 or r.isupper())):
            seen.add(n)
            out.append(r)
    return out[:limit]


def category_members(category: str, lang: str = "en") -> Tuple[List[str], List[str]]:
    pages, subcats = [], []
    cont: Dict = {}
    for _ in range(6):
        data = api_get({"action": "query", "list": "categorymembers", "cmtitle": f"Category:{category}",
                        "cmlimit": "500", "cmtype": "page|subcat", **cont}, lang=lang)
        for m in data.get("query", {}).get("categorymembers", []):
            if m.get("ns") == 0:
                pages.append(m["title"])
            elif m.get("ns") == 14:
                subcats.append(m["title"].split(":", 1)[1])
        if "continue" not in data:
            break
        cont = data["continue"]
    return pages, subcats


def crawl_domain(domain_key: str, cap: int, depth: int = 1, max_subcats: int = 30) -> List[str]:
    """Collects article titles for one domain: seed topics, then its category tree (breadth-first)."""
    cfg = DOMAINS[domain_key]
    titles: List[str] = list(cfg["topics"])
    if len(titles) >= cap:
        return titles[:cap]
    seen = set(titles)
    queue = [(c, 0) for c in cfg["categories"]]
    visited = set()
    subcats_used = 0
    while queue and len(titles) < cap:
        cat, level = queue.pop(0)
        if cat in visited:
            continue
        visited.add(cat)
        pages, subcats = category_members(cat)
        for p in pages:
            if p not in seen and not p.startswith(SKIP_TITLE_PREFIXES):
                seen.add(p)
                titles.append(p)
        if level < depth:
            for sc in subcats:
                if subcats_used >= max_subcats:
                    break
                if sc not in visited and not re.search(r"stubs?$|by |images|people|researchers|scientists|companies|"
                                                       r"journals|conferences|awards|organizations|books|software$|"
                                                       r"films|history of|video games|fiction", sc, re.I):
                    queue.append((sc, level + 1))
                    subcats_used += 1
    return titles[:cap]


def vital_titles() -> List[str]:
    """Titles from Wikipedia's Vital Articles lists — broad general knowledge."""
    titles: List[str] = []
    seen = set()
    for page in VITAL_PAGES:
        cont: Dict = {}
        count = 0
        for _ in range(12):
            data = api_get({"action": "query", "prop": "links", "titles": page, "plnamespace": "0",
                            "pllimit": "500", "redirects": "1", **cont})
            for p in data.get("query", {}).get("pages", []):
                for link in p.get("links", []):
                    t = link["title"]
                    if t not in seen and not t.startswith(SKIP_TITLE_PREFIXES):
                        seen.add(t)
                        titles.append(t)
                        count += 1
            if "continue" not in data:
                break
            cont = data["continue"]
        logger.info("  vital list %-44s -> %d titles", page.split("articles/")[-1], count)
    return titles


def fetch_batch(titles: List[str], lang: str = "en") -> List[Dict]:
    """Downloads lead extracts (+ redirect aliases) for up to 20 titles."""
    pages, redirects = api_query_all({
        "action": "query", "prop": "extracts|pageprops|redirects", "exintro": "1", "explaintext": "1",
        "exlimit": str(BATCH_SIZE), "ppprop": "disambiguation", "rdlimit": "500", "rdnamespace": "0",
        "redirects": "1", "titles": "|".join(titles),
    }, lang=lang)
    requested_via = {}
    for r in redirects:
        requested_via.setdefault(r.get("to"), []).append(r.get("from"))

    docs = []
    for page in pages.values():
        title = page.get("title", "")
        if page.get("missing") or "disambiguation" in (page.get("pageprops") or {}) or not title:
            continue
        if title.startswith(SKIP_TITLE_PREFIXES):
            continue
        text = truncate_at_sentence(clean_extract(page.get("extract", "")))
        if len(text) < 80:
            continue
        redirect_titles = [r.get("title", "") for r in page.get("redirects", [])] + requested_via.get(title, [])
        doc = {"t": title, "x": text}
        aliases = pick_aliases(title, redirect_titles, text)
        if aliases:
            doc["a"] = aliases
        if lang != "en":
            doc["l"] = lang
            doc["u"] = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
        docs.append(doc)
    return docs


def fetch_articles(titles: List[str], domain_of: Optional[Dict[str, str]] = None, lang: str = "en") -> List[Dict]:
    """Fetches many articles concurrently, tagging each with its domain."""
    domain_of = domain_of or {}
    batches = [titles[i: i + BATCH_SIZE] for i in range(0, len(titles), BATCH_SIZE)]
    docs: List[Dict] = []
    done = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_batch, b, lang): b for b in batches}
        for fut in as_completed(futures):
            batch = futures[fut]
            try:
                got = fut.result()
            except Exception as ex:
                logger.warning("Batch failed: %s", ex)
                got = []
            batch_domain = next((domain_of[t] for t in batch if t in domain_of), None)
            for d in got:
                dom = domain_of.get(d["t"]) or batch_domain
                if dom:
                    d["d"] = dom
            docs.extend(got)
            done += 1
            if done % 25 == 0 or done == len(batches):
                logger.info("  fetched %d/%d batches | %d articles | %.0fs", done, len(batches), len(docs), time.time() - started)
    return docs


def merge_learned(articles: Dict[str, Dict]) -> int:
    """Folds articles the live server learned from the web into the permanent pool."""
    added = 0
    for doc in kb_store.load_learned():
        title = doc.get("title")
        if not title or not doc.get("text"):
            continue
        existing = articles.get(title)
        entry = {"t": title, "x": truncate_at_sentence(doc["text"], 2400)}
        aliases = list(dict.fromkeys((existing or {}).get("a", []) + (doc.get("aliases") or [])))[:10]
        if aliases:
            entry["a"] = aliases
        if existing and existing.get("d"):
            entry["d"] = existing["d"]
        src = doc.get("source") or "Web"
        lang = doc.get("lang") or "en"
        if not (src.startswith("Wikipedia") and lang == "en"):
            entry["s"] = src
            if doc.get("url"):
                entry["u"] = doc["url"]
        elif existing is None:
            entry["s"] = "Wikipedia (learned)"
        if lang != "en":
            entry["l"] = lang
            if doc.get("url"):
                entry["u"] = doc["url"]
        if existing is None or len(entry["x"]) > len(existing.get("x", "")):
            articles[title] = entry
            added += 1 if existing is None else 0
    return added


def harvest_all_domains(
    max_articles: int = 20000,
    per_domain: int = 800,
    include_vital: bool = True,
    max_topics_per_domain: Optional[int] = None,
) -> Dict:
    """Full harvest: every CS/Math domain plus general-knowledge vital articles.
    When `max_topics_per_domain` is provided (e.g. from quick UI auto-upgrade),
    runs a fast seed-only / incremental expansion instead of a multi-minute full crawl.
    """
    if max_topics_per_domain is not None and max_topics_per_domain <= 5:
        existing = kb_store.load_kb()
        if existing:
            return expand_pool(max_new=max(25, max_topics_per_domain * len(DOMAINS)))
        per_domain = max(5, max_topics_per_domain * 4)
        include_vital = False

    started = time.time()
    logger.info("Starting full knowledge harvest (target <= %d articles)...", max_articles)
    articles = kb_store.load_kb()
    logger.info("Existing pool: %d articles", len(articles))

    domain_of: Dict[str, str] = {}
    ordered: List[str] = []
    for key in DOMAINS:
        titles = crawl_domain(key, cap=per_domain)
        fresh = 0
        for t in titles:
            if t not in domain_of:
                domain_of[t] = key
                ordered.append(t)
                fresh += 1
        logger.info("Domain %-34s -> %4d titles (%d new)", key, len(titles), fresh)

    if include_vital:
        logger.info("Collecting Wikipedia Vital Articles (general knowledge)...")
        for t in vital_titles():
            if t not in domain_of:
                domain_of[t] = "general"
                ordered.append(t)

    wanted = [t for t in ordered if t not in articles][: max(0, max_articles - len(articles))]
    logger.info("Titles collected: %d | to download: %d", len(ordered), len(wanted))

    # Batches are grouped by domain so redirected titles still inherit the right tag.
    by_domain: Dict[str, List[str]] = {}
    for t in wanted:
        by_domain.setdefault(domain_of[t], []).append(t)
    for dom, titles in by_domain.items():
        # Downloaded in chunks with a checkpoint after each, so an interrupted
        # harvest keeps everything fetched so far and simply resumes next run.
        for start in range(0, len(titles), 2000):
            chunk = titles[start: start + 2000]
            logger.info("Downloading %d articles for [%s] (%d/%d)...", len(chunk), dom, start + len(chunk), len(titles))
            for doc in fetch_articles(chunk, {t: dom for t in chunk}):
                if doc["t"] not in articles:
                    doc.setdefault("d", dom)
                    if doc["d"] == "general":
                        doc.pop("d")
                    articles[doc["t"]] = doc
            kb_store.save_kb(articles, note="harvest in progress")
            logger.info("Checkpoint saved: %d articles", len(articles))

    learned = merge_learned(articles)
    manifest = kb_store.save_kb(articles, note="full harvest")
    elapsed = round(time.time() - started, 1)
    logger.info("Harvest complete: %d articles (%d self-learned) in %ss -> %s",
                manifest["total_articles"], learned, elapsed, kb_store.KB_DIR)
    return {"ok": True, "total_articles": manifest["total_articles"], "learned_merged": learned,
            "duration_seconds": elapsed, "version": manifest["version"]}


def expand_pool(max_new: int = 400) -> Dict:
    """Incremental growth for the auto-upgrade loop: self-learned articles, the pages they
    link to, and deeper levels of the domain category trees."""
    started = time.time()
    articles = kb_store.load_kb()
    before = len(articles)
    learned = merge_learned(articles)

    candidates: List[str] = []
    domain_of: Dict[str, str] = {}

    # 1) Topics adjacent to what users actually asked about.
    recent = [d["title"] for d in kb_store.load_learned()[-25:] if d.get("lang", "en") == "en"]
    for i in range(0, len(recent), 10):
        pages, _ = api_query_all({"action": "query", "prop": "links", "plnamespace": "0", "pllimit": "200",
                                  "titles": "|".join(recent[i: i + 10])}, max_rounds=2)
        for page in pages.values():
            for link in page.get("links", [])[:40]:
                candidates.append(link["title"])

    # 2) One level deeper into a rotating slice of domains.
    keys = list(DOMAINS)
    offset = int(time.time() // 3600) % len(keys)
    for key in (keys[offset:] + keys[:offset])[:4]:
        for t in crawl_domain(key, cap=2500, depth=2, max_subcats=60):
            domain_of.setdefault(t, key)
            candidates.append(t)

    wanted, seen = [], set()
    for t in candidates:
        if t not in articles and t not in seen and not t.startswith(SKIP_TITLE_PREFIXES):
            seen.add(t)
            wanted.append(t)
        if len(wanted) >= max_new:
            break

    by_domain: Dict[str, List[str]] = {}
    for t in wanted:
        by_domain.setdefault(domain_of.get(t, "general"), []).append(t)
    for dom, titles in by_domain.items():
        for doc in fetch_articles(titles, {t: dom for t in titles} if dom != "general" else None):
            articles.setdefault(doc["t"], doc)

    manifest = kb_store.save_kb(articles, note="auto-upgrade")
    return {"ok": True, "total_articles": manifest["total_articles"], "new_articles": len(articles) - before,
            "learned_merged": learned, "duration_seconds": round(time.time() - started, 1),
            "version": manifest["version"]}


def search_wikipedia(query: str, lang: str = "en", limit: int = 3) -> List[str]:
    data = api_get({"action": "query", "list": "search", "srsearch": query, "srlimit": str(limit)}, lang=lang)
    return [hit["title"] for hit in data.get("query", {}).get("search", [])]


def harvest_custom_topic(topic: str, domain_key: Optional[str] = None, lang: str = "en") -> Dict:
    """Scrapes one on-demand topic (exact title first, then full-text search) into the pool."""
    topic = (topic or "").strip()
    if not topic:
        return {"ok": False, "error": "Topic is required"}
    logger.info("On-demand scraping for topic: %s (lang=%s)", topic, lang)
    docs = fetch_batch([topic], lang=lang)
    if not docs:
        hits = search_wikipedia(topic, lang=lang)
        if hits:
            docs = fetch_batch(hits[:3], lang=lang)
            order = {t: i for i, t in enumerate(hits)}
            docs.sort(key=lambda d: order.get(d["t"], 99))
    if not docs:
        return {"ok": False, "error": f"Could not find an authoritative article for '{topic}'"}

    doc = docs[0]
    if domain_key in DOMAINS:
        doc["d"] = domain_key
    if _norm(topic) != _norm(doc["t"]):
        doc["a"] = list(dict.fromkeys(doc.get("a", []) + [topic]))[:10]
    articles = kb_store.load_kb()
    articles[doc["t"]] = {**articles.get(doc["t"], {}), **doc}
    manifest = kb_store.save_kb(articles, note=f"on-demand: {topic}")
    return {"ok": True, "total_articles": manifest["total_articles"], "version": manifest["version"],
            "article": {"title": doc["t"], "summary": doc["x"], "source": "Wikipedia (On-Demand)",
                        "url": doc.get("u") or f"https://en.wikipedia.org/wiki/{doc['t'].replace(' ', '_')}"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harvest the chatbot knowledge pool from Wikipedia")
    parser.add_argument("--max", type=int, default=20000, help="maximum articles in the pool")
    parser.add_argument("--per-domain", type=int, default=800, help="article cap per CS/Math domain")
    parser.add_argument("--no-vital", action="store_true", help="skip the general-knowledge vital articles")
    parser.add_argument("--topic", type=str, help="scrape a single topic")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--expand", type=int, help="grow the existing pool by about N articles")
    args = parser.parse_args()

    if args.topic:
        result = harvest_custom_topic(args.topic, lang=args.lang)
    elif args.expand:
        result = expand_pool(max_new=args.expand)
    else:
        result = harvest_all_domains(max_articles=args.max, per_domain=args.per_domain, include_vital=not args.no_vital)
    print({k: v for k, v in result.items() if k != "article"})
