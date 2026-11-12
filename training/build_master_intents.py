"""
Master Intent Corpus Generator for VoxASR Chatbot (v4.0 — Omni-Domain World Knowledge)
Covers 35 Certified World Languages + Computer Science, Mathematics, Physics, Biology,
Engineering, Commerce, Arts, Sports, Music, Dance, Entertainment, History & Society.
"""

import json
from pathlib import Path
from certified_languages import CERTIFIED_LANGUAGES, get_all_greetings

ROOT_DIR = Path(__file__).resolve().parent.parent
INTENTS_PATH = ROOT_DIR / "training" / "intents.json"

def build_intents():
    greetings_patterns = get_all_greetings()
    greetings_patterns.extend([
        "hello voxasr", "hi assistant", "voice chatbot", "hey chatbot",
        "can you hear me", "test audio", "audio check", "are you online"
    ])

    intents = [
        {
            "tag": "greeting",
            "title": "Greeting & Certified Language Check",
            "patterns": greetings_patterns,
            "responses": [
                "Hello! I am **VoxASR**, an AI Knowledge Browser & Voice Assistant across **35 certified languages** covering every major human discipline: **Science & Medicine, Mathematics & CS, Engineering, Commerce & Economics, Arts & Literature, Music & Dance, Sports, Entertainment & Cinema, History & Geography**.\n\n- **Complete Web Browser**: Ask in brief (`\"in short\"`) or in depth (`\"in detail\"`), and I browse multiple live websites, synthesize facts, and cite sources.\n- **Neural Intelligence**: Powered by a Multi-Scale Convolutional Neural Network (**TextCNN**) + Recursive Action-Reaction Introspection Engine.",
                "Greetings! Voice and text channels are active across 35 certified languages. Ask me anything across Science, Sports, Music, Dance, Commerce, Arts, Engineering, Entertainment, or CS/Math — in brief summary or deep detailed breakdown!"
            ],
            "follow_ups": [
                "Explain Quantum Mechanics in detail",
                "Explain Cricket rules in brief",
                "Compare Classical Ballet and Bharatanatyam",
                "How does compound interest work in finance?"
            ]
        },
        {
            "tag": "language_support",
            "title": "Certified World Languages & Speech Locales",
            "patterns": [
                "what languages do you speak", "what languages do you understand", "certified languages",
                "multilingual support", "can you understand hindi", "can you speak spanish",
                "can you understand chinese", "list supported languages", "quelles langues parlez-vous",
                "qué idiomas hablas", "welche sprachen sprichst du", "какие языки вы поддерживаете",
                "ما هي اللغات التي تفهمها", "你能说哪些语言", "आप कौन सी भाषाएं समझते हैं",
                "quais idiomas você fala", "どの言語を理解できますか"
            ],
            "responses": [
                "I support all certified international languages recognized across the UN and global digital standards (ISO-639 / BCP-47):\n\n- **Western & Central European**: English (`en-US`, `en-GB`, `en-IN`), Spanish (`es-ES`, `es-MX`), French (`fr-FR`), German (`de-DE`), Italian (`it-IT`), Portuguese (`pt-BR`), Dutch (`nl-NL`), Polish (`pl-PL`), Czech (`cs-CZ`), Swedish (`sv-SE`).\n- **Asian & Indic**: Mandarin Chinese (`zh-CN`, `zh-TW`), Hindi (`hi-IN`), Bengali (`bn-IN`), Japanese (`ja-JP`), Korean (`ko-KR`), Vietnamese (`vi-VN`), Indonesian (`id-ID`), Tamil (`ta-IN`), Telugu (`te-IN`), Marathi (`mr-IN`), Thai (`th-TH`).\n- **Middle Eastern & Eurasian**: Arabic (`ar-SA`, `ar-EG`), Russian (`ru-RU`), Ukrainian (`uk-UA`), Turkish (`tr-TR`), Persian/Farsi (`fa-IR`), Urdu (`ur-PK`), Hebrew (`he-IL`).\n- **African**: Swahili (`sw-KE`).\n\nYou can select any certified locale in the **Recognition Locale** dropdown to speak live via Speech-to-Text and hear responses via Text-to-Speech!"
            ],
            "follow_ups": [
                "How does multilingual ASR work?",
                "Explain Unicode normalization for NLP",
                "How do you train multilingual tokenizers?"
            ]
        },
        {
            "tag": "algorithms_complexity",
            "title": "Algorithms & Computational Complexity",
            "patterns": [
                "what is dijkstra algorithm", "explain dijkstras algorithm", "how does a star search work",
                "what is dynamic programming", "explain dynamic programming", "what is time complexity",
                "what is big o notation", "explain p vs np", "what is np completeness",
                "cook levin theorem", "what is divide and conquer", "master theorem for recurrences",
                "sorting algorithms comparison", "quicksort vs mergesort", "tarjan strongly connected components",
                "floyd warshall algorithm", "bellman ford algorithm", "amortized time complexity",
                "qué es complejidad temporal", "algoritmo de dijkstra", "qué es p contra np",
                "什么是时间复杂度", "解释动态规划", "什么是p与np问题",
                "डायनेमिक प्रोग्रामिंग क्या है", "डाइक्स्ट्रा एल्गोरिदम क्या है", "पी बनाम एनपी क्या है",
                "qu'est-ce que la programmation dynamique", "complexité algorithmique grand o"
            ],
            "responses": [
                "**Algorithms & Computational Complexity** formalize the resource bounds (time and space) of computable functions.\n\n### 1. Asymptotic Complexity (Big-O)\n- **$O(g(n))$**: Upper asymptotic bound ($f(n) \\le c \\cdot g(n)$ for large $n$).\n- **$\\Omega(g(n))$**: Lower asymptotic bound.\n- **$\\Theta(g(n))$**: Tight asymptotic bound ($c_1 g(n) \\le f(n) \\le c_2 g(n)$).\n\n### 2. Core Algorithmic Paradigms\n- **Dynamic Programming (DP)**: Solves problems exhibiting **optimal substructure** and **overlapping subproblems** (e.g. Bellman-Ford $O(VE)$, Knapsack $O(nW)$, Floyd-Warshall $O(V^3)$).\n- **Shortest Paths**: Dijkstra's algorithm uses a Fibonacci / Min-Heap priority queue to achieve $O(|E| + |V| \\log |V|)$ non-negative edge relaxation.\n- **A* Search**: Heuristic search using evaluation function $f(n) = g(n) + h(n)$, optimal when $h(n)$ is admissible ($h(n) \\le h^*(n)$) and consistent.\n\n### 3. Complexity Classes: P vs. NP\n- **P**: Decision problems solvable in polynomial time $O(n^k)$ by a Deterministic Turing Machine (DTM).\n- **NP**: Problems verifiable in polynomial time by a DTM, or decidable in polynomial time by a Non-Deterministic Turing Machine (NTM).\n- **NP-Complete (Cook-Levin Theorem)**: The hardest problems in NP (e.g. 3-SAT, Clique, Hamiltonian Cycle, TSP). If any NP-Complete problem is in P, then $P = NP$."
            ],
            "follow_ups": [
                "Explain Dynamic Programming with an example",
                "Compare Dijkstra and A* search",
                "What is the Master Theorem?"
            ]
        },
        {
            "tag": "data_structures",
            "title": "Advanced Data Structures & Indexing",
            "patterns": [
                "what is a b tree", "explain b trees and b plus trees", "what is a red black tree",
                "avl tree vs red black tree", "what is a bloom filter", "how do bloom filters work",
                "disjoint set union find", "explain segment tree", "what is a fenwick tree binary indexed tree",
                "how do hash tables work", "cuckoo hashing vs robin hood hashing", "what is a trie",
                "suffix automaton and suffix tree", "hnsw vector indexing",
                "qué es un árbol b", "cómo funciona un filtro de bloom", "estructuras de datos avanzadas",
                "什么是红黑树", "解释布隆过滤器", "什么是b树和b加树",
                "रेड ब्लैक ट्री क्या है", "ब्लूम फ़िल्टर कैसे काम करता है", "डेटा संरचनाएं क्या हैं"
            ],
            "responses": [
                "**Advanced Data Structures** govern spatial locality, branch factor, and logarithmic lookup invariants:\n\n1. **B-Trees & B+ Trees**:\n   - Self-balancing $M$-way search trees designed for block storage and database disk pages.\n   - **B+ Tree**: Internal nodes hold keys only for routing; all actual values reside in linked leaf nodes, enabling $O(\\log_B N)$ random lookup and ultra-fast $O(K)$ sequential range scans.\n2. **Red-Black Trees**:\n   - Balanced binary search tree maintaining invariants: Root is black, no two consecutive red nodes, every root-to-leaf path contains equal black count.\n   - Guarantees max height $\\le 2 \\log_2(n+1)$, ensuring $O(\\log n)$ search, insertion, and deletion.\n3. **Bloom Filters**:\n   - Space-efficient probabilistic data structure answering set membership with **zero false negatives** and tunable false positive probability $p \\approx (1 - e^{-km/n})^k$.\n   - Uses $k$ independent hash functions mapped to an $m$-bit array.\n4. **Vector Search Indexes (HNSW)**:\n   - Hierarchical Navigable Small World graphs provide $O(\\log N)$ Approximate Nearest Neighbor (ANN) search for high-dimensional embeddings."
            ],
            "follow_ups": [
                "Why do databases use B+ trees instead of B-trees?",
                "How do you choose optimal bit-size for a Bloom Filter?",
                "Explain HNSW vector indexing"
            ]
        },
        {
            "tag": "numerical_analysis",
            "title": "Numerical Analysis & Mathematical Computing",
            "patterns": [
                "what is singular value decomposition svd", "explain svd", "singular value decomposition formula",
                "what is qr decomposition", "lu decomposition with pivoting", "runge kutta method rk4",
                "explain numerical ode solvers", "fast fourier transform fft cooley tukey",
                "numerical stability and condition number", "newton raphson method", "floating point precision ieee 754",
                "catastrophic cancellation in numerical computing", "monte carlo integration",
                "cholesky decomposition positive definite", "eigenvalue power iteration qr algorithm",
                "qué es la descomposición en valores singulares svd", "método de runge kutta",
                "什么是奇异值分解", "解释快速傅里叶变换fft", "数值分析与科学计算",
                "सिंगुलर वैल्यू डिकम्पोज़िशन क्या है", "फास्ट फूरियर ट्रांसफॉर्म समझाइए", "न्यूमेरिकल एनालिसिस क्या है"
            ],
            "responses": [
                "**Numerical Analysis & Mathematical Computing** bridges continuous mathematics with discrete, finite-precision floating-point hardware.\n\n### 1. Singular Value Decomposition (SVD)\nEvery real matrix $A \\in \\mathbb{R}^{m \\times n}$ factorizes into:\n$$A = U \\Sigma V^T$$\n- $U \\in \\mathbb{R}^{m \\times m}$: Orthonormal left singular vectors (eigenvectors of $A A^T$).\n- $\\Sigma \\in \\mathbb{R}^{m \\times n}$: Diagonal matrix of non-negative singular values $\\sigma_1 \\ge \\sigma_2 \\ge \\dots \\ge \\sigma_r > 0$.\n- $V \\in \\mathbb{R}^{n \\times n}$: Orthonormal right singular vectors (eigenvectors of $A^T A$).\n- **Eckart-Young-Mirsky Theorem**: Truncating to rank $k$ gives the optimal low-rank matrix approximation in Frobenius and spectral norms.\n\n### 2. Matrix Factorizations\n- **LU with Partial Pivoting**: $P A = L U$, solves linear systems $A x = b$ in $\\frac{2}{3}n^3$ FLOPs.\n- **QR Decomposition**: $A = Q R$ via Householder reflections or Givens rotations, foundational for least-squares and computing eigenvalues via the QR algorithm.\n- **Cholesky**: $A = L L^T$ for symmetric positive-definite matrices (twice as fast as LU).\n\n### 3. Differential Equations: Runge-Kutta 4th Order (RK4)\nSolves $y' = f(t, y)$ with local truncation error $O(h^5)$ and global error $O(h^4)$:\n$$y_{n+1} = y_n + \\frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$\n\n### 4. Fast Fourier Transform (Cooley-Tukey FFT)\nReduces Discrete Fourier Transform (DFT) from $O(N^2)$ to $O(N \\log N)$ via divide-and-conquer on odd and even indices."
            ],
            "follow_ups": [
                "How does SVD perform Principal Component Analysis (PCA)?",
                "Explain the Cooley-Tukey FFT algorithm",
                "What causes catastrophic cancellation in IEEE 754?"
            ]
        },
        {
            "tag": "machine_learning_dl",
            "title": "Machine Learning & Deep Learning (ML/DL)",
            "patterns": [
                "how does backpropagation work", "explain backpropagation calculus", "what is gradient descent",
                "adam vs adamw optimizer", "what is self attention mechanism", "transformer architecture explained",
                "convolutional neural network cnn", "what is rotary position embedding rope",
                "diffusion models explained ddpm", "reinforcement learning from human feedback rlhf",
                "direct preference optimization dpo", "loss functions cross entropy mse focal loss",
                "regularization dropout batchnorm layernorm rmsnorm", "activation functions relu gelu silu",
                "cómo funciona la retropropagación backpropagation", "qué es la arquitectura transformer",
                "redes neuronales y aprendizaje profundo", "mecanismo de autoatención",
                "什么是反向传播算法", "解释transformer自注意力机制", "深度学习与神经网络",
                "什么是transformer架构", "transformer架构和自注意力机制", "大语言模型神经网络",
                "बैकप्रॉपैगैशन कैसे काम करता है", "ट्रांसफॉर्मर आर्किटेक्चर क्या है", "मशीन लर्निंग और डीप लर्निंग क्या है",
                "न्यूरल नेटवर्क और बैकप्रॉपैगैशन", "सेल्फ अटेंशन मैकेनिज्म क्या है",
                "comment fonctionne la rétropropagation", "architecture transformeur auto-attention",
                "wie funktioniert backpropagation", "transformer architektur aufmerksamkeitsmechanismus",
                "архитектура трансформер", "механизм самовнимания", "обратное распространение ошибки",
                "обучение нейронных сетей", "глубокое обучение и нейросети",
                "بنية المحولات والذكاء الاصطناعي", "آلية الانتباه الذاتي والشبكات العصبية",
                "トランスフォーマー構造と自己注意機構", "深層学習とニューラルネットワーク"
            ],
            "responses": [
                "**Deep Learning & Neural Networks** learn parameterized non-linear representations through gradient-based optimization.\n\n### 1. Backpropagation & Computational Graphs\nBackpropagation applies the multi-variable chain rule to compute gradients of scalar loss $\\mathcal{L}$ with respect to weights $W$:\n$$\\frac{\\partial \\mathcal{L}}{\\partial W^{[l]}} = \\frac{\\partial \\mathcal{L}}{\\partial Z^{[l]}} (A^{[l-1]})^T, \\quad \\delta^{[l]} = \\frac{\\partial \\mathcal{L}}{\\partial Z^{[l]}} = (W^{[l+1]})^T \\delta^{[l+1]} \\odot \\sigma'(Z^{[l]})$$\n\n### 2. Transformer Self-Attention (Scaled Dot-Product)\n$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V$$\n- **Rotary Position Embedding (RoPE)**: Encodes relative distance by rotating query and key vectors in complex 2D subspaces: $\\langle R_{\\Theta, m}^d q, R_{\\Theta, n}^d k \\rangle = g(q, k, m-n)$.\n- **RMSNorm & SwiGLU**: Modern LLMs replace LayerNorm with Root Mean Square Normalization and ReLU with Swish Gated Linear Units.\n\n### 3. Optimization: AdamW\nAdamW decouples weight decay $\\lambda w_t$ from moving moment updates ($m_t, v_t$):\n$$m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t, \\quad v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2$$\n$$\\theta_{t+1} = \\theta_t - \\eta_t \\left( \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon} + \\lambda \\theta_t \\right)$$"
            ],
            "follow_ups": [
                "What is the difference between LayerNorm and RMSNorm?",
                "How does FlashAttention avoid high HBM memory access?",
                "Explain the SwiGLU activation function"
            ]
        },
        {
            "tag": "speech_ai_asr",
            "title": "Automatic Speech Recognition & Audio AI",
            "patterns": [
                "what is automatic speech recognition", "explain speech recognition", "how does whisper work",
                "openai whisper architecture", "what is conformer", "ctc loss connectionist temporal classification",
                "rnn transducer rnnt vs ctc", "what is wav2vec 2 0", "mel frequency cepstral coefficients mfcc",
                "what is a log mel spectrogram", "voice activity detection vad silero",
                "speech to text audio pipeline", "end to end asr models",
                "qué es el reconocimiento automático del habla asr", "cómo funciona whisper",
                "什么是自动语音识别", "解释whisper语音模型", "梅尔倒谱系数mfcc",
                "स्वचालित वाक् पहचान क्या है", "व्हिस्पर एआई कैसे काम करता है", "ध्वनि स्पेक्ट्रोग्राम क्या है"
            ],
            "responses": [
                "**Automatic Speech Recognition (ASR)** transforms acoustic pressure waves into text tokens through acoustic modeling and language decoding.\n\n### 1. Front-End Audio Feature Extraction\n- Audio sampled at 16 kHz is framed via Short-Time Fourier Transform (STFT) with 25ms windows and 10ms hops.\n- Linear frequency bins pass through triangular Mel-scale filterbanks to produce **80-channel Log-Mel Spectrograms** mimicking human auditory perception.\n\n### 2. State-of-the-Art ASR Architectures\n- **OpenAI Whisper**: Weakly supervised sequence-to-sequence Encoder-Decoder Transformer trained on 680,000+ hours of multilingual audio. Uses learned positional embeddings, cross-attention, and multitask conditioning tokens (`<|transcribe|>`, `<|en|>`).\n- **Conformer (Convolution-augmented Transformer)**: Interleaves Macaron-style Feed-Forward Modules, Multi-Head Self-Attention (global context), and Depthwise Separable Convolutions (local acoustic context).\n- **CTC (Connectionist Temporal Classification)**: Uses a blank token `ε` and dynamic programming forward-backward algorithm to align variable-length audio frames to text without explicit alignments.\n- **Wav2Vec 2.0**: Self-supervised contrastive learning directly on raw audio waveforms, quantizing latent representations with Gumbel-Softmax vector codebooks."
            ],
            "follow_ups": [
                "Compare CTC Loss with RNN-Transducer (RNN-T)",
                "How does Whisper perform multilingual translation?",
                "What is Word Error Rate (WER)?"
            ]
        },
        {
            "tag": "theoretical_cs_discrete_math",
            "title": "Theoretical CS, Automata & Discrete Mathematics",
            "patterns": [
                "what is a turing machine", "explain turing machines", "what is the halting problem",
                "chomsky hierarchy of formal languages", "regular languages vs context free",
                "deterministic finite automata dfa nfa", "pushdown automata pda",
                "graph theory eulerian and hamiltonian", "modular arithmetic and fermat little theorem",
                "chinese remainder theorem", "generating functions combinatorics", "pigeonhole principle",
                "máquina de turing", "problema de la parada halting problem", "jerarquía de chomsky",
                "什么是图灵机", "解释停机问题", "乔姆斯基谱系", "离散数学与自动机",
                "ट्यूरिंग मशीन क्या है", "हॉल्टिंग समस्या क्या है", "असतत गणित क्या है",
                "что такое машина тьюринга", "проблема остановки", "иерархия хомского",
                "теория графов и дискретная математика", "машина тьюринга в информатике",
                "machine de turing et calculabilité", "problème de l'arrêt",
                "turingmaschine und berechenbarkeit", "halteproblem turing",
                "آلة تورنغ ونظرية الحوسبة", "مسألة التوقف",
                "チューリングマシンと計算理論", "停止性問題"
            ],
            "responses": [
                "**Theoretical Computer Science & Discrete Mathematics** delineate the fundamental boundaries of what is mathematically computable.\n\n### 1. Turing Machines & Undecidability\n- **Turing Machine**: Formally defined as a 7-tuple $M = \\langle Q, \\Sigma, \\Gamma, \\delta, q_0, q_{accept}, q_{reject} \\rangle$ with an infinite read/write tape.\n- **Halting Problem ($A_{TM}$)**: Alan Turing proved in 1936 that no general algorithm can decide whether an arbitrary program $M$ halts on input $w$. Proved via **diagonalization**:\n  $$\\text{Assume } H(\\langle M \\rangle, w) \\text{ decides halting. Construct } D(\\langle M \\rangle): \\text{ halts iff } M(M) \\text{ loops forever. } D(D) \\text{ yields a contradiction!}$$\n\n### 2. Chomsky Hierarchy\n1. **Type 3 (Regular)**: Recognized by Finite Automata (DFA/NFA); generated by Regular Grammars ($A \\to aB$).\n2. **Type 2 (Context-Free)**: Recognized by Pushdown Automata (PDA with stack); generated by CFGs ($A \\to \\alpha$).\n3. **Type 1 (Context-Sensitive)**: Recognized by Linear Bounded Automata ($LBA$).\n4. **Type 0 (Recursively Enumerable)**: Recognized by Turing Machines.\n\n### 3. Number Theory: Modular Arithmetic\n- **Fermat's Little Theorem**: If $p$ is prime and $\\gcd(a, p) = 1$, then $a^{p-1} \\equiv 1 \\pmod p$.\n- **Euler's Totient $\\phi(n)$**: Fundamental to RSA key generation, where $a^{\\phi(n)} \\equiv 1 \\pmod n$."
            ],
            "follow_ups": [
                "How does Rice's Theorem generalize the Halting Problem?",
                "What is the difference between DFA and NFA equivalence?",
                "Explain the Chinese Remainder Theorem"
            ]
        },
        {
            "tag": "systems_distributed_computing",
            "title": "Operating Systems & Distributed Systems",
            "patterns": [
                "what is the cap theorem", "explain cap theorem pacelc", "what is raft consensus algorithm",
                "paxos vs raft", "virtual memory and paging", "how does tlb work",
                "process vs thread concurrency", "deadlock conditions coffman", "mutex vs semaphore vs spinlock",
                "consistent hashing in distributed systems", "two phase commit 2pc", "distributed transactions saga pattern",
                "qué es el teorema cap", "algoritmo de consenso raft", "memoria virtual y paginación",
                "什么是cap定理", "解释raft共识算法", "虚拟内存与分页机制",
                "सीएपी प्रमेय क्या है", "राफ्ट सर्वसम्मति एल्गोरिदम क्या है", "ऑपरेटिंग सिस्टम और मेमोरी प्रबंधन"
            ],
            "responses": [
                "**Systems & Distributed Computing** manage concurrency, fault tolerance, and consensus across physical hardware nodes.\n\n### 1. The CAP Theorem & PACELC\n- **CAP Theorem (Brewer)**: A distributed data store can guarantee at most two out of three:\n  - **Consistency (C)**: Every read receives the most recent write or an error.\n  - **Availability (A)**: Every non-failing node returns a response (without guarantee of latest data).\n  - **Partition Tolerance (P)**: The system continues operating despite dropped or delayed network packets.\n- **PACELC Extension**: If Partitioned (**P**), choose between Availability (**A**) and Consistency (**C**); Else (**E**), choose between Latency (**L**) and Consistency (**C**).\n\n### 2. Distributed Consensus: Raft\nDecomposes consensus into three independent sub-problems:\n1. **Leader Election**: Randomized election timers prevent split votes; nodes transition between Follower, Candidate, and Leader.\n2. **Log Replication**: Leader receives commands, appends them to local log, and broadcasts `AppendEntries` RPCs to achieve quorum commitment.\n3. **Safety**: A candidate is only elected if its log is at least as up-to-date as any other node in the majority.\n\n### 3. Virtual Memory & Paging\n- Translates Virtual Addresses to Physical Addresses via multi-level Page Tables (e.g. 4-level x86-64 paging).\n- **Translation Lookaside Buffer (TLB)** caches recent virtual-to-physical translations; TLB misses trigger hardware page table walks."
            ],
            "follow_ups": [
                "What is the difference between Raft and Paxos?",
                "Explain the 4 Coffman conditions for Deadlock",
                "How does Consistent Hashing minimize key remapping?"
            ]
        },
        {
            "tag": "compilers_programming_languages",
            "title": "Compilers & Programming Language Theory",
            "patterns": [
                "how do compilers work", "compiler pipeline phases", "what is an abstract syntax tree ast",
                "llvm intermediate representation ir", "static single assignment ssa form",
                "recursive descent parser vs pratt parser", "type inference hindley milner",
                "garbage collection mark and sweep generational", "jit compilation vs aot",
                "qué es un árbol de sintaxis abstracta ast", "cómo funciona llvm",
                "什么是抽象语法树ast", "解释llvm编译器框架", "垃圾回收机制gc",
                "कंपाइलर कैसे काम करता है", "एब्सट्रैक्ट सिंटैक्स ट्री क्या है", "एलएलवीएम क्या है"
            ],
            "responses": [
                "**Compilers & Language Runtimes** translate high-level semantic code into performant machine binaries or bytecode.\n\n### 1. The Compiler Pipeline\n1. **Lexical Analysis (Lexing)**: Stream of source characters $\\to$ Tokens (using DFAs).\n2. **Syntactic Analysis (Parsing)**: Stream of tokens $\\to$ **Abstract Syntax Tree (AST)** using LL(k), LR(1), or Pratt parsing for expressions.\n3. **Semantic Analysis**: Type checking, symbol table resolution, and scope binding.\n4. **Intermediate Representation (IR)**: Translation into architecture-independent IR, such as **LLVM SSA (Static Single Assignment)** where every variable is assigned exactly once.\n5. **Optimization Passes**: Dead Code Elimination (DCE), Common Subexpression Elimination (CSE), Loop Invariant Code Motion (LICM), inlining, auto-vectorization (SIMD).\n6. **Code Generation & Register Allocation**: Graph coloring algorithms allocate hardware registers before emitting target assembly.\n\n### 2. Type Systems & Hindley-Milner Type Inference\n- Computes the most general principal type for lambda expressions without explicit type annotations using **Algorithm W** and Robinson's First-Order Unification."
            ],
            "follow_ups": [
                "Why is SSA form so useful for compiler optimization?",
                "Compare Generational Garbage Collection with Reference Counting",
                "How does Pratt parsing handle operator precedence?"
            ]
        },
        {
            "tag": "cryptography_security",
            "title": "Modern Cryptography & Cyber Security",
            "patterns": [
                "how does rsa cryptography work", "explain rsa algorithm", "elliptic curve cryptography ecc",
                "aes gcm authenticated encryption", "zero knowledge proofs zk snarks",
                "diffie hellman key exchange", "post quantum cryptography lattice based",
                "sha 256 cryptographic hash function", "tls 1 3 handshake security",
                "wie funktioniert die rsa verschlüsselung", "kryptographie und sicherheit",
                "comment fonctionne le chiffrement rsa", "cryptographie à courbe elliptique",
                "criptografía rsa", "curvas elípticas ecc", "pruebas de conocimiento cero zk",
                "什么是rsa加密", "解释椭圆曲线密码学ecc", "零知识证明zk snarks",
                "आरएसए क्रिप्टोग्राफी क्या है", "एलिप्टिक कर्व क्रिप्टोग्राफी क्या है", "ज़ीरो नॉलेज प्रूफ क्या है"
            ],
            "responses": [
                "**Modern Cryptography** provides confidentiality, integrity, authenticity, and non-repudiation based on computational hardness assumptions.\n\n### 1. RSA Cryptosystem\nBased on the intractability of integer factorization:\n1. Choose two large primes $p, q$. Compute $n = p \\cdot q$ and Euler's totient $\\phi(n) = (p-1)(q-1)$.\n2. Select public exponent $e$ such that $\\gcd(e, \\phi(n)) = 1$ (commonly $e = 65537$).\n3. Compute private key $d \\equiv e^{-1} \\pmod{\\phi(n)}$ using the Extended Euclidean Algorithm.\n- **Encryption**: $c = m^e \\pmod n$\n- **Decryption**: $m = c^d \\pmod n$\n\n### 2. Elliptic Curve Cryptography (ECC)\n- Defined over Weierstrass form: $y^2 = x^3 + ax + b \\pmod p$.\n- Based on the **Elliptic Curve Discrete Logarithm Problem (ECDLP)**: Given base point $G$ and public key $P = d \\cdot G$, finding scalar $d$ requires $O(\\sqrt{p})$ operations (Pollard's rho), achieving 128-bit security with just 256-bit keys (e.g. Curve25519, secp256k1).\n\n### 3. Post-Quantum Cryptography (PQC)\n- Resistant to Shor's algorithm on quantum computers.\n- **Lattice-Based Standards**: ML-KEM (CRYSTALS-Kyber) for key exchange and ML-DSA (CRYSTALS-Dilithium) for digital signatures, founded on the Learning With Errors (LWE) problem."
            ],
            "follow_ups": [
                "How does Shor's algorithm break RSA and ECC?",
                "What is the mathematical foundation of Zero-Knowledge Proofs?",
                "Explain the Diffie-Hellman key exchange"
            ]
        },
        {
            "tag": "calculus_linear_algebra_prob",
            "title": "Linear Algebra, Calculus & Probability for Computing",
            "patterns": [
                "what is a jacobian and hessian matrix", "explain eigenvalues and eigenvectors",
                "what is bayes theorem", "markov chain stationary distribution",
                "central limit theorem explained", "vector space basis and dimension",
                "gradient and directional derivative", "maximum likelihood estimation mle vs map",
                "matriz jacobiana y hessiana", "teorema de bayes", "autovalores y autovectores",
                "什么是雅可比矩阵和黑塞矩阵", "解释特征值和特征向量", "贝叶斯定理与马尔可夫链",
                "आइगेनवैल्यू और आइगेनवेक्टर क्या हैं", "बेयस प्रमेय क्या है", "जैकोबियन और हेसियन मैट्रिक्स क्या है"
            ],
            "responses": [
                "**Calculus, Linear Algebra & Probability** are the analytical bedrock of machine learning, physics simulation, and optimization.\n\n### 1. Derivatives in Vector Spaces\n- **Jacobian Matrix**: First-order partial derivatives of a vector function $f: \\mathbb{R}^n \\to \\mathbb{R}^m$:\n  $$J_{ij} = \\frac{\\partial f_i}{\\partial x_j}$$\n- **Hessian Matrix**: Second-order partial derivatives of a scalar function $f: \\mathbb{R}^n \\to \\mathbb{R}$:\n  $$H_{ij} = \\frac{\\partial^2 f}{\\partial x_i \\partial x_j}$$\n  Defines local curvature: If $H$ is positive definite at a stationary point ($\\nabla f = 0$), the point is a local strict minimum.\n\n### 2. Eigenvalues & Eigenvectors\n$$A v = \\lambda v \\iff (A - \\lambda I) v = 0$$\n- Characteristic polynomial $\\det(A - \\lambda I) = 0$ yields eigenvalues $\\lambda_i$.\n- Symmetric real matrices feature real eigenvalues and orthogonal eigenvectors (Spectral Theorem: $A = Q \\Lambda Q^T$).\n\n### 3. Bayes' Theorem & Markov Chains\n$$P(A|B) = \\frac{P(B|A) P(A)}{P(B)}$$\n- **Markov Property**: $P(X_{t+1} = s | X_t, X_{t-1}, \\dots) = P(X_{t+1} = s | X_t)$.\n- Stationary distribution $\\pi$ satisfies $\\pi P = \\pi$, where $P$ is the transition probability matrix."
            ],
            "follow_ups": [
                "How is the Hessian matrix used in Newton's optimization?",
                "What is the difference between MLE and MAP estimation?",
                "Explain the Spectral Theorem"
            ]
        },
        {
            "tag": "gpu_hpc_computing",
            "title": "GPU Computing & High-Performance Computing",
            "patterns": [
                "how does cuda gpu computing work", "explain cuda warps and thread blocks",
                "what is openmp and mpi", "blas basic linear algebra subprograms gemm",
                "tensor cores systolic arrays", "coalesced memory access gpu", "warp divergence cuda",
                "cómo funciona la computación en gpu cuda", "qué es mpi y openmp",
                "什么是cuda编程", "解释gpu线程块和warp", "高性能计算hpc",
                "जीपीयू कंप्यूटिंग और कुडा क्या है", "ओपनएमपी और एमपीआई क्या है", "हाई परफॉर्मेंस कंप्यूटिंग क्या है"
            ],
            "responses": [
                "**High-Performance Computing (HPC) & GPU Architecture** exploit massive data parallelism through SIMD/SIMT execution.\n\n### 1. NVIDIA CUDA Execution Model\n- **Hierarchy**: Threads are grouped into **Warps** (32 parallel threads executing in lockstep), Warps into **Thread Blocks** (sharing L1 / Shared Memory), and Blocks into a **Grid** dispatched across Streaming Multiprocessors (SMs).\n- **Warp Divergence**: Occurs when threads within a single warp follow different execution branches (`if/else`), serializing execution paths.\n- **Coalesced Memory Access**: Consecutive threads in a warp accessing contiguous 32-, 64-, or 128-byte DRAM segments merge requests into a single memory transaction.\n\n### 2. BLAS & Tensor Hardware\n- **BLAS-3 (GEMM)**: Matrix-matrix multiplication $C = \\alpha A B + \\beta C$ performs $O(N^3)$ operations on $O(N^2)$ data, maximizing arithmetic intensity.\n- **Tensor Cores**: Specialized mixed-precision hardware executing $D = A \\times B + C$ (e.g. FP16/BF16/FP8 multiplication with FP32 accumulation) in single clock cycles."
            ],
            "follow_ups": [
                "How do you prevent warp divergence in CUDA kernels?",
                "What is the difference between OpenMP and MPI?",
                "How do Tensor Cores achieve high TFLOPS?"
            ]
        },
        {
            "tag": "database_vector_search",
            "title": "Database Internals & Vector Search",
            "patterns": [
                "what is an lsm tree log structured merge tree", "acid properties databases",
                "what is mvcc multiversion concurrency control", "vector databases and similarity search",
                "cosine similarity vs euclidean distance", "approximate nearest neighbor ann hnsw",
                "write ahead logging wal", "innodb vs rocksdb",
                "qué son las propiedades acid", "cómo funciona un árbol lsm", "bases de datos vectoriales",
                "什么是acid特性", "解释lsm树存储引擎", "向量数据库与相似度搜索",
                "डेटाबेस में एसीआईडी गुण क्या हैं", "वेक्टर डेटाबेस कैसे काम करता है", "एलएसएम ट्री क्या है"
            ],
            "responses": [
                "**Database Internals & Vector Search** ensure high-throughput persistence, transactional isolation, and high-dimensional semantic retrieval.\n\n### 1. LSM-Trees vs. B+ Trees\n- **B+ Trees (e.g. PostgreSQL, MySQL InnoDB)**: In-place page updates optimized for random reads; writes incur random disk I/O.\n- **Log-Structured Merge-Trees (e.g. RocksDB, Cassandra)**: Converts all writes into sequential I/O via in-memory **MemTable** (SkipList) + sequential **Write-Ahead Log (WAL)**, flushed periodically to immutable sorted string tables (**SSTables**) with background leveled compaction.\n\n### 2. ACID & MVCC\n- **MVCC (Multi-Version Concurrency Control)**: Readers do not lock writers and writers do not lock readers. Each transaction reads a consistent snapshot identified by a Transaction ID / LSN.\n\n### 3. Vector Similarity Search\n- **Distance Metrics**: Cosine Similarity $\\frac{u \\cdot v}{\\|u\\| \\|v\\|}$, Euclidean distance $\\|u - v\\|_2$, Inner Product $\\langle u, v \\rangle$.\n- **HNSW (Hierarchical Navigable Small World)**: Multi-layer proximity graph where upper layers have longer skip connections, enabling $O(\\log N)$ nearest neighbor traversal."
            ],
            "follow_ups": [
                "How does LSM-tree compaction eliminate deleted tombstones?",
                "Explain MVCC snapshot isolation anomalies",
                "Compare HNSW with Inverted File Index (IVF-PQ)"
            ]
        },
        {
            "tag": "nlp_large_language_models",
            "title": "Natural Language Processing & Large Language Models",
            "patterns": [
                "how do large language models work", "what is byte pair encoding bpe tokenization",
                "explain kv caching in transformers", "what is flashattention",
                "retrieval augmented generation rag explained", "model quantization gptq awq gguf",
                "what is context window in llms", "fine tuning lora and qlora",
                "cómo funcionan los modelos de lenguaje llm", "qué es la generación aumentada por recuperación rag",
                "什么是大语言模型llm", "解释bpe分词算法", "什么是检索增强生成rag",
                "लार्ज लैंग्वेज मॉडल कैसे काम करते हैं", "बीपीई टोकनाइजेशन क्या है", "आरएजी क्या है"
            ],
            "responses": [
                "**Large Language Models (LLMs)** are autoregressive transformer decoders predicting the next subword token probability distribution: $P(w_t | w_1, \\dots, w_{t-1})$.\n\n### 1. Subword Tokenization (BPE & WordPiece)\n- **Byte-Pair Encoding (BPE)** starts from individual bytes or characters and iteratively merges the most frequent adjacent symbol pairs until reaching a target vocabulary size (typically 32,000 to 128,000 tokens).\n\n### 2. Inference Optimizations: KV Caching & FlashAttention\n- **KV Cache**: Caches past Key and Value projection tensors across decoding steps, turning $O(N^2)$ generation into $O(N)$ operations per generated token.\n- **FlashAttention (Dao et al.)**: Restructures self-attention computation using GPU SRAM tiling and online softmax rescaling, avoiding $O(N^2)$ reads/writes to slow GPU High Bandwidth Memory (HBM).\n\n### 3. Parameter-Efficient Fine-Tuning (LoRA)\n- Decomposes dense weight update matrices $\\Delta W \\in \\mathbb{R}^{d \\times k}$ into low-rank matrices $B \\cdot A$ with rank $r \\ll \\min(d, k)$, cutting trainable parameters by $99\\%+$ while retaining full model expressiveness."
            ],
            "follow_ups": [
                "How does KV caching work mathematically?",
                "Explain the LoRA low-rank decomposition",
                "What is the difference between Dense Passage Retrieval and BM25 in RAG?"
            ]
        },
        {
            "tag": "wer_evaluation",
            "title": "ASR Accuracy Metrics (WER / CER / Levenshtein)",
            "patterns": [
                "how do you calculate word error rate", "what is wer", "word error rate formula",
                "what is character error rate cer", "levenshtein edit distance asr",
                "substitutions deletions insertions wer", "how to measure speech recognition accuracy",
                "calculate wer", "evaluating asr models",
                "cómo se calcula la tasa de error de palabras wer", "fórmula de wer",
                "如何计算词错误率wer", "解释编辑距离levenshtein", "语音识别准确率评估",
                "वर्ड एरर रेट कैसे मापते हैं", "लेवेनशेटिन दूरी क्या है", "डब्ल्यूईआर क्या है"
            ],
            "responses": [
                "**Word Error Rate (WER)** is the standard benchmark metric for evaluating Automatic Speech Recognition accuracy.\n\n### 1. The WER Mathematical Formula\n$$\\text{WER} = \\frac{S + D + I}{N} = \\frac{S + D + I}{S + D + C}$$\nWhere:\n- **$S$ (Substitutions)**: Spoken words transcribed as incorrect words (e.g., 'wreck a nice beach' instead of 'recognize speech').\n- **$D$ (Deletions)**: Words present in reference that were omitted in hypothesis.\n- **$I$ (Insertions)**: Words hallucinated or added by the model not in reference.\n- **$C$ (Correct Matches)**: Words accurately transcribed.\n- **$N$**: Total word count in reference ($N = S + D + C$).\n\n*Note*: WER can exceed $100\\%$ if insertions outnumber reference words!\n\n### 2. Alignment via Dynamic Programming\nCalculated using the Levenshtein minimum edit distance dynamic programming matrix $D[i,j]$ in $O(M \\cdot N)$ time."
            ],
            "follow_ups": [
                "Test live transcripts in the WER Evaluator tab",
                "What is the difference between WER and Character Error Rate (CER)?",
                "How do homophones affect speech recognition accuracy?"
            ]
        },
        {
            "tag": "knowledge_auto_upgrade",
            "title": "Continuous Knowledge Upgradation & Auto-Training",
            "patterns": [
                "how do you upgrade your knowledge", "automatic knowledge update", "web scraping training",
                "how does the auto train pipeline work", "continuous learning in chatbot",
                "where do you get your knowledge", "scrape new computer science topics",
                "cómo actualizas tu conocimiento", "entrenamiento automático con web scraping",
                "你是如何自动更新知识库的", "网络爬虫与机器学习自动训练",
                "आप अपने ज्ञान को कैसे अपडेट करते हैं", "वेब स्क्रैपिंग और एमएल ऑटो ट्रेनिंग"
            ],
            "responses": [
                "**Continuous Knowledge Upgradation Pipeline**:\n\n1. **Automated Web Harvesting (`knowledge_harvester.py`)**:\n   - Programmatically scrapes verified Computer Science & Mathematical Computing topics from Wikipedia REST API and ArXiv preprints (`cs.AI`, `cs.LG`, `cs.DS`, `math.NA`, `math.OC`).\n   - Synthesizes clean summaries, technical definitions, and multilingual training queries across 30+ certified world languages.\n   - Ingests new articles into `training/dynamic_knowledge_pool.json`.\n\n2. **ML/DL Model Retraining (`auto_train_pipeline.py`)**:\n   - Merges certified core intents with dynamic scraped articles.\n   - Re-extracts Sublinear TF-IDF features across word 1-2 n-grams, per-word character 3-4 n-grams, and CJK character tokens.\n   - Retrains the Softmax classifier with Stratified 5-Fold Cross Validation.\n   - Exports portable JSON model weights (`src/model/chatbot_model.json`).\n\n3. **Zero-Downtime Hot Reload**:\n   - Web server endpoints (`POST /api/knowledge/upgrade`) reload model weights in-memory instantly without restarting the server!"
            ],
            "follow_ups": [
                "Trigger an auto-upgrade in the Knowledge Base tab",
                "Scrape a custom Computer Science topic",
                "View model training statistics"
            ]
        },
        {
            "tag": "physics_astronomy_earth",
            "title": "Physics, Astronomy & Earth Science",
            "patterns": [
                "what is quantum mechanics", "explain general relativity", "newtons laws of motion",
                "what are black holes", "how does nuclear fusion work", "laws of thermodynamics entropy",
                "quantum entanglement physics", "electromagnetic waves optics light", "big bang theory cosmology",
                "plate tectonics earthquakes volcanoes", "climate science greenhouse effect", "higgs boson particle physics",
                "física cuántica y relatividad", "física y astronomía", "aguieros negros universo",
                "la physique quantique et relativité", "qué es la termodinámica", "quantenmechanik erklärt",
                "量子力学和广义相对论", "黑洞与天体物理学", "क्वांटम भौतिकी और खगोल विज्ञान क्या है",
                "न्यूटन के गति के नियम", "квантовая механика и астрономия"
            ],
            "responses": [
                "**Physics, Astronomy & Earth Sciences** investigate matter, energy, spacetime, and cosmic evolution — ask for a brief definition or detailed laws on any topic such as **Quantum Mechanics**, **Relativity**, **Thermodynamics**, **Optics**, **Cosmology**, or **Plate Tectonics**."
            ],
            "follow_ups": [
                "Explain Quantum Entanglement in detail",
                "Summarize Newton's Laws of Motion in brief",
                "How do black holes form?"
            ]
        },
        {
            "tag": "chemistry_biology_medicine",
            "title": "Chemistry, Biology, Genetics & Medicine",
            "patterns": [
                "how does photosynthesis work", "explain dna replication transcription", "crispr gene editing",
                "human immune system antibodies", "periodic table elements chemical bonds", "organic chemistry hydrocarbons",
                "cellular respiration atp", "evolution natural selection darwin", "human brain neuroscience neurons",
                "vaccines immunology antibiotics", "ecology food web biodiversity", "enzymes proteins amino acids",
                "cómo funciona la fotosíntesis", "biología molecular y adn", "química orgánica tabla periódica",
                "qu'est-ce que l'adn et la génétique", "biologie et médecine", "chemie und biologie",
                "光合作用与细胞呼吸", "基因编辑与分子生物学", "डीएनए और प्रकाश संश्लेषण क्या है",
                "मानव प्रतिरक्षा प्रणाली कैसे काम करती है", "биология и генетика человека"
            ],
            "responses": [
                "**Chemistry, Biology & Medicine** cover atomic bonding, molecular pathways, genetics, neuroscience, ecology, and healthcare. Specify any molecule, organ system, disease mechanism, or biological process for brief facts or an in-depth multi-source synthesis."
            ],
            "follow_ups": [
                "Explain CRISPR-Cas9 gene editing in detail",
                "What is photosynthesis in brief?",
                "How does mRNA vaccination work?"
            ]
        },
        {
            "tag": "engineering_robotics_energy",
            "title": "Engineering, Robotics, Aerospace & Energy",
            "patterns": [
                "mechanical engineering thermodynamics stress", "civil engineering bridges structural load",
                "electrical engineering circuits transistors", "aerospace engineering aerodynamics lift rocket",
                "robotics kinematics pid control", "renewable energy solar photovoltaic wind turbine",
                "internal combustion engine vs electric motor", "brushless dc motor inverter powertrain",
                "control systems stability feedback loop", "jet engine turbine propulsion nozzle",
                "hydraulic pneumatic actuator sensor fusion", "lithium ion battery chemistry ev",
                "semiconductor fabrication lithography", "chemical engineering distillation",
                "power grid transformer substation transmission", "finite element analysis cad structural",
                "ingeniería mecánica y robótica", "energías renovables y aeroespacial",
                "génie électrique et robotique", "maschinenbau und elektrotechnik",
                "机械工程与航空航天", "机器人控制与新能源技术", "रोबोटिक्स और एयरोस्पेस इंजीनियरिंग क्या है"
            ],
            "responses": [
                "**Engineering & Applied Technology** span **Mechanical, Civil, Electrical, Aerospace, Chemical, Robotics**, and **Clean Energy Systems** — ask about fluid mechanics, structural girders, rocket propulsion, PID control, or battery storage."
            ],
            "follow_ups": [
                "How does an airplane wing generate lift?",
                "Compare solid-state batteries and lithium-ion batteries",
                "Explain PID control in robotics"
            ]
        },
        {
            "tag": "commerce_finance_economics",
            "title": "Commerce, Finance, Economics & Business",
            "patterns": [
                "what is inflation and monetary policy", "supply and demand economics", "how does the stock market work",
                "compound interest and time value of money", "balance sheet income statement cash flow",
                "macroeconomics gdp fiscal policy", "microeconomics elasticity equilibrium", "marketing strategy 4ps",
                "supply chain management logistics", "venture capital startup valuation", "central bank interest rates",
                "accounting debit credit double entry", "international trade tariffs forex",
                "economía finanzas y mercado de valores", "contabilidad y estrategia empresarial",
                "économie finance et bourse", "wirtschaft und finanzen aktienmarkt",
                "经济学与金融市场", "股票市场与宏观经济", "शेयर बाजार और मुद्रास्फीति क्या है",
                "वित्त और अर्थशास्त्र के मूल सिद्धांत", "экономика и фондовый рынок"
            ],
            "responses": [
                "**Commerce, Economics & Finance** cover **Macro/Microeconomics, Equity & Bond Markets, Banking, Corporate Accounting, Valuation, Marketing**, and **Supply Chains**. Ask for formulas, market mechanics, financial statements, or economic history in brief or full detail."
            ],
            "follow_ups": [
                "Explain Quantitative Easing vs Quantitative Tightening in detail",
                "What is EBITDA in brief?",
                "Compare Microeconomics and Macroeconomics"
            ]
        },
        {
            "tag": "arts_literature_design",
            "title": "Fine Arts, Literature, Architecture & Design",
            "patterns": [
                "renaissance art history leonardo michelangelo", "impressionism and modern painting",
                "gothic architecture flying buttress", "classic world literature shakespeare homer",
                "photography exposure triangle aperture shutter", "graphic design typography color theory",
                "what is surrealism and cubism", "poetry sonnet haiku literary devices",
                "sculpture carving bronze marble", "industrial design bauhaus modernism",
                "historia del arte y literatura", "renacimiento arquitectura diseño",
                "histoire de l'art et littérature", "kunstgeschichte und architektur",
                "艺术史与世界文学", "文艺复兴与建筑设计", "कला इतिहास और विश्व साहित्य क्या है"
            ],
            "responses": [
                "**Arts, Literature & Design** embrace painting movements, architecture, classical & contemporary literature, poetry, sculpture, photography, and design principles. Ask about artists, movements, techniques, or literary works."
            ],
            "follow_ups": [
                "Explain the Renaissance art movement in detail",
                "What is the exposure triangle in photography in brief?",
                "Compare Impressionism and Expressionism"
            ]
        },
        {
            "tag": "music_audio_theory",
            "title": "Music Theory, Genres & Instruments",
            "patterns": [
                "music theory scales chords harmony", "circle of fifths in music",
                "indian classical music hindustani carnatic raga tala", "western classical composers mozart beethoven bach",
                "history of jazz improvisation", "rock music hip hop electronic production",
                "how does a guitar or piano produce sound", "violin tabla sitar flute musical instruments",
                "tempo rhythm time signature melody", "synthesizers daw music mixing mastering",
                "teoría musical escalas acordes", "música clásica jazz y rock",
                "théorie musicale et instruments", "musiktheorie und klassische musik",
                "音乐理论与古典音乐", "爵士乐与乐器演奏", "संगीत सिद्धांत राग और ताल क्या है",
                "शास्त्रीय संगीत और गिटार कॉर्ड्स कैसे काम करते हैं"
            ],
            "responses": [
                "**Music Theory, Genres & Instruments** span **Harmony, Scales, Ragas & Talas, Classical, Jazz, Rock, Hip-Hop, EDM**, and acoustic/electronic instruments — ask about scales, composers, genres, instruments, or production techniques."
            ],
            "follow_ups": [
                "Explain Hindustani vs Carnatic classical music",
                "What is the Circle of Fifths in brief?",
                "How did Jazz evolve in the 20th century?"
            ]
        },
        {
            "tag": "dance_performing_arts",
            "title": "Dance Forms, Choreography & Performing Arts",
            "patterns": [
                "what is bharatanatyam classical dance", "kathak kuchipudi odissi indian dance",
                "history of ballet dance technique", "hip hop dance breaking popping locking",
                "contemporary and modern dance choreography", "flamenco salsa tango ballroom dance",
                "opera musical theatre stage craft", "folk dances around the world",
                "danza clásica ballet y flamenco", "bailes tradicionales y coreografía",
                "danse classique ballet et hip hop", "tanz und ballett geschichte",
                "舞蹈艺术与芭蕾舞", "印度古典舞与现代舞", "भरतनाट्यम और कथक नृत्य क्या है",
                "बैले और हिप हॉप डांस के प्रकार"
            ],
            "responses": [
                "**Dance & Performing Arts** encompass Indian classical traditions (**Bharatanatyam, Kathak, Odissi, Kathakali**), **Ballet, Contemporary, Hip-Hop, Salsa, Tango, Flamenco**, choreography, and theatrical arts. Ask about origins, costumes, mudras, footwork, or iconic styles."
            ],
            "follow_ups": [
                "Explain Bharatanatyam origin, mudras, and costume in detail",
                "Compare Kathak and Ballet",
                "What are the core elements of Hip-Hop dance?"
            ]
        },
        {
            "tag": "sports_athletics_games",
            "title": "Sports, Athletics, Olympics & Mind Games",
            "patterns": [
                "cricket rules lbw duckworth lewis icc world cup", "football soccer rules offside fifa world cup",
                "basketball nba rules three point shot", "tennis grand slam wimbledon scoring",
                "badminton olympics shuttlecock rules", "chess rules castling en passant grandmaster",
                "formula 1 f1 aerodynamics pit stop championship", "olympic games track and field marathon",
                "swimming freestyle butterfly stroke", "martial arts judo karate boxing mma",
                "reglas del fútbol y críquet", "baloncesto tenis y olimpiadas",
                "règles du football et tennis", "fußball regeln und olympische spiele",
                "足球规则与板球比赛", "篮球网球与奥林匹克运动会", "क्रिकेट और फुटबॉल के नियम क्या हैं",
                "शतरंज ओलंपिक और फॉर्मूला वन रेसिंग"
            ],
            "responses": [
                "**Sports & Athletics** cover rules, scoring systems, tactics, tournaments, leagues, and legends across **Cricket, Football (Soccer), Basketball, Tennis, Badminton, Chess, Formula 1, Swimming, Martial Arts**, and **The Olympic Games**."
            ],
            "follow_ups": [
                "Explain the LBW and Duckworth-Lewis rules in Cricket in detail",
                "What is the offside rule in Football in brief?",
                "How does Formula 1 scoring work?"
            ]
        },
        {
            "tag": "entertainment_cinema_gaming",
            "title": "Cinema, Animation, Television & Video Games",
            "patterns": [
                "history of cinema filmmaking cinematography", "academy awards oscars grammy emmy",
                "anime and manga japan studio ghibli", "3d animation pixar cgi visual effects vfx",
                "video game design unreal engine unity", "esports competitive gaming tournaments",
                "television streaming ott platforms screenwriting", "bollywood hollywood world cinema",
                "historia del cine y animación", "videojuegos y deportes electrónicos esports",
                "cinéma animation et jeux vidéo", "filmgeschichte und videospiele",
                "电影艺术与动画制作", "电子游戏与电子竞技", "सिनेमा फिल्म निर्माण और एनीमेशन क्या है",
                "वीडियो गेम और ईस्पोर्ट्स टूर्नामेंट"
            ],
            "responses": [
                "**Entertainment, Cinema & Gaming** cover film history, cinematography, VFX, animation & anime, screenwriting, major awards (Oscars/Grammys), streaming media, game engines, and esports."
            ],
            "follow_ups": [
                "How are 3D animated movies rendered at Pixar?",
                "Explain Studio Ghibli's impact on animation in brief",
                "Compare Hollywood and Indian Cinema traditions"
            ]
        },
        {
            "tag": "history_geography_law_society",
            "title": "World History, Geography, Philosophy, Law & Culture",
            "patterns": [
                "industrial revolution causes and impact", "ancient egypt mesopotamia indus valley",
                "world war 1 and world war 2 history", "continents oceans highest mountains rivers",
                "united nations international law human rights", "philosophy stoicism existentialism ethics",
                "constitutional democracy separation of powers", "magna carta treaty parliamentary law",
                "ancient greece roman empire silk road", "french revolution enlightenment colonialism",
                "culinary arts fermentation world cuisine",
                "geografía historia universal y filosofía", "derecho internacional y civilizaciones",
                "histoire géographie et philosophie", "weltgeschichte geographie und recht",
                "世界历史与地理文化", "哲学伦理与国际法", "विश्व इतिहास भूगोल और दर्शनशास्त्र क्या है",
                "संविधान कानून और प्राचीन सभ्यताएं"
            ],
            "responses": [
                "**World History, Geography, Law & Society** include ancient civilizations, modern history, physical & human geography, philosophy, constitutional governance, culinary traditions, and global culture."
            ],
            "follow_ups": [
                "Explain the causes of the Industrial Revolution in detail",
                "What is Stoicism philosophy in brief?",
                "Compare the legislative and judicial branches"
            ]
        },
        {
            "tag": "thanks_goodbye",
            "title": "Thanks & Farewell",
            "patterns": [
                "thank you", "thanks a lot", "bye", "goodbye", "see you later",
                "gracias", "muchas gracias", "adiós", "hasta luego",
                "merci", "merci beaucoup", "au revoir", "à bientôt",
                "danke", "vielen dank", "auf wiedersehen", "tschüss",
                "спасибо", "большое спасибо", "до свидания", "пока",
                "شكرا", "شكرا جزيلا", "مع السلامة", "إلى اللقاء",
                "谢谢", "多谢", "再见", "拜拜",
                "धन्यवाद", "बहुत बहुत धन्यवाद", "अलविदा", "फिर मिलेंगे",
                "obrigado", "muito obrigado", "tchau", "até logo",
                "arigato", "sayonara", "kamsa hamnida", "annyong"
            ],
            "responses": [
                "You are very welcome! If you have more questions about Computer Science, Mathematical Computing, or Speech AI, feel free to speak or type in any certified language.",
                "Glad I could help! I am always ready to analyze algorithms, solve mathematical computing questions, or test speech audio in the WER Lab."
            ],
            "follow_ups": [
                "What is Automatic Speech Recognition?",
                "Explain Singular Value Decomposition (SVD)",
                "How does OpenAI Whisper work?"
            ]
        }
    ]

    dataset = {
        "metadata": {
            "domain": "Omni-Domain Multilingual AI Web Browser & Knowledge Assistant (35 Certified Languages, 28 World Disciplines)",
            "version": "4.0.0",
            "total_intents": len(intents),
            "certified_languages_count": len(CERTIFIED_LANGUAGES)
        },
        "intents": intents
    }

    INTENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INTENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Master intents generated: {len(intents)} intents saved to {INTENTS_PATH}")
    return dataset

if __name__ == "__main__":
    build_intents()
