# In RAG We Trust? Measuring Vulnerability to Document Poisoning

**Project:** In RAG We Trust? Measuring Vulnerability to Document Poisoning  
**Author:** Talha Köse   
**Professor:** Alfio Ferrara   
**Course:** NLP 2025-26 | Università degli Studi di Milano

---

## 🎯 Problem

Retrieval-Augmented Generation (RAG) systems power modern AI applications (ChatGPT with browsing, Microsoft Copilot, Perplexity AI), but they operate under a critical assumption: **retrieved documents are trustworthy**. This project asks: *What happens when adversaries inject false information into the knowledge base?*

## 🔬 What We Found

RAG systems are **severely vulnerable** to document poisoning attacks:

| Metric | V1 Clean | V1 Poisoned | V2 Clean | V2 Poisoned |
|--------|----------|-------------|----------|-------------|
| **Exact Match** | 34.5% | 7.5% | 38.5% | 7.0% |
| **Attack Success Rate** | 0% | 25.0% | 0% | 28.0% |
| **EM Drop (Relative)** | — | **-78%** | — | **-82%** |

**Counterintuitive Finding:** Improving retrieval (more documents, larger context) *increases* attack success rate from 25% → 28%. Better retrieval provides more attack surface, not more robustness.

## 🧪 Methodology

### Attack: Semantic Document Poisoning
We replace correct answers in retrieved documents with plausible distractors:
- **Example 1:** "Born in **1959**" → "Born in **1961**" (same entity type: DATE)
- **Example 2:** "Located in **Oldham County**" → "Located in **New York**" (same type: GPE)

### RAG Pipeline
1. **Retrieval:** ChromaDB with `all-MiniLM-L6-v2` embeddings
2. **Generation:** Llama-3-8B-Instruct (4-bit quantized) with factoid extraction prompt
3. **Dataset:** 200 HotpotQA multi-hop questions (148 successfully poisoned)

### Experimental Conditions
- **V1 (Baseline):** Retrieve top-2 docs, 4K char context
- **V2 (Enhanced):** Retrieve top-3 docs, 6K char context
- **A/B Test:** Each question tested under clean vs. poisoned conditions

## 📁 Repository Structure

```
nlp-rag-robustness/
├── phase1/
│   ├── Phase1_Baseline_Study.ipynb    # Main experiment notebook
│   ├── data/processed/                # Pre-generated poisoned benchmarks
│   │   ├── benchmark_semantic.json
│   │   └── benchmark_syntactic.json
│   └── results/                       # Experimental results (V1 & V2)
│       ├── baseline_results_v1.json
│       ├── baseline_results_v2.json
│       └── evaluation_v1.json
│
├── src/                               # Core RAG implementation
│   ├── config.py                      # Configuration & paths
│   ├── data_loader.py                 # HotpotQA + poisoning logic
│   ├── retrieval.py                   # ChromaDB wrapper
│   ├── generator.py                   # Llama-3 wrapper
│   ├── pipeline.py                    # RAG orchestration
│   └── evaluation.py                  # Metrics (EM, F1, ASR)
│
└── requirements.txt
```

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
1. Open [`phase1/Phase1_Baseline_Study.ipynb`](phase1/Phase1_Baseline_Study.ipynb)
2. Click "Open in Colab" badge
3. Add your HuggingFace token to Colab Secrets (`HF_TOKEN`)
4. Run all cells (takes ~30 min on T4 GPU)

### Option 2: Local Setup
**Requirements:** Python 3.10+, CUDA GPU (8GB+ VRAM), HuggingFace account with Llama-3 access

```bash
git clone https://github.com/kosetalha/nlp-rag-robustness.git
cd nlp-rag-robustness

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Configure HuggingFace token
cp .env.example .env
# Edit .env and add: HF_TOKEN=your_token_here

# Run notebook
jupyter notebook phase1/Phase1_Baseline_Study.ipynb
```

## 📊 Evaluation Metrics

- **Exact Match (EM):** Strict answer correctness (normalized)
- **F1 Score:** Token-level overlap between prediction and ground truth
- **Attack Success Rate (ASR):** Model outputs planted distractor (not true answer)
- **Refusal Rate:** Model responds "UNANSWERABLE"

## 🔮 Phase 2: Defense Mechanisms

Planned thesis extension implementing **Dialectical Verification**:
- Multi-agent debate system (Generator + Auditor agents)
- Cross-document consistency checking
- Confidence calibration before final output

