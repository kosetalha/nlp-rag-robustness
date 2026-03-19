# In RAG We Trust? Measuring Vulnerability to Document Poisoning

Project: In RAG We Trust? Measuring Vulnerability to Document Poisoning  
Author: Talha Kose  
Professor: Alfio Ferrara  
Course: NLP 2025-26 | Universita degli Studi di Milano

---

## Overview

This project measures how vulnerable a RAG pipeline is to semantic document poisoning.

Core idea:
1. Build a clean benchmark from HotpotQA contexts.
2. Build a poisoned benchmark by replacing true entities with plausible distractors.
3. Run the same questions in clean and poisoned conditions.
4. Compare quality and attack metrics.

Main finding:
Increasing retrieval depth and context size did not improve robustness in this setup.

---

## Results Snapshot

| Metric | V1 Clean | V1 Poisoned | V2 Clean | V2 Poisoned |
|--------|----------|-------------|----------|-------------|
| Exact Match | 34.5% | 7.5% | 38.5% | 7.0% |
| Attack Success Rate | 0% | 25.0% | 0% | 28.0% |
| Relative EM Drop | - | -78% | - | -82% |

---

## Repository Map

```
nlp-rag-robustness/
|-- phase1/
|   |-- Phase1_Baseline_Study.ipynb
|   |-- data/
|   |   `-- processed/
|   |       `-- benchmark_semantic.json
|   `-- results/
|       |-- baseline_results_v1.json
|       |-- baseline_results_v2.json
|       |-- evaluation_v1.json
|       `-- evaluation_v2.json
|-- src/
|   |-- config.py
|   |-- data_loader.py
|   |-- retrieval.py
|   |-- generator.py
|   |-- pipeline.py
|   `-- evaluation.py
|-- requirements.txt
`-- README.md
```

---

## What Each File Does

### Orchestration
- [phase1/Phase1_Baseline_Study.ipynb](phase1/Phase1_Baseline_Study.ipynb)  
Notebook controller for setup, benchmark generation, experiment execution, plotting, and report export.

### Core Modules
- [src/config.py](src/config.py)  
All paths, model IDs, and experiment defaults.

- [src/data_loader.py](src/data_loader.py)  
Loads HotpotQA and creates semantic poisoned samples.

- [src/retrieval.py](src/retrieval.py)  
ChromaDB wrapper for document indexing and question-based retrieval.

- [src/generator.py](src/generator.py)  
Llama-3 loading (4-bit) and answer generation with consistency-aware prompt rules.

- [src/pipeline.py](src/pipeline.py)  
End-to-end run logic for clean/poisoned conditions, retrieval mode, and optional verification mode.

- [src/evaluation.py](src/evaluation.py)  
Metric computation and reporting (EM, F1, ASR, refusal, plus verification metrics when enabled).

---

## How To Orchestrate (Recommended)

Use [phase1/Phase1_Baseline_Study.ipynb](phase1/Phase1_Baseline_Study.ipynb) as the single entry point.

Notebook flow:
1. Setup Colab environment and install dependencies.
2. Load HotpotQA and inject semantic poison.
3. Filter to successfully poisoned samples and save benchmark.
4. Run V1 and V2 with retrieval mode enabled.
5. Evaluate and export tables/charts/reports.

Why this structure:
- Easy to reproduce in one run.
- Clear separation between orchestration (notebook) and logic (src modules).
- Consistent with Colab-only GPU availability.

---

## Quick Start

### Option A: Google Colab (recommended)
1. Open [phase1/Phase1_Baseline_Study.ipynb](phase1/Phase1_Baseline_Study.ipynb).
2. Add HF_TOKEN to Colab secrets.
3. Run all cells in order.

### Option B: Local run (GPU required)

```bash
git clone https://github.com/kosetalha/nlp-rag-robustness.git
cd nlp-rag-robustness

python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# set HF_TOKEN in your environment or .env
jupyter notebook phase1/Phase1_Baseline_Study.ipynb
```

---

## Experiment Modes in Pipeline

In [src/pipeline.py](src/pipeline.py), run_experiment supports:

- condition: clean or poisoned
- use_retrieval: enables question-driven Chroma retrieval
- enable_verification: enables deterministic multi-source conflict handling

Minimal example:

```python
from src.pipeline import RAGPipeline

pipe = RAGPipeline(top_k=3, max_context_chars=6000)
pipe.load_benchmark("benchmark_semantic.json")

clean = pipe.run_experiment(
	condition="clean",
	use_retrieval=True,
	enable_verification=False,
)

poisoned_verified = pipe.run_experiment(
	condition="poisoned",
	use_retrieval=True,
	enable_verification=True,
)

pipe.save_results("baseline_results_with_verification.json")
```

---

## Metrics

Computed in [src/evaluation.py](src/evaluation.py):

- Exact Match
- F1 Score
- Contains Answer
- Refusal Rate
- Attack Success Rate (poisoned condition)

When verification is enabled, additional metrics include conflict and override rates.

---

## Notes for Reproducibility

1. Keep RANDOM_SEED fixed.
2. Keep benchmark file under phase1/data/processed.
3. Run notebook cells in order.
4. Save all outputs under phase1/results.

---

## Next Step

Phase 2 extends this baseline with stronger verification and reliability mechanisms for thesis work.

