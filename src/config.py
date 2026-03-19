"""
Configuration settings for the RAG Robustness project.
All paths, constants, and hyperparameters are defined here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============== PATHS ==============
# Centralized path config keeps notebook/script code clean and reproducible across
# local and Colab environments.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE1_DIR = PROJECT_ROOT / "phase1"
DATA_DIR = PHASE1_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PHASE1_DIR / "results"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============== API KEYS ==============
# HF token is read once here so all modules use the same auth source.
HF_TOKEN = os.getenv("HF_TOKEN")

# ============== MODEL SETTINGS ==============
# Llama-3-8B is the generation model; MiniLM provides fast embedding search.
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============== EXPERIMENT SETTINGS ==============
# Defaults are V2-oriented; V1 can override these at pipeline construction time.
SAMPLE_SIZE = 200  # Number of HotpotQA samples to use
TOP_K_RETRIEVAL = 3  # Number of documents to retrieve
RANDOM_SEED = 42

# ============== BENCHMARK FILENAMES ==============
# Semantic poisoning is the only active benchmark in this project.
BENCHMARK_SEMANTIC = "benchmark_semantic.json"
RESULTS_FILENAME = "baseline_results.json"

# ============== GENERATION SETTINGS ==============
# Low temperature keeps outputs stable for A/B comparisons.
MAX_NEW_TOKENS = 15  # Short factoid answers only (3-4 words max)
MAX_CONTEXT_CHARS = 6000  # Context truncation limit
TEMPERATURE = 0.1
TOP_P = 0.9
