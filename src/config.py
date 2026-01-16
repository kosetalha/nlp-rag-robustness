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
HF_TOKEN = os.getenv("HF_TOKEN")

# ============== MODEL SETTINGS ==============
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============== EXPERIMENT SETTINGS ==============
SAMPLE_SIZE = 200  # Number of HotpotQA samples to use
TOP_K_RETRIEVAL = 3  # Number of documents to retrieve
RANDOM_SEED = 42

# ============== BENCHMARK FILENAMES ==============
BENCHMARK_SYNTACTIC = "benchmark_syntactic.json"
BENCHMARK_SEMANTIC = "benchmark_semantic.json"
RESULTS_FILENAME = "baseline_results.json"

# ============== GENERATION SETTINGS ==============
MAX_NEW_TOKENS = 15  # Short factoid answers only (3-4 words max)
MAX_CONTEXT_CHARS = 6000  # Context truncation limit
TEMPERATURE = 0.1
TOP_P = 0.9
