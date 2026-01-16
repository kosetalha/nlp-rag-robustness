"""
End-to-end RAG pipeline for baseline experiments.
Combines retrieval and generation with experiment tracking.
"""
import json
import gc
import torch
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm

from src.config import PROCESSED_DATA_DIR, RESULTS_DIR, TOP_K_RETRIEVAL, MAX_CONTEXT_CHARS
from src.retrieval import RAGRetriever
from src.generator import LlamaGenerator


class RAGPipeline:
    """
    Complete RAG pipeline for running baseline experiments.
    Supports configurable parameters for sensitivity analysis (V1/V2).
    """
    
    def __init__(self, lazy_load: bool = False, top_k: int = TOP_K_RETRIEVAL, 
                 max_context_chars: int = MAX_CONTEXT_CHARS):
        """
        Initialize the RAG pipeline.
        
        Args:
            lazy_load: If True, don't load the LLM until needed (saves memory)
            top_k: Number of documents to retrieve (default from config)
            max_context_chars: Max context length in chars (default from config)
        """
        self.retriever = None
        self.generator = None
        self.data = []
        self.results = []
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        
        if not lazy_load:
            self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize retriever and generator."""
        print("\n" + "="*60)
        print("INITIALIZING RAG PIPELINE")
        print(f"   TOP_K: {self.top_k}, MAX_CONTEXT: {self.max_context_chars}")
        print("="*60)
        
        self.retriever = RAGRetriever(reset_db=True)
        self.generator = LlamaGenerator()
        
        print("="*60 + "\n")
    
    def _clear_gpu_memory(self) -> None:
        """Clear GPU cache to prevent OOM errors."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def load_benchmark(self, filename: str) -> None:
        """
        Load a benchmark file.
        
        Args:
            filename: Benchmark JSON filename
        """
        path = PROCESSED_DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Benchmark not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        print(f"📂 Loaded {len(self.data)} samples from {filename}")
    
    def run_experiment(
        self, 
        condition: str = "clean",
        limit: Optional[int] = None,
        verbose: bool = True,
        use_retrieval: bool = False
    ) -> List[Dict]:
        """
        Run the RAG experiment.
        
        Args:
            condition: 'clean' or 'poisoned' - which context to use
            limit: Max samples to process (None = all)
            verbose: Print detailed output
            use_retrieval: If True, use cross-sample retrieval (original behavior).
                          If False (default), use direct per-sample context (recommended).
            
        Returns:
            List of result dictionaries
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_benchmark() first!")
        
        if self.generator is None:
            self._initialize_components()
        
        # Select subset
        samples = self.data[:limit] if limit else self.data
        
        mode = "RETRIEVAL" if use_retrieval else "DIRECT CONTEXT"
        print(f"\n🚀 Running experiment: condition={condition.upper()}, samples={len(samples)}, mode={mode}")
        print("-" * 60)
        
        # Only index if using retrieval mode
        if use_retrieval:
            docs = []
            ids = []
            metadatas = []
            
            for item in samples:
                context = item['poisoned_context'] if condition == "poisoned" else item['clean_context']
                docs.append(context)
                ids.append(item['id'])
                metadatas.append({
                    "real_answer": str(item['real_answer']),
                    "distractor": str(item.get('distractor_answer', ''))
                })
            
            # Reset and index
            self.retriever = RAGRetriever(reset_db=True)
            self.retriever.index_documents(docs, ids, metadatas)
        
        # Run queries
        results = []
        
        iterator = tqdm(samples, desc="Generating answers") if not verbose else samples
        
        for i, item in enumerate(iterator):
            # Get context - either via retrieval or directly from sample
            if use_retrieval:
                # Original behavior: retrieve from indexed documents (may get cross-sample context)
                retrieved_docs = self.retriever.retrieve(item['question'], top_k=self.top_k)
                combined_context = "\n\n---\n\n".join(retrieved_docs)
            else:
                # Direct context: use this sample's own context (clean or poisoned)
                combined_context = item['poisoned_context'] if condition == "poisoned" else item['clean_context']
                # Truncate if needed
                if len(combined_context) > self.max_context_chars:
                    combined_context = combined_context[:self.max_context_chars] + "\n[Context truncated...]"
            
            # Generate answer
            generated_answer = self.generator.generate_answer(item['question'], combined_context)
            
            result = {
                "id": item['id'],
                "question": item['question'],
                "real_answer": item['real_answer'],
                "distractor_answer": item.get('distractor_answer', ''),
                "generated_answer": generated_answer,
                "condition": condition,
                "poison_strategy": item.get('poison_strategy', 'none'),
                "top_k": self.top_k,
                "max_context_chars": self.max_context_chars,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            
            if verbose:
                print(f"\n📌 Q: {item['question']}")
                print(f"   Generated: {generated_answer[:100]}...")
                print(f"   Real: {item['real_answer']}")
                if condition == "poisoned":
                    print(f"   Poison: {item.get('distractor_answer', 'N/A')}")
            
            # Clear GPU memory every 10 samples to prevent OOM
            if (i + 1) % 10 == 0:
                self._clear_gpu_memory()
        
        # Final cleanup
        self._clear_gpu_memory()
        
        self.results.extend(results)
        print(f"\n✅ Completed {len(results)} queries")
        
        return results
    
    def save_results(self, filename: str = "baseline_results.json") -> None:
        """Save experiment results to disk (overwrites existing file)."""
        if not self.results:
            print("⚠️ No results to save!")
            return
        
        path = RESULTS_DIR / filename
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(self.results)} results to: {path}")
    
    def clear_results(self) -> None:
        """Clear in-memory results."""
        self.results = []
        print("🗑️ Results cleared")
