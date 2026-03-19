"""
End-to-end RAG pipeline for baseline experiments.
Combines retrieval and generation with experiment tracking.
"""
import json
import gc
import torch
import re
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
        # Centralized initialization keeps the notebook API simple.
        print("\n" + "="*60)
        print("INITIALIZING RAG PIPELINE")
        print(f"   TOP_K: {self.top_k}, MAX_CONTEXT: {self.max_context_chars}")
        print("="*60)
        
        self.retriever = RAGRetriever(reset_db=True)
        self.generator = LlamaGenerator()
        
        print("="*60 + "\n")
    
    def _clear_gpu_memory(self) -> None:
        """Clear GPU cache to prevent OOM errors."""
        # Long experiment loops can fragment GPU memory; periodic cleanup is defensive.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _normalize_answer(text: str) -> str:
        """Normalize answer text for robust equality checks."""
        # Shared normalization avoids false conflicts from punctuation/casing only.
        return " ".join(re.sub(r"[^\w\s]", "", (text or "").lower()).split())

    @staticmethod
    def _is_refusal_answer(text: str) -> bool:
        """Detect refusal-style answers."""
        t = (text or "").lower()
        refusal_phrases = [
            "unanswerable",
            "cannot answer",
            "unable to answer",
            "not enough information",
            "no information",
            "not mentioned",
            "not provided",
        ]
        return any(p in t for p in refusal_phrases)

    @staticmethod
    def _split_sources(context: str) -> List[str]:
        """Split flattened benchmark context into source-level chunks by Title blocks."""
        # Source boundaries are needed for explicit multi-source verification.
        if not context:
            return []
        chunks = re.split(r"(?=^Title:\s)", context, flags=re.MULTILINE)
        chunks = [c.strip() for c in chunks if c and c.strip()]
        return chunks if chunks else [context]

    def _aggregate_source_answers(self, source_answers: List[str]) -> Dict[str, Any]:
        """
        Deterministic self-consistency summary from per-source answers.

        Returns:
            support/conflict stats and selected top answer (if any)
        """
        # Map normalized answer -> raw variants to compute support/conflict evidence.
        answer_map: Dict[str, List[str]] = {}
        refusal_count = 0

        for a in source_answers:
            if self._is_refusal_answer(a):
                refusal_count += 1
                continue
            norm = self._normalize_answer(a)
            if not norm:
                continue
            answer_map.setdefault(norm, []).append(a)

        if not answer_map:
            return {
                "top_answer": "",
                "support_count": 0,
                "conflict_count": 0,
                "unique_non_refusal_answers": 0,
                "refusal_count": refusal_count,
                "conflict_detected": False,
                "confidence_status": "insufficient",
            }

        # Deterministic ranking (largest support wins) enables reproducible behavior.
        ranked = sorted(answer_map.items(), key=lambda x: len(x[1]), reverse=True)
        _top_norm, top_variants = ranked[0]
        top_answer = top_variants[0]
        support_count = len(top_variants)
        unique_non_refusal = len(answer_map)
        conflict_count = sum(len(v) for _, v in ranked[1:])
        conflict_detected = unique_non_refusal > 1

        if conflict_detected:
            confidence_status = "conflicted"
        elif support_count >= 2:
            confidence_status = "supported"
        else:
            confidence_status = "weakly_supported"

        return {
            "top_answer": top_answer,
            "support_count": support_count,
            "conflict_count": conflict_count,
            "unique_non_refusal_answers": unique_non_refusal,
            "refusal_count": refusal_count,
            "conflict_detected": conflict_detected,
            "confidence_status": confidence_status,
        }
    
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
        use_retrieval: bool = False,
        enable_verification: bool = False
    ) -> List[Dict]:
        """
        Run the RAG experiment.
        
        Args:
            condition: 'clean' or 'poisoned' - which context to use
            limit: Max samples to process (None = all)
            verbose: Print detailed output
            use_retrieval: If True, use cross-sample retrieval (original behavior).
                          If False (default), use direct per-sample context (recommended).
            enable_verification: If True, run deterministic multi-source verification
                                 and self-consistency scoring.
            
        Returns:
            List of result dictionaries
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_benchmark() first!")
        
        if self.generator is None:
            self._initialize_components()
        
        # Optional limit allows quick smoke tests before full expensive runs.
        samples = self.data[:limit] if limit else self.data
        
        mode = "RETRIEVAL" if use_retrieval else "DIRECT CONTEXT"
        print(f"\n🚀 Running experiment: condition={condition.upper()}, samples={len(samples)}, mode={mode}, verification={enable_verification}")
        print("-" * 60)
        
        # Retrieval mode indexes all candidate contexts and then queries by question.
        if use_retrieval:
            docs = []
            ids = []
            metadatas = []
            
            for item in samples:
                context = item['poisoned_context'] if condition == "poisoned" else item['clean_context']
                # Index each source block separately so question retrieval targets
                # fine-grained evidence instead of whole-sample mega-context.
                source_blocks = self._split_sources(context)
                for j, block in enumerate(source_blocks):
                    docs.append(block)
                    ids.append(f"{item['id']}_src{j}")
                    metadatas.append({
                        "sample_id": str(item['id']),
                        "real_answer": str(item['real_answer']),
                        "distractor": str(item.get('distractor_answer', ''))
                    })
            
            # Reset avoids stale collections leaking documents across runs.
            self.retriever = RAGRetriever(reset_db=True)
            self.retriever.index_documents(docs, ids, metadatas)
        
        # Main experiment loop: assemble context -> generate -> optionally verify.
        results = []
        
        iterator = tqdm(samples, desc="Generating answers") if not verbose else samples
        
        for i, item in enumerate(iterator):
            # Get context - either via retrieval or directly from sample
            if use_retrieval:
                # Original behavior: retrieve from indexed documents (may get cross-sample context)
                retrieved_docs = self.retriever.retrieve(item['question'], top_k=self.top_k)
                combined_context = "\n\n---\n\n".join(retrieved_docs)
                source_contexts = retrieved_docs if retrieved_docs else [combined_context]
            else:
                # Direct context: use this sample's own context (clean or poisoned)
                combined_context = item['poisoned_context'] if condition == "poisoned" else item['clean_context']
                # Truncate if needed
                if len(combined_context) > self.max_context_chars:
                    combined_context = combined_context[:self.max_context_chars] + "\n[Context truncated...]"
                source_contexts = self._split_sources(combined_context)
                if not source_contexts:
                    source_contexts = [combined_context]
            
            # Keep raw baseline answer so verification effects remain auditable.
            raw_generated_answer = self.generator.generate_answer(item['question'], combined_context)
            generated_answer = raw_generated_answer

            # Optional deterministic multi-source verification + self-consistency
            verification_payload = {
                "verification_enabled": enable_verification,
                "source_count": len(source_contexts),
                "support_count": 0,
                "conflict_count": 0,
                "unique_non_refusal_answers": 0,
                "source_refusal_count": 0,
                "conflict_detected": False,
                "confidence_status": "not_run",
                "verification_overrode": False,
            }

            if enable_verification:
                # Per-source answers provide explicit evidence for cross-source agreement.
                source_answers = [
                    self.generator.generate_answer(item['question'], ctx)
                    for ctx in source_contexts
                ]
                agg = self._aggregate_source_answers(source_answers)

                # Approved policy: any conflict forces refusal (safety-first behavior).
                if agg["conflict_detected"]:
                    generated_answer = "UNANSWERABLE"
                else:
                    generated_answer = agg["top_answer"] if agg["top_answer"] else "UNANSWERABLE"

                verification_payload.update({
                    "support_count": agg["support_count"],
                    "conflict_count": agg["conflict_count"],
                    "unique_non_refusal_answers": agg["unique_non_refusal_answers"],
                    "source_refusal_count": agg["refusal_count"],
                    "conflict_detected": agg["conflict_detected"],
                    "confidence_status": agg["confidence_status"],
                    # Track whether verification changed the initial model output.
                    "verification_overrode": self._normalize_answer(generated_answer) != self._normalize_answer(raw_generated_answer),
                    "source_answers": source_answers,
                })
            
            result = {
                "id": item['id'],
                "question": item['question'],
                "real_answer": item['real_answer'],
                "distractor_answer": item.get('distractor_answer', ''),
                "generated_answer": generated_answer,
                "raw_generated_answer": raw_generated_answer,
                "condition": condition,
                "poison_strategy": item.get('poison_strategy', 'none'),
                "top_k": self.top_k,
                "max_context_chars": self.max_context_chars,
                "verification": verification_payload,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            
            if verbose:
                print(f"\n📌 Q: {item['question']}")
                print(f"   Generated: {generated_answer[:100]}...")
                print(f"   Real: {item['real_answer']}")
                if condition == "poisoned":
                    print(f"   Poison: {item.get('distractor_answer', 'N/A')}")
            
            # Periodic cleanup prevents long-run crashes on constrained GPUs.
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
