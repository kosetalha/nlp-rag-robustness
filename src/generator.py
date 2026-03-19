"""
LLM Generator module using Llama-3.
Handles model loading and answer generation.
"""
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from typing import Optional
from src.config import MODEL_NAME, HF_TOKEN, MAX_NEW_TOKENS, MAX_CONTEXT_CHARS, TEMPERATURE, TOP_P


class LlamaGenerator:
    """
    Llama-3 based text generator for RAG.
    Uses 4-bit quantization for efficient GPU usage.
    """
    
    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialize Llama-3 with quantization.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipe = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the model with 4-bit quantization."""
        print(f"🦙 Loading {self.model_name}...")
        
        # Check GPU availability
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("   ⚠️ No GPU detected - will be slow!")
        
        try:
            # Tokenizer must match the instruction-tuned model family.
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=HF_TOKEN
            )
            
            # 4-bit quantization keeps VRAM usage low enough for Colab/T4 while
            # preserving adequate quality for factoid extraction.
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # device_map="auto" lets transformers place layers on available devices.
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=HF_TOKEN,
                device_map="auto",
                quantization_config=bnb_config,
                torch_dtype=torch.float16
            )
            
            # Pipeline API provides a concise generation interface used by pipeline.py.
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=TOP_P
            )
            
            print("✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("\nTroubleshooting:")
            print("  1. Check HF_TOKEN is set in .env file")
            print("  2. Accept Llama-3 license at: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
            print("  3. Ensure enough GPU memory (8GB+ recommended)")
            raise
    
    def _truncate_context(self, context: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """Truncate context to prevent OOM errors."""
        # Character-based clipping is a pragmatic safety guard for long contexts.
        if len(context) > max_chars:
            return context[:max_chars] + "\n[Context truncated for length...]"
        return context
    
    def _clear_gpu_memory(self) -> None:
        """Clear GPU cache to prevent OOM errors."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate an answer based on question and context.
        
        Args:
            question: User question
            context: Retrieved context documents
            
        Returns:
            Generated answer string
        """
        # Truncate context to prevent OOM
        context = self._truncate_context(context)
        
        # Single consistency-aware prompt (Phase 2): this directly encodes the
        # requirement that answers should be corroborated and conflicts refused.
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a reliability-focused question answering assistant.
    Use only the provided context.

    Rules:
    1) Compare evidence across the provided sources (titles/segments).
    2) Prefer answers corroborated by multiple sources.
    3) If sources conflict OR evidence is insufficient, output exactly: UNANSWERABLE
    4) Output ONLY the final answer entity (1-4 words). No sentences, no explanation.

Examples:
Q: What is the capital of France? A: Paris
Q: Who wrote Hamlet? A: William Shakespeare
Q: When was Einstein born? A: 1879
Q: What nationality was Chopin? A: Polish
    Q: If two sources disagree on a date? A: UNANSWERABLE

If not in context: UNANSWERABLE<|eot_id|><|start_header_id|>user<|end_header_id|>

Context: {context}

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
A:
"""
        
        # no_grad avoids autograd overhead during inference.
        with torch.no_grad():
            outputs = self.pipe(
                prompt,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=True
            )
        
        # The pipeline returns prompt + completion, so split on assistant header.
        generated_text = outputs[0]["generated_text"]
        answer = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        # Remove trailing special tokens when present.
        if "<|eot_id|>" in answer:
            answer = answer.split("<|eot_id|>")[0].strip()
        
        # Keep output as raw factoid entity for evaluation metrics.
        if answer.startswith("A:"):
            answer = answer[2:].strip()
        
        # Strip punctuation artifacts that can reduce EM/F1 unfairly.
        answer = answer.strip('"\'.')
        
        return answer
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.pipe is not None
