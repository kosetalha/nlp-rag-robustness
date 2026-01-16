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
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=HF_TOKEN
            )
            
            # Configure 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load model with quantization config
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=HF_TOKEN,
                device_map="auto",
                quantization_config=bnb_config,
                torch_dtype=torch.float16
            )
            
            # Create pipeline
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
        
        # Factoid extraction prompt - outputs only the answer entity (3-4 words max)
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Extract the answer from the context. Output ONLY the answer entity (1-4 words). No sentences.

Examples:
Q: What is the capital of France? A: Paris
Q: Who wrote Hamlet? A: William Shakespeare
Q: When was Einstein born? A: 1879
Q: What nationality was Chopin? A: Polish

If not in context: UNANSWERABLE<|eot_id|><|start_header_id|>user<|end_header_id|>

Context: {context}

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
A:
"""
        
        # Generate with memory cleanup
        with torch.no_grad():
            outputs = self.pipe(
                prompt,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=True
            )
        
        # Extract response (remove prompt)
        generated_text = outputs[0]["generated_text"]
        answer = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        # Clean up any remaining tokens
        if "<|eot_id|>" in answer:
            answer = answer.split("<|eot_id|>")[0].strip()
        
        # Remove prompt artifact "A:" or "A:\n"
        if answer.startswith("A:"):
            answer = answer[2:].strip()
        
        # Clean up leading/trailing quotes or periods
        answer = answer.strip('"\'.')
        
        return answer
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.pipe is not None
