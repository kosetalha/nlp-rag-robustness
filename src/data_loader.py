"""
Data loading and poisoning module for HotpotQA benchmark.
Uses semantic poisoning with type-consistent distractors.
"""
import json
import random
import re
import spacy
from datasets import load_dataset
from typing import List, Dict, Optional
from src.config import PROCESSED_DATA_DIR, RANDOM_SEED


class HotpotQALoader:
    """
    Loads HotpotQA dataset and creates poisoned benchmarks.
    
    Poisoning Strategy:
    - Semantic: Replaces answers with realistic but wrong entities using NER
    """
    
    def __init__(self, split: str = "validation", sample_size: int = 200):
        """
        Initialize the loader.
        
        Args:
            split: Dataset split ('train' or 'validation')
            sample_size: Number of samples to process
        """
        self.split = split
        self.sample_size = sample_size
        self.dataset = None
        self.processed_data = []
        
        # Load Spacy for Semantic Poisoning
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ Spacy model loaded successfully")
        except OSError:
            print("⚠️ Spacy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Curated distractors for semantic swapping (by entity type)
        self.distractors = {
            "PERSON": [
                "John Smith", "Alice Johnson", "Robert Doe", "Emily Davis",
                "Michael Brown", "Emma Wilson", "David Miller", "Sarah Anderson"
            ],
            "GPE": [
                "New York", "London", "Paris", "Tokyo", "Berlin", 
                "Sydney", "Moscow", "Beijing", "Toronto", "Mumbai"
            ],
            "ORG": [
                "Acme Corp", "Global Industries", "Omni Consumer Products",
                "Massive Dynamic", "Cyberdyne Systems", "Umbrella Corporation"
            ],
            "DATE": ["2025", "1999", "2005", "1984", "2010", "1900", "1975"],
            "CARDINAL": ["42", "100", "seven", "twelve", "fifty"],
        }
    
    def load_data(self) -> None:
        """Downloads HotpotQA from Hugging Face."""
        print(f"📥 Downloading HotpotQA ({self.split} split)...")
        
        dataset = load_dataset("hotpot_qa", "distractor", split=self.split)
        
        # Fixed-seed shuffle guarantees reproducible sampling across runs.
        random.seed(RANDOM_SEED)
        self.dataset = dataset.shuffle(seed=RANDOM_SEED).select(range(self.sample_size))
        
        print(f"✅ Loaded {len(self.dataset)} samples")
    
    def _generate_semantic_distractor(self, answer: str) -> str:
        """
        Semantic poisoning: Creates realistic-looking but wrong entities.
        Harder for models to detect - tests true robustness.
        """
        # If spaCy is unavailable, use neutral fallbacks and simple heuristics.
        if not self.nlp:
            clean_answer = answer.replace(",", "")
            if clean_answer.isdigit():
                offset = random.choice([-500, -200, -100, 100, 200, 500])
                new_val = max(1, int(clean_answer) + offset)
                return f"{new_val:,}" if "," in answer else str(new_val)
            return random.choice([
                "John Smith",
                "New York",
                "Global Industries",
                "1999",
                "42",
            ])
        
        # Numeric perturbation keeps attack plausible while preserving answer format.
        clean_answer = answer.replace(",", "")
        if clean_answer.isdigit():
            offset = random.choice([-500, -200, -100, 100, 200, 500])
            new_val = max(1, int(clean_answer) + offset)  # Keep positive
            # Format with commas if original had them
            if "," in answer:
                return f"{new_val:,}"
            return str(new_val)
        
        # NER-based type detection helps create type-consistent distractors.
        doc = self.nlp(answer)
        label = "UNKNOWN"
        if doc.ents:
            label = doc.ents[0].label_
        
        # Prefer curated distractor pools for common entity classes.
        if label in self.distractors:
            options = [d for d in self.distractors[label] if d.lower() != answer.lower()]
            if options:
                return random.choice(options)
        
        # Heuristic fallback prevents dropping samples when NER coverage is weak.
        answer_lower = answer.lower()
        
        # Short answers (likely acronyms, band names, etc.)
        if len(answer) <= 3:
            return random.choice(["ABC", "XYZ", "CDC", "BBC", "NHK"])
        
        # Nationality/adjective words (ending in -an, -ish, -ese)
        if answer_lower.endswith(('an', 'ish', 'ese', 'ian')):
            return random.choice(["German", "French", "British", "Italian", "Spanish"])
        
        # Single words - likely proper nouns
        if len(answer.split()) == 1:
            return random.choice(["Winston", "Eleanor", "Marcus", "Victoria", "Benedict"])
        
        # Multi-word answers - likely titles, names, or organizations
        return random.choice([
            "The Windsor Committee",
            "Northern Alliance",
            "The Phoenix Foundation", 
            "Sterling & Associates",
            "The Manchester Initiative"
        ])
    
    def _flatten_context(self, row: Dict) -> str:
        """Converts HotpotQA nested context to flat text."""
        # "Title:" prefixes are intentionally preserved because downstream
        # verification splits sources using this marker.
        context_parts = []
        for title, sentences in zip(row['context']['title'], row['context']['sentences']):
            context_parts.append(f"Title: {title}")
            context_parts.append(" ".join(sentences))
            context_parts.append("")  # Empty line between documents
        return "\n".join(context_parts)
    
    def inject_poison(self, strategy: str = "semantic") -> None:
        """
        Creates the poisoned benchmark.
        
        Args:
            strategy: Must be 'semantic'
        """
        if self.dataset is None:
            raise ValueError("Call load_data() first!")
        if strategy != "semantic":
            raise ValueError("Only semantic poisoning is supported in this project version.")
        
        print(f"🧪 Injecting poison... Strategy: {strategy.upper()}")
        
        processed_samples = []
        
        for row in self.dataset:
            question = row['question']
            answer = row['answer']
            
            # Get original context
            original_context = self._flatten_context(row)
            
            # Generate semantic distractor only.
            distractor = self._generate_semantic_distractor(answer)
            
            # Case-insensitive replacement handles capitalization variations.
            pattern = re.compile(re.escape(answer), re.IGNORECASE)
            poisoned_context = pattern.sub(distractor, original_context)
            
            # This flag is critical for filtering valid poisoned samples.
            is_poisoned = distractor in poisoned_context
            
            entry = {
                "id": row["id"],
                "question": question,
                "real_answer": answer,
                "distractor_answer": distractor,
                "clean_context": original_context,
                "poisoned_context": poisoned_context,
                "is_poisoned_successfully": is_poisoned,
                "poison_strategy": strategy
            }
            processed_samples.append(entry)
        
        self.processed_data = processed_samples
        
        # Report helps justify usable sample count in the final analysis.
        success_count = sum(1 for x in self.processed_data if x['is_poisoned_successfully'])
        print(f"✅ Processed {len(self.processed_data)} samples")
        print(f"📊 Successful poisonings: {success_count}/{len(self.processed_data)} ({100*success_count/len(self.processed_data):.1f}%)")
    
    def save_benchmark(self, filename: str) -> str:
        """
        Saves the benchmark to disk.
        
        Args:
            filename: Output filename (saved to data/processed/)
            
        Returns:
            Path to saved file
        """
        if not self.processed_data:
            raise ValueError("No data to save. Call inject_poison() first!")
        
        save_path = PROCESSED_DATA_DIR / filename
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.processed_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Benchmark saved to: {save_path}")
        return str(save_path)
    
    def load_benchmark(self, filename: str) -> List[Dict]:
        """
        Loads a previously created benchmark.
        
        Args:
            filename: Benchmark filename
            
        Returns:
            List of benchmark samples
        """
        load_path = PROCESSED_DATA_DIR / filename
        if not load_path.exists():
            raise FileNotFoundError(f"Benchmark not found: {load_path}")
        
        with open(load_path, "r", encoding="utf-8") as f:
            self.processed_data = json.load(f)
        
        print(f"📂 Loaded {len(self.processed_data)} samples from {load_path}")
        return self.processed_data
    
    def inspect_samples(self, n: int = 3) -> None:
        """Print sample entries for verification."""
        print("\n" + "="*60)
        print("SAMPLE INSPECTION")
        print("="*60)
        
        count = 0
        for item in self.processed_data:
            if item['is_poisoned_successfully']:
                print(f"\n📌 Sample {count + 1}")
                print(f"   Question: {item['question']}")
                print(f"   Real Answer: {item['real_answer']}")
                print(f"   Fake Answer: {item['distractor_answer']}")
                print(f"   Strategy: {item['poison_strategy']}")
                count += 1
                if count >= n:
                    break
        print("="*60)
