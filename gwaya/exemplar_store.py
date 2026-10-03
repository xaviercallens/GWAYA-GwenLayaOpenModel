import hashlib
import json
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

logger = logging.getLogger("ExemplarStore")

class ExemplarStore:
    def __init__(self, store_path: str = "results/gwaya_low_tier/exemplars.json"):
        self.store_path = Path(store_path)
        self.exemplars = []
        self._load()
        
    def _load(self):
        if self.store_path.exists():
            with open(self.store_path, "r") as f:
                self.exemplars = json.load(f)
                
    def _save(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w") as f:
            json.dump(self.exemplars, f, indent=2)
            
    def get_embedding(self, text: str) -> list[float]:
        try:
            import requests  # fixed http://localhost URL; requests has no file:/ scheme handler

            resp = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "all-minilm", "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("embedding", [])
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return []

    def cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        if not v1 or not v2: return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(a * a for a in v2) ** 0.5
        return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

    def add_exemplar(self, problem_hash: str, split: str, goal: str, code: str):
        if split == "test":
            raise ValueError("Leakage guard: test-split problems cannot enter the exemplar store")
        
        emb = self.get_embedding(goal)
        self.exemplars.append({
            "problem_hash": problem_hash,
            "goal": goal,
            "code": code,
            "embedding": emb
        })
        self._save()
        
    def retrieve_exemplars(self, goal: str, top_k: int = 1) -> list[dict[str, Any]]:
        if not self.exemplars:
            return []
            
        emb = self.get_embedding(goal)
        if not emb:
            return []
            
        scored = []
        for ex in self.exemplars:
            if not ex.get("embedding"): continue
            sim = self.cosine_similarity(emb, ex["embedding"])
            scored.append((sim, ex))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:top_k]]
