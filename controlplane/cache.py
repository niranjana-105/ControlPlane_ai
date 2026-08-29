"""
ControlPlane.ai - L1/L2 Hierarchical Semantic Cache
L1: SHA-256 exact-match dictionary (sub-millisecond)
L2: Lightweight TF-IDF cosine-similarity semantic cache (1-3ms)
"""

import hashlib
import time
import math
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any


@dataclass
class CacheEntry:
    prompt_hash: str
    response_text: str
    created_at: float
    hits: int = 0
    token_vector: Optional[Dict[str, float]] = None


@dataclass
class CacheResult:
    hit: bool
    tier: str          # "L1_EXACT" | "L2_SEMANTIC" | "MISS"
    response: Optional[str]
    similarity_score: float
    latency_ms: float
    cache_key: Optional[str] = None


class L1ExactCache:
    """SHA-256 keyed exact-match cache with LRU eviction."""

    def __init__(self, max_size: int = 1024, ttl_seconds: int = 3600):
        self._store: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _hash(self, prompt: str) -> str:
        return hashlib.sha256(prompt.strip().lower().encode()).hexdigest()

    def get(self, prompt: str) -> Optional[CacheEntry]:
        key = self._hash(prompt)
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        if time.time() - entry.created_at > self._ttl:
            del self._store[key]
            self.misses += 1
            return None
        self._store.move_to_end(key)
        entry.hits += 1
        self.hits += 1
        return entry

    def put(self, prompt: str, response: str, token_vector=None) -> str:
        key = self._hash(prompt)
        if len(self._store) >= self._max_size:
            self._store.popitem(last=False)
        self._store[key] = CacheEntry(
            prompt_hash=key, response_text=response,
            created_at=time.time(), token_vector=token_vector)
        return key

    def stats(self) -> Dict[str, Any]:
        return {"size": len(self._store), "hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / max(1, self.hits + self.misses)}


class L2SemanticCache:
    """TF-IDF + cosine-similarity semantic cache. Pure Python, no ML deps."""

    def __init__(self, max_size: int = 256, ttl_seconds: int = 3600):
        self._entries: List[CacheEntry] = []
        self._max_size = max_size
        self._ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    def _tf(self, tokens: List[str]) -> Dict[str, float]:
        counts = Counter(tokens)
        total = max(len(tokens), 1)
        return {tok: count / total for tok, count in counts.items()}

    def _build_vector(self, text: str) -> Dict[str, float]:
        return self._tf(self._tokenize(text))

    @staticmethod
    def _cosine_similarity(va: Dict[str, float], vb: Dict[str, float]) -> float:
        if not va or not vb:
            return 0.0
        common = set(va) & set(vb)
        dot = sum(va[k] * vb[k] for k in common)
        mag_a = math.sqrt(sum(v ** 2 for v in va.values()))
        mag_b = math.sqrt(sum(v ** 2 for v in vb.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def get(self, prompt: str, threshold: float = 0.94) -> Tuple[Optional[CacheEntry], float]:
        now = time.time()
        query_vec = self._build_vector(prompt)
        best_entry = None
        best_score = 0.0
        for entry in self._entries:
            if now - entry.created_at > self._ttl or entry.token_vector is None:
                continue
            score = self._cosine_similarity(query_vec, entry.token_vector)
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_entry and best_score >= threshold:
            best_entry.hits += 1
            self.hits += 1
            return best_entry, best_score
        self.misses += 1
        return None, best_score

    def put(self, prompt: str, response: str) -> None:
        if len(self._entries) >= self._max_size:
            self._entries.pop(0)
        vector = self._build_vector(prompt)
        self._entries.append(CacheEntry(
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:12],
            response_text=response, created_at=time.time(), token_vector=vector))

    def stats(self) -> Dict[str, Any]:
        return {"size": len(self._entries), "hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / max(1, self.hits + self.misses)}


class HierarchicalCache:
    """Tiered cache: L1 (exact) -> L2 (semantic) -> MISS."""

    def __init__(self, l1_max_size=1024, l2_max_size=256, ttl_seconds=3600, similarity_threshold=0.94):
        self.l1 = L1ExactCache(max_size=l1_max_size, ttl_seconds=ttl_seconds)
        self.l2 = L2SemanticCache(max_size=l2_max_size, ttl_seconds=ttl_seconds)
        self._threshold = similarity_threshold

    def lookup(self, prompt: str) -> CacheResult:
        t0 = time.perf_counter()
        entry = self.l1.get(prompt)
        if entry:
            return CacheResult(hit=True, tier="L1_EXACT", response=entry.response_text,
                               similarity_score=1.0, latency_ms=(time.perf_counter()-t0)*1000,
                               cache_key=entry.prompt_hash)
        entry, score = self.l2.get(prompt, threshold=self._threshold)
        if entry:
            return CacheResult(hit=True, tier="L2_SEMANTIC", response=entry.response_text,
                               similarity_score=round(score, 4), latency_ms=(time.perf_counter()-t0)*1000,
                               cache_key=entry.prompt_hash)
        return CacheResult(hit=False, tier="MISS", response=None, similarity_score=0.0,
                           latency_ms=(time.perf_counter()-t0)*1000)

    def store(self, prompt: str, response: str) -> None:
        vec = self.l2._build_vector(prompt)
        self.l1.put(prompt, response, token_vector=vec)
        self.l2.put(prompt, response)

    def stats(self) -> Dict[str, Any]:
        return {"l1": self.l1.stats(), "l2": self.l2.stats(), "threshold": self._threshold}


_global_cache: Optional[HierarchicalCache] = None

def get_cache(l1_max_size=1024, l2_max_size=256, ttl_seconds=3600, similarity_threshold=0.94) -> HierarchicalCache:
    global _global_cache
    if _global_cache is None:
        _global_cache = HierarchicalCache(l1_max_size, l2_max_size, ttl_seconds, similarity_threshold)
    return _global_cache
