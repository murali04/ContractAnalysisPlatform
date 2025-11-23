"""
Caching module for contract analysis results.
Provides in-memory caching to avoid re-analyzing identical obligations.
"""
import hashlib
import json
from functools import lru_cache
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class AnalysisCache:
    """Cache for obligation analysis results."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of cached results
        """
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0
    
    def _generate_key(self, obligation: str, contract_hash: str) -> str:
        """
        Generate cache key from obligation and contract.
        
        Args:
            obligation: Obligation text
            contract_hash: Hash of contract content
            
        Returns:
            Cache key
        """
        combined = f"{obligation}|{contract_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, obligation: str, contract_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result.
        
        Args:
            obligation: Obligation text
            contract_hash: Hash of contract content
            
        Returns:
            Cached result or None if not found
        """
        key = self._generate_key(obligation, contract_hash)
        result = self._cache.get(key)
        
        if result:
            self._hits += 1
            logger.info(f"Cache HIT for obligation: {obligation[:50]}...")
        else:
            self._misses += 1
            logger.info(f"Cache MISS for obligation: {obligation[:50]}...")
        
        return result
    
    def set(self, obligation: str, contract_hash: str, result: Dict[str, Any]) -> None:
        """
        Store result in cache.
        
        Args:
            obligation: Obligation text
            contract_hash: Hash of contract content
            result: Analysis result to cache
        """
        key = self._generate_key(obligation, contract_hash)
        
        # Implement simple LRU by removing oldest if at capacity
        if len(self._cache) >= self.max_size:
            # Remove first (oldest) item
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = result
        logger.info(f"Cached result for obligation: {obligation[:50]}...")
    
    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 2),
            "total_requests": total_requests
        }

# Global cache instance
_global_cache = AnalysisCache(max_size=1000)

def get_cache() -> AnalysisCache:
    """Get global cache instance."""
    return _global_cache

def hash_contract(contract_text: str) -> str:
    """
    Generate hash for contract content.
    
    Args:
        contract_text: Full contract text
        
    Returns:
        SHA256 hash of contract
    """
    return hashlib.sha256(contract_text.encode()).hexdigest()
