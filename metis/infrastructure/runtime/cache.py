"""Shared cache for expensive computations."""

import logging
import time
from collections.abc import Callable
from typing import Any


class StatsStore:
    """
    Shared cache for expensive statistical computations.

    This implementation uses a simple in-memory dictionary cache.
    TODO: Consider implementing:
    - LRU cache with size limits
    - Persistent cache (Redis, file-based)
    - Cache expiration policies
    - Thread-safe cache operations
    - Memory usage monitoring
    """

    def __init__(self, max_size: int = 1000):
        self._cache: dict[str, Any] = {}
        self._access_times: dict[str, float] = {}
        self._hit_count = 0
        self._miss_count = 0
        self._max_size = max_size
        self.logger = logging.getLogger(__name__)

    def get_or_compute(self, key: str, compute_fn: Callable[[], Any]) -> Any:
        """
        Get value from cache or compute it using the provided function.

        Args:
            key: Unique cache key for the computation
            compute_fn: Function to compute the value if not cached

        Returns:
            Cached or computed value
        """
        # Check if already cached
        if key in self._cache:
            self._hit_count += 1
            self._access_times[key] = time.time()
            self.logger.debug(f"Cache HIT for key: {key}")
            return self._cache[key]

        # Not cached, compute the value
        self._miss_count += 1
        self.logger.debug(f"Cache MISS for key: {key}, computing...")

        start_time = time.time()
        try:
            value = compute_fn()
            compute_time = time.time() - start_time

            # Store in cache
            self._store_value(key, value)

            self.logger.debug(f"Computed and cached {key} in {compute_time:.3f}s")
            return value

        except Exception as e:
            self.logger.error(f"Failed to compute value for key {key}: {e}")
            raise

    def _store_value(self, key: str, value: Any) -> None:
        """Store value in cache with size management."""
        current_time = time.time()

        # Check if cache is full
        if len(self._cache) >= self._max_size:
            self._evict_oldest()

        self._cache[key] = value
        self._access_times[key] = current_time

    def _evict_oldest(self) -> None:
        """Evict least recently accessed item."""
        if not self._access_times:
            return

        oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        del self._cache[oldest_key]
        del self._access_times[oldest_key]
        self.logger.debug(f"Evicted cache entry: {oldest_key}")

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
        self._access_times.clear()
        self._hit_count = 0
        self._miss_count = 0
        self.logger.debug("Cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_requests if total_requests > 0 else 0.0

        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }

    def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._cache

    def invalidate(self, key: str) -> bool:
        """
        Remove specific key from cache.

        Args:
            key: Cache key to remove

        Returns:
            True if key was found and removed, False otherwise
        """
        if key in self._cache:
            del self._cache[key]
            del self._access_times[key]
            self.logger.debug(f"Invalidated cache entry: {key}")
            return True
        return False

    def keys(self) -> list[str]:
        """Get list of cached keys."""
        return list(self._cache.keys())

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


# Common cache key generators for typical computations
def make_univariate_key(column: str, computation: str) -> str:
    """Generate cache key for univariate statistics."""
    return f"univariate:{column}:{computation}"


def make_bivariate_key(col1: str, col2: str, computation: str) -> str:
    """Generate cache key for bivariate statistics."""
    # Ensure consistent ordering
    cols = tuple(sorted([col1, col2]))
    return f"bivariate:{cols[0]}:{cols[1]}:{computation}"


def make_dataset_key(computation: str, seed: int = None) -> str:
    """Generate cache key for dataset-level computations."""
    key = f"dataset:{computation}"
    if seed is not None:
        key += f":seed{seed}"
    return key
