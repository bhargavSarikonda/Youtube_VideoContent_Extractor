import json
import logging
import time
from typing import Optional, Dict, Any
import redis.asyncio as aioredis
from app.config import settings

logger = logging.getLogger("agentic_pipeline.redis_service")


class RedisService:
    """
    High-Performance Redis Service:
    - Distributed Caching (MD5 / Video ID) with TTL
    - Token Bucket Distributed Rate Limiter (10,000 concurrent users)
    - Pub/Sub for Live Event Streaming
    - In-Memory resilient fallback when standalone Redis server is offline.
    """

    def __init__(self):
        self.redis_url = settings.REDIS_URL
        self._redis: Optional[aioredis.Redis] = None
        self._memory_cache: Dict[str, Any] = {}
        self._memory_rate_limits: Dict[str, list] = {}
        self._connected = False

    async def connect(self):
        """Initializes Redis connection pool."""
        try:
            self._redis = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=500  # High concurrency pool
            )
            # Test ping with quick timeout
            await self._redis.ping()
            self._connected = True
            logger.info("Connected to Redis successfully.")
        except Exception as e:
            self._connected = False
            logger.warning(f"Redis not available ({e}). Using high-speed in-memory engine fallback.")

    async def close(self):
        if self._redis and self._connected:
            await self._redis.close()

    # =========================================================================
    # Caching Methods (Transcripts & Summaries)
    # =========================================================================
    async def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached JSON result."""
        full_key = f"yt_cache:{cache_key}"
        if self._connected and self._redis:
            try:
                data = await self._redis.get(full_key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.error(f"Redis get failed: {e}")

        # In-memory fallback
        return self._memory_cache.get(full_key)

    async def set_cached_result(self, cache_key: str, data: Dict[str, Any], ttl_seconds: int = None):
        """Caches JSON result with TTL."""
        full_key = f"yt_cache:{cache_key}"
        ttl = ttl_seconds or settings.REDIS_CACHE_TTL_SECONDS
        json_str = json.dumps(data, default=str)

        if self._connected and self._redis:
            try:
                await self._redis.setex(full_key, ttl, json_str)
                return
            except Exception as e:
                logger.error(f"Redis set failed: {e}")

        self._memory_cache[full_key] = data

    # =========================================================================
    # Job State Management
    # =========================================================================
    async def get_job_state(self, job_id: str) -> Optional[Dict[str, Any]]:
        full_key = f"yt_job:{job_id}"
        if self._connected and self._redis:
            try:
                val = await self._redis.get(full_key)
                if val:
                    return json.loads(val)
            except Exception as e:
                logger.error(f"Redis job get error: {e}")

        return self._memory_cache.get(full_key)

    async def save_job_state(self, job_id: str, job_dict: Dict[str, Any]):
        full_key = f"yt_job:{job_id}"
        json_str = json.dumps(job_dict, default=str)
        if self._connected and self._redis:
            try:
                await self._redis.setex(full_key, 86400, json_str)
                # Also publish to PubSub channel
                await self._redis.publish(f"yt_channel:{job_id}", json_str)
            except Exception as e:
                logger.error(f"Redis job save error: {e}")

        self._memory_cache[full_key] = job_dict

    # =========================================================================
    # Distributed Rate Limiting (Token Bucket)
    # =========================================================================
    async def check_rate_limit(self, client_identifier: str, limit: int = 600, window_seconds: int = 60) -> bool:
        """
        Sliding window / Token Bucket rate limiter.
        Returns True if request is within limit, False if rate limit exceeded.
        """
        now = time.time()
        key = f"rate_limit:{client_identifier}"

        if self._connected and self._redis:
            try:
                pipe = self._redis.pipeline()
                pipe.zremrangebyscore(key, 0, now - window_seconds)
                pipe.zadd(key, {str(now): now})
                pipe.zcard(key)
                pipe.expire(key, window_seconds)
                results = await pipe.execute()
                current_requests = results[2]
                return current_requests <= limit
            except Exception as e:
                logger.warning(f"Redis rate limit error: {e}")

        # In-memory rate limiting fallback
        req_list = self._memory_rate_limits.setdefault(key, [])
        cutoff = now - window_seconds
        self._memory_rate_limits[key] = [t for t in req_list if t > cutoff]
        self._memory_rate_limits[key].append(now)
        return len(self._memory_rate_limits[key]) <= limit


# Singleton instance
redis_service = RedisService()
