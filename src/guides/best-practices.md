# Best Practices for Production-Ready Chutes

This comprehensive guide covers production best practices for building, deploying, and maintaining robust, scalable, and secure Chutes applications in production environments.

## Overview

Production-ready Chutes applications require:

- **Scalable Architecture**: Design for growth and varying loads
- **Security**: Protect data, models, and infrastructure
- **Performance**: Optimize for speed, memory, and resource efficiency
- **Reliability**: Handle failures gracefully with high availability
- **Monitoring**: Complete observability and alerting
- **Maintainability**: Code quality, documentation, and operational procedures

## Application Architecture

### Modular Design Patterns

```python
from abc import ABC, abstractmethod
from typing import Protocol, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum

# Define clear interfaces
class ModelInterface(Protocol):
    """Protocol for AI model implementations."""

    async def load(self) -> None:
        """Load the model into memory."""
        ...

    async def predict(self, input_data: Any) -> Any:
        """Make prediction on input data."""
        ...

    async def unload(self) -> None:
        """Unload model from memory."""
        ...

class CacheInterface(Protocol):
    """Protocol for caching implementations."""

    async def get(self, key: str) -> Optional[Any]:
        ...

    async def set(self, key: str, value: Any, ttl: int = None) -> None:
        ...

    async def delete(self, key: str) -> None:
        ...

# Implement dependency injection
@dataclass
class Dependencies:
    """Application dependencies container."""

    model: ModelInterface
    cache: CacheInterface
    logger: logging.Logger
    metrics: Any  # Metrics collector
    config: Dict[str, Any]

class ServiceBase(ABC):
    """Base class for application services."""

    def __init__(self, deps: Dependencies):
        self.deps = deps
        self.logger = deps.logger
        self.model = deps.model
        self.cache = deps.cache

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        pass

class TextGenerationService(ServiceBase):
    """Text generation service implementation."""

    async def initialize(self) -> None:
        """Initialize text generation service."""
        await self.model.load()
        self.logger.info("Text generation service initialized")

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text with caching and error handling."""

        # Create cache key
        cache_key = self._create_cache_key(prompt, kwargs)

        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            self.logger.info("Cache hit for text generation")
            return cached_result

        # Generate new result
        try:
            result = await self.model.predict(prompt, **kwargs)

            # Cache result
            await self.cache.set(cache_key, result, ttl=3600)

            return result

        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            raise

    def _create_cache_key(self, prompt: str, kwargs: Dict) -> str:
        """Create deterministic cache key."""
        import hashlib
        import json

        key_data = {"prompt": prompt, "params": sorted(kwargs.items())}
        key_str = json.dumps(key_data, sort_keys=True)
        return f"text_gen:{hashlib.md5(key_str.encode()).hexdigest()}"

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.model.unload()
        self.logger.info("Text generation service cleaned up")

# Chute implementation with dependency injection
chute = Chute(username="production", name="text-service")

@chute.on_startup()
async def initialize_app(self):
    """Initialize application with proper dependency injection."""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("text-service")

    # Initialize model
    model = await self._create_model()

    # Initialize cache
    cache = await self._create_cache()

    # Initialize metrics
    metrics = await self._create_metrics()

    # Load configuration
    config = await self._load_config()

    # Create dependencies container
    self.deps = Dependencies(
        model=model,
        cache=cache,
        logger=logger,
        metrics=metrics,
        config=config
    )

    # Initialize services
    self.text_service = TextGenerationService(self.deps)
    await self.text_service.initialize()

async def _create_model(self):
    """Factory method for model creation."""
    # Implementation depends on your specific model
    pass

async def _create_cache(self):
    """Factory method for cache creation."""
    # Could be Redis, Memcached, or in-memory cache
    pass
```

### Configuration Management

```python
import os
from typing import Optional, Union
from pydantic import BaseSettings, Field, validator
from pathlib import Path

class ApplicationConfig(BaseSettings):
    """Production application configuration."""

    # Environment
    environment: str = Field("production", env="APP_ENV")
    debug: bool = Field(False, env="APP_DEBUG")

    # Model settings
    model_name: str = Field(..., env="MODEL_NAME")
    model_path: Optional[str] = Field(None, env="MODEL_PATH")
    max_batch_size: int = Field(8, env="MAX_BATCH_SIZE")

    # Performance settings
    max_workers: int = Field(4, env="MAX_WORKERS")
    request_timeout: float = Field(30.0, env="REQUEST_TIMEOUT")
    max_memory_usage: float = Field(0.9, env="MAX_MEMORY_USAGE")

    # Cache settings
    cache_backend: str = Field("redis", env="CACHE_BACKEND")
    cache_url: str = Field("redis://localhost:6379", env="CACHE_URL")
    cache_ttl: int = Field(3600, env="CACHE_TTL")

    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")

    # Security
    api_key_required: bool = Field(True, env="API_KEY_REQUIRED")
    allowed_origins: list = Field(["*"], env="ALLOWED_ORIGINS")
    rate_limit_requests: int = Field(100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(60, env="RATE_LIMIT_WINDOW")

    # Monitoring
    metrics_enabled: bool = Field(True, env="METRICS_ENABLED")
    health_check_interval: int = Field(30, env="HEALTH_CHECK_INTERVAL")

    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()

    @validator('max_memory_usage')
    def validate_memory_usage(cls, v):
        if not 0.1 <= v <= 1.0:
            raise ValueError('Memory usage must be between 0.1 and 1.0')
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Environment-specific configurations
class DevelopmentConfig(ApplicationConfig):
    """Development environment configuration."""

    environment: str = "development"
    debug: bool = True
    log_level: str = "DEBUG"
    api_key_required: bool = False

class StagingConfig(ApplicationConfig):
    """Staging environment configuration."""

    environment: str = "staging"
    debug: bool = False
    log_level: str = "INFO"

class ProductionConfig(ApplicationConfig):
    """Production environment configuration."""

    environment: str = "production"
    debug: bool = False
    log_level: str = "WARNING"
    api_key_required: bool = True

def get_config() -> ApplicationConfig:
    """Get configuration based on environment."""

    env = os.getenv("APP_ENV", "production").lower()

    config_classes = {
        "development": DevelopmentConfig,
        "staging": StagingConfig,
        "production": ProductionConfig
    }

    config_class = config_classes.get(env, ProductionConfig)
    return config_class()

# Usage in Chute
@chute.on_startup()
async def load_configuration(self):
    """Load and validate configuration."""
    self.config = get_config()

    # Configure logging based on config
    import logging
    logging.basicConfig(
        level=getattr(logging, self.config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    self.logger = logging.getLogger(f"chute.{self.config.environment}")
    self.logger.info(f"Application started in {self.config.environment} mode")
```

## Performance Optimization

### Resource Management

```python
import asyncio
import psutil
import torch
from typing import Optional
import threading
import time

class ResourceManager:
    """Manage compute resources efficiently."""

    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.memory_monitor_task = None
        self.cpu_monitor_task = None
        self._should_monitor = True

        # Resource limits
        self.max_memory_usage = config.max_memory_usage
        self.max_cpu_usage = 0.8

        # Current usage tracking
        self.current_memory_usage = 0.0
        self.current_cpu_usage = 0.0
        self.current_gpu_usage = 0.0

    async def start_monitoring(self):
        """Start resource monitoring."""
        self.memory_monitor_task = asyncio.create_task(self._monitor_memory())
        self.cpu_monitor_task = asyncio.create_task(self._monitor_cpu())

    async def stop_monitoring(self):
        """Stop resource monitoring."""
        self._should_monitor = False

        if self.memory_monitor_task:
            self.memory_monitor_task.cancel()
        if self.cpu_monitor_task:
            self.cpu_monitor_task.cancel()

    async def _monitor_memory(self):
        """Monitor memory usage."""
        while self._should_monitor:
            try:
                # System memory
                memory = psutil.virtual_memory()
                self.current_memory_usage = memory.percent / 100.0

                # GPU memory
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                    self.current_gpu_usage = gpu_memory

                # Check limits
                if self.current_memory_usage > self.max_memory_usage:
                    await self._handle_memory_pressure()

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                print(f"Memory monitoring error: {e}")
                await asyncio.sleep(10)

    async def _monitor_cpu(self):
        """Monitor CPU usage."""
        while self._should_monitor:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                self.current_cpu_usage = cpu_percent / 100.0

                if self.current_cpu_usage > self.max_cpu_usage:
                    await self._handle_cpu_pressure()

                await asyncio.sleep(5)

            except Exception as e:
                print(f"CPU monitoring error: {e}")
                await asyncio.sleep(10)

    async def _handle_memory_pressure(self):
        """Handle high memory usage."""
        print(f"High memory usage detected: {self.current_memory_usage:.2%}")

        # Clear caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Could also:
        # - Reduce batch sizes
        # - Trigger garbage collection
        # - Scale down model precision
        # - Reject new requests temporarily

    async def _handle_cpu_pressure(self):
        """Handle high CPU usage."""
        print(f"High CPU usage detected: {self.current_cpu_usage:.2%}")

        # Could:
        # - Reduce worker threads
        # - Increase request timeouts
        # - Enable request queuing

    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        return {
            "memory_usage": self.current_memory_usage,
            "cpu_usage": self.current_cpu_usage,
            "gpu_usage": self.current_gpu_usage,
            "memory_limit": self.max_memory_usage,
            "cpu_limit": self.max_cpu_usage
        }

    def should_accept_request(self) -> bool:
        """Determine if new requests should be accepted."""
        return (
            self.current_memory_usage < self.max_memory_usage * 0.9 and
            self.current_cpu_usage < self.max_cpu_usage * 0.9
        )

class ModelLoadBalancer:
    """Load balance between multiple model instances."""

    def __init__(self):
        self.model_instances = []
        self.current_instance = 0
        self.instance_loads = {}
        self.lock = threading.Lock()

    def add_model_instance(self, model, instance_id: str):
        """Add a model instance to the pool."""
        self.model_instances.append({
            "model": model,
            "id": instance_id,
            "active_requests": 0
        })
        self.instance_loads[instance_id] = 0

    async def get_model_instance(self):
        """Get the least loaded model instance."""
        with self.lock:
            if not self.model_instances:
                raise RuntimeError("No model instances available")

            # Find instance with lowest load
            best_instance = min(
                self.model_instances,
                key=lambda x: x["active_requests"]
            )

            best_instance["active_requests"] += 1
            return best_instance

    async def release_model_instance(self, instance):
        """Release a model instance back to the pool."""
        with self.lock:
            instance["active_requests"] = max(0, instance["active_requests"] - 1)

    async def get_load_statistics(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self.lock:
            total_requests = sum(inst["active_requests"] for inst in self.model_instances)

            return {
                "total_instances": len(self.model_instances),
                "total_active_requests": total_requests,
                "instance_loads": {
                    inst["id"]: inst["active_requests"]
                    for inst in self.model_instances
                }
            }

# Usage in Chute
@chute.on_startup()
async def initialize_resource_management(self):
    """Initialize resource management."""
    self.resource_manager = ResourceManager(self.config)
    self.model_balancer = ModelLoadBalancer()

    await self.resource_manager.start_monitoring()

    # Add model instances (if using multiple instances)
    for i in range(self.config.max_workers):
        model_instance = await self._create_model_instance(f"model_{i}")
        self.model_balancer.add_model_instance(model_instance, f"model_{i}")
```

### Caching Strategies

```python
import hashlib
import json
import time
from typing import Any, Optional, Union
import redis
import pickle
from dataclasses import dataclass

@dataclass
class CacheConfig:
    """Cache configuration."""
    backend: str = "redis"
    url: str = "redis://localhost:6379"
    default_ttl: int = 3600
    max_memory: str = "1gb"
    eviction_policy: str = "allkeys-lru"

class CacheInterface(ABC):
    """Abstract cache interface."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        pass

    @abstractmethod
    async def clear(self) -> None:
        pass

class RedisCache(CacheInterface):
    """Redis-based cache implementation."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_client = None

    async def connect(self):
        """Connect to Redis."""
        self.redis_client = redis.Redis.from_url(
            self.config.url,
            decode_responses=False  # Handle binary data
        )

        # Configure memory and eviction
        await self.redis_client.config_set("maxmemory", self.config.max_memory)
        await self.redis_client.config_set("maxmemory-policy", self.config.eviction_policy)

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            data = await self.redis_client.get(key)
            if data is None:
                return None
            return pickle.loads(data)
        except Exception as e:
            print(f"Cache get error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        try:
            serialized_value = pickle.dumps(value)
            ttl = ttl or self.config.default_ttl
            await self.redis_client.setex(key, ttl, serialized_value)
        except Exception as e:
            print(f"Cache set error: {e}")

    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        try:
            await self.redis_client.delete(key)
        except Exception as e:
            print(f"Cache delete error: {e}")

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return bool(await self.redis_client.exists(key))
        except Exception as e:
            print(f"Cache exists error: {e}")
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        try:
            await self.redis_client.flushdb()
        except Exception as e:
            print(f"Cache clear error: {e}")

class InMemoryCache(CacheInterface):
    """In-memory cache implementation."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = {}
        self.timestamps = {}
        self.max_size = 10000  # Maximum number of entries

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None

        # Check TTL
        if self._is_expired(key):
            await self.delete(key)
            return None

        return self.cache[key]

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        # Evict old entries if cache is full
        if len(self.cache) >= self.max_size:
            await self._evict_lru()

        self.cache[key] = value
        self.timestamps[key] = {
            "created": time.time(),
            "ttl": ttl or self.config.default_ttl,
            "accessed": time.time()
        }

    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if key not in self.cache:
            return False

        if self._is_expired(key):
            await self.delete(key)
            return False

        # Update access time
        self.timestamps[key]["accessed"] = time.time()
        return True

    async def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.timestamps.clear()

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.timestamps:
            return True

        timestamp_info = self.timestamps[key]
        age = time.time() - timestamp_info["created"]
        return age > timestamp_info["ttl"]

    async def _evict_lru(self):
        """Evict least recently used entries."""
        if not self.timestamps:
            return

        # Find least recently accessed key
        lru_key = min(
            self.timestamps.keys(),
            key=lambda k: self.timestamps[k]["accessed"]
        )

        await self.delete(lru_key)

class SmartCacheManager:
    """Intelligent caching with multiple strategies."""

    def __init__(self, cache: CacheInterface):
        self.cache = cache
        self.hit_rates = {}
        self.access_patterns = {}

    async def get_with_fallback(
        self,
        key: str,
        fallback_func: callable,
        ttl: Optional[int] = None,
        cache_negative: bool = False
    ) -> Any:
        """Get from cache with automatic fallback to function."""

        # Try cache first
        cached_value = await self.cache.get(key)
        if cached_value is not None:
            self._record_hit(key)
            return cached_value

        # Call fallback function
        try:
            value = await fallback_func() if asyncio.iscoroutinefunction(fallback_func) else fallback_func()

            # Cache the result
            if value is not None or cache_negative:
                await self.cache.set(key, value, ttl)

            self._record_miss(key)
            return value

        except Exception as e:
            self._record_miss(key)
            raise

    def create_cache_key(self, prefix: str, **kwargs) -> str:
        """Create a deterministic cache key."""
        key_data = {
            "prefix": prefix,
            "args": sorted(kwargs.items())
        }

        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()

        return f"{prefix}:{key_hash[:16]}"

    def _record_hit(self, key: str):
        """Record cache hit."""
        if key not in self.hit_rates:
            self.hit_rates[key] = {"hits": 0, "misses": 0}
        self.hit_rates[key]["hits"] += 1

    def _record_miss(self, key: str):
        """Record cache miss."""
        if key not in self.hit_rates:
            self.hit_rates[key] = {"hits": 0, "misses": 0}
        self.hit_rates[key]["misses"] += 1

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_hits = sum(stats["hits"] for stats in self.hit_rates.values())
        total_misses = sum(stats["misses"] for stats in self.hit_rates.values())
        total_requests = total_hits + total_misses

        hit_rate = total_hits / total_requests if total_requests > 0 else 0

        return {
            "total_requests": total_requests,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": hit_rate,
            "key_statistics": self.hit_rates
        }

# Usage with Chute
@chute.on_startup()
async def initialize_caching(self):
    """Initialize smart caching system."""

    cache_config = CacheConfig(
        backend=self.config.cache_backend,
        url=self.config.cache_url,
        default_ttl=self.config.cache_ttl
    )

    if cache_config.backend == "redis":
        cache_impl = RedisCache(cache_config)
        await cache_impl.connect()
    else:
        cache_impl = InMemoryCache(cache_config)

    self.cache_manager = SmartCacheManager(cache_impl)

@chute.cord(public_api_path="/generate_cached", method="POST")
async def generate_with_caching(self, prompt: str, max_tokens: int = 100):
    """Generate text with intelligent caching."""

    cache_key = self.cache_manager.create_cache_key(
        "text_generation",
        prompt=prompt,
        max_tokens=max_tokens,
        model=self.config.model_name
    )

    async def generate_fallback():
        return await self.model.generate(prompt, max_tokens=max_tokens)

    result = await self.cache_manager.get_with_fallback(
        cache_key,
        generate_fallback,
        ttl=3600  # Cache for 1 hour
    )

    return result
```

## Security Best Practices

### Authentication and Authorization

```python
import jwt
import time
import secrets
from typing import Optional, List, Dict, Any
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import hashlib
import hmac

class APIKeyManager:
    """Manage API keys for authentication."""

    def __init__(self):
        self.api_keys = {}  # In production, use a database
        self.key_permissions = {}
        self.usage_tracking = {}

    def create_api_key(
        self,
        user_id: str,
        permissions: List[str] = None,
        rate_limit: int = 1000
    ) -> str:
        """Create a new API key."""

        api_key = self._generate_api_key()

        self.api_keys[api_key] = {
            "user_id": user_id,
            "created_at": time.time(),
            "last_used": None,
            "is_active": True,
            "rate_limit": rate_limit
        }

        self.key_permissions[api_key] = permissions or ["basic"]
        self.usage_tracking[api_key] = {
            "requests_today": 0,
            "last_reset": time.time()
        }

        return api_key

    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return key info."""

        if api_key not in self.api_keys:
            return None

        key_info = self.api_keys[api_key]

        if not key_info["is_active"]:
            return None

        # Update last used
        key_info["last_used"] = time.time()

        return key_info

    def check_rate_limit(self, api_key: str) -> bool:
        """Check if API key has exceeded rate limit."""

        if api_key not in self.usage_tracking:
            return False

        usage = self.usage_tracking[api_key]
        key_info = self.api_keys[api_key]

        # Reset daily counter if needed
        if time.time() - usage["last_reset"] > 86400:  # 24 hours
            usage["requests_today"] = 0
            usage["last_reset"] = time.time()

        # Check rate limit
        if usage["requests_today"] >= key_info["rate_limit"]:
            return True

        # Increment counter
        usage["requests_today"] += 1
        return False

    def has_permission(self, api_key: str, permission: str) -> bool:
        """Check if API key has specific permission."""

        if api_key not in self.key_permissions:
            return False

        permissions = self.key_permissions[api_key]
        return permission in permissions or "admin" in permissions

    def _generate_api_key(self) -> str:
        """Generate a secure API key."""
        return secrets.token_urlsafe(32)

class JWTManager:
    """Manage JWT tokens for authentication."""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm

    def create_token(
        self,
        user_id: str,
        permissions: List[str] = None,
        expires_in: int = 3600
    ) -> str:
        """Create a JWT token."""

        now = time.time()
        payload = {
            "user_id": user_id,
            "permissions": permissions or ["basic"],
            "iat": now,
            "exp": now + expires_in,
            "iss": "chutes-api"
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return payload."""

        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": True}
            )
            return payload

        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

class SecurityMiddleware:
    """Security middleware for Chutes applications."""

    def __init__(
        self,
        api_key_manager: APIKeyManager,
        jwt_manager: Optional[JWTManager] = None,
        require_https: bool = True
    ):
        self.api_key_manager = api_key_manager
        self.jwt_manager = jwt_manager
        self.require_https = require_https
        self.security = HTTPBearer(auto_error=False)

    async def __call__(self, request: Request, call_next):
        """Process request with security checks."""

        # Check HTTPS requirement
        if self.require_https and request.url.scheme != "https":
            if not request.url.hostname in ["localhost", "127.0.0.1"]:
                return JSONResponse(
                    status_code=400,
                    content={"error": "HTTPS required"}
                )

        # Add security headers
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response

    async def authenticate_request(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
    ) -> Dict[str, Any]:
        """Authenticate incoming request."""

        # Check for API key in headers
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return await self._authenticate_api_key(api_key)

        # Check for JWT token
        if credentials:
            return await self._authenticate_jwt(credentials.credentials)

        # Check for API key in query params (less secure, for compatibility)
        api_key = request.query_params.get("api_key")
        if api_key:
            return await self._authenticate_api_key(api_key)

        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )

    async def _authenticate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Authenticate using API key."""

        key_info = self.api_key_manager.validate_api_key(api_key)
        if not key_info:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )

        # Check rate limit
        if self.api_key_manager.check_rate_limit(api_key):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )

        return {
            "auth_type": "api_key",
            "user_id": key_info["user_id"],
            "permissions": self.api_key_manager.key_permissions.get(api_key, []),
            "api_key": api_key
        }

    async def _authenticate_jwt(self, token: str) -> Dict[str, Any]:
        """Authenticate using JWT token."""

        if not self.jwt_manager:
            raise HTTPException(
                status_code=401,
                detail="JWT authentication not configured"
            )

        payload = self.jwt_manager.validate_token(token)

        return {
            "auth_type": "jwt",
            "user_id": payload["user_id"],
            "permissions": payload.get("permissions", []),
            "token_payload": payload
        }

# Usage in Chute
@chute.on_startup()
async def initialize_security(self):
    """Initialize security components."""

    # Create API key manager
    self.api_key_manager = APIKeyManager()

    # Create JWT manager if enabled
    jwt_secret = os.getenv("JWT_SECRET_KEY")
    if jwt_secret:
        self.jwt_manager = JWTManager(jwt_secret)
    else:
        self.jwt_manager = None

    # Create security middleware
    self.security_middleware = SecurityMiddleware(
        self.api_key_manager,
        self.jwt_manager,
        require_https=self.config.environment == "production"
    )

    # Add middleware to app
    self.app.middleware("http")(self.security_middleware)

# Protected endpoint example
@chute.cord(public_api_path="/protected_generate", method="POST")
async def protected_generate(
    self,
    prompt: str,
    auth_info: Dict[str, Any] = Depends(lambda: self.security_middleware.authenticate_request)
):
    """Protected text generation endpoint."""

    # Check permissions
    if not self.api_key_manager.has_permission(auth_info.get("api_key"), "text_generation"):
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions for text generation"
        )

    # Generate text
    result = await self.model.generate(prompt)

    # Log usage for audit
    self.logger.info(
        f"Text generation request",
        extra={
            "user_id": auth_info["user_id"],
            "auth_type": auth_info["auth_type"],
            "prompt_length": len(prompt)
        }
    )

    return result
```

### Input Sanitization and Validation

```python
import re
import html
import bleach
from typing import Union, List, Dict, Any
import base64
import mimetypes

class InputSanitizer:
    """Comprehensive input sanitization."""

    @staticmethod
    def sanitize_text(
        text: str,
        max_length: int = 10000,
        allow_html: bool = False,
        allowed_tags: List[str] = None
    ) -> str:
        """Sanitize text input."""

        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # Remove null bytes and control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')

        # Handle HTML
        if allow_html and allowed_tags:
            # Use bleach to clean HTML
            text = bleach.clean(
                text,
                tags=allowed_tags,
                attributes={},
                strip=True
            )
        else:
            # Escape HTML entities
            text = html.escape(text)

        # Check length
        if len(text) > max_length:
            raise ValueError(f"Input too long (max {max_length} characters)")

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            raise ValueError("Input cannot be empty after sanitization")

        return text

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage."""

        if not filename:
            raise ValueError("Filename cannot be empty")

        # Remove path separators and dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = re.sub(r'\.\.+', '.', filename)  # Remove directory traversal

        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')

        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext

        if not filename:
            raise ValueError("Filename invalid after sanitization")

        return filename

    @staticmethod
    def validate_base64(data: str, max_size_mb: int = 10) -> bytes:
        """Validate and decode base64 data."""

        try:
            # Remove data URL prefix if present
            if data.startswith('data:'):
                header, data = data.split(',', 1)

            # Decode base64
            decoded = base64.b64decode(data)

            # Check size
            if len(decoded) > max_size_mb * 1024 * 1024:
                raise ValueError(f"Data too large (max {max_size_mb}MB)")

            return decoded

        except Exception as e:
            raise ValueError(f"Invalid base64 data: {e}")

    @staticmethod
    def validate_url(url: str, allowed_schemes: List[str] = None) -> str:
        """Validate URL format and scheme."""

        from urllib.parse import urlparse

        if not url:
            raise ValueError("URL cannot be empty")

        try:
            parsed = urlparse(url)
        except Exception:
            raise ValueError("Invalid URL format")

        # Check scheme
        allowed_schemes = allowed_schemes or ['http', 'https']
        if parsed.scheme not in allowed_schemes:
            raise ValueError(f"URL scheme must be one of: {allowed_schemes}")

        # Check hostname
        if not parsed.netloc:
            raise ValueError("URL must have a hostname")

        # Prevent localhost access in production
        if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
            raise ValueError("Localhost URLs not allowed")

        return url

class ContentValidator:
    """Validate content for safety and appropriateness."""

    def __init__(self):
        # Initialize content filters
        self.forbidden_patterns = [
            r'\b(?:password|secret|key|token)\b',
            r'\b(?:attack|hack|exploit)\b',
            r'<script[^>]*>.*?</script>',
        ]

        # Compile patterns for efficiency
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for pattern in self.forbidden_patterns
        ]

    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """Validate AI prompt for safety."""

        issues = []

        # Check for forbidden patterns
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(prompt):
                issues.append(f"Contains forbidden pattern {i+1}")

        # Check for excessive repetition
        if self._has_excessive_repetition(prompt):
            issues.append("Contains excessive repetition")

        # Check for potential prompt injection
        if self._has_prompt_injection(prompt):
            issues.append("Potential prompt injection detected")

        return {
            "is_safe": len(issues) == 0,
            "issues": issues,
            "risk_score": len(issues) / len(self.compiled_patterns)
        }

    def _has_excessive_repetition(self, text: str, threshold: float = 0.5) -> bool:
        """Check for excessive character/word repetition."""

        if len(text) < 10:
            return False

        # Check character repetition
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1

        max_char_freq = max(char_counts.values()) / len(text)
        if max_char_freq > threshold:
            return True

        # Check word repetition
        words = text.split()
        if len(words) < 5:
            return False

        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        max_word_freq = max(word_counts.values()) / len(words)
        return max_word_freq > threshold

    def _has_prompt_injection(self, text: str) -> bool:
        """Check for potential prompt injection attempts."""

        injection_patterns = [
            r'ignore\s+(?:previous|above|all)\s+(?:instructions|commands)',
            r'system\s*:\s*you\s+are\s+now',
            r'forget\s+(?:everything|all)\s+(?:above|before)',
            r'new\s+(?:role|persona|character)',
        ]

        for pattern in injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

# Validation middleware
class ValidationMiddleware:
    """Middleware for request validation."""

    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.content_validator = ContentValidator()

    async def __call__(self, request: Request, call_next):
        """Validate incoming requests."""

        # Validate request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 50 * 1024 * 1024:  # 50MB
            return JSONResponse(
                status_code=413,
                content={"error": "Request too large"}
            )

        # Validate content type for POST requests
        if request.method == "POST":
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith(("application/json", "multipart/form-data")):
                return JSONResponse(
                    status_code=415,
                    content={"error": "Unsupported content type"}
                )

        response = await call_next(request)
        return response

# Usage in endpoint
@chute.cord(public_api_path="/safe_generate", method="POST")
async def safe_generate(self, prompt: str):
    """Generate text with comprehensive input validation."""

    # Sanitize input
    try:
        sanitized_prompt = InputSanitizer.sanitize_text(prompt, max_length=5000)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Validate content
    validation_result = ContentValidator().validate_prompt(sanitized_prompt)
    if not validation_result["is_safe"]:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unsafe content detected",
                "issues": validation_result["issues"]
            }
        )

    # Generate response
    result = await self.model.generate(sanitized_prompt)

    return {
        "generated_text": result,
        "validation_passed": True,
        "risk_score": validation_result["risk_score"]
    }
```

## Monitoring and Observability

### Comprehensive Logging

```python
import logging
import json
import time
import traceback
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
from contextlib import contextmanager

class StructuredLogger:
    """Structured logging for better observability."""

    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Configure JSON formatter
        handler = logging.StreamHandler()
        formatter = JSONFormatter()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Request context
        self.request_context = {}

    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, error: Exception = None, **kwargs):
        """Log error message with context."""
        extra_data = kwargs.copy()

        if error:
            extra_data.update({
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc()
            })

        self._log("ERROR", message, **extra_data)

    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method."""

        log_data = {
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "level": level,
            **self.request_context,
            **kwargs
        }

        getattr(self.logger, level.lower())(json.dumps(log_data))

    @contextmanager
    def request_context(self, **context):
        """Add request context to all logs."""
        old_context = self.request_context.copy()
        self.request_context.update(context)
        try:
            yield
        finally:
            self.request_context = old_context

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        """Format log record as JSON."""

        # Check if message is already JSON
        try:
            return json.loads(record.getMessage())
        except (json.JSONDecodeError, TypeError):
            # Create JSON structure
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }

            # Add exception info if present
            if record.exc_info:
                log_entry["exception"] = self.formatException(record.exc_info)

            return json.dumps(log_entry)

class PerformanceMonitor:
    """Monitor application performance metrics."""

    def __init__(self):
        self.metrics = {
            "request_count": 0,
            "total_request_time": 0.0,
            "error_count": 0,
            "request_times": [],
            "memory_usage": [],
            "gpu_usage": []
        }

        self.start_time = time.time()

    @contextmanager
    def measure_request(self, endpoint: str, method: str):
        """Measure request performance."""

        start_time = time.time()

        try:
            yield

            # Record successful request
            duration = time.time() - start_time
            self._record_request(endpoint, method, duration, success=True)

        except Exception as e:
            # Record failed request
            duration = time.time() - start_time
            self._record_request(endpoint, method, duration, success=False)
            raise

    def _record_request(self, endpoint: str, method: str, duration: float, success: bool):
        """Record request metrics."""

        self.metrics["request_count"] += 1
        self.metrics["total_request_time"] += duration

        if not success:
            self.metrics["error_count"] += 1

        # Keep last 1000 request times for percentile calculation
        self.metrics["request_times"].append(duration)
        if len(self.metrics["request_times"]) > 1000:
            self.metrics["request_times"].pop(0)

    def record_system_metrics(self):
        """Record system metrics."""

        # Memory usage
        import psutil
        memory = psutil.virtual_memory()
        self.metrics["memory_usage"].append({
            "timestamp": time.time(),
            "percent": memory.percent,
            "available_gb": memory.available / (1024**3)
        })

        # GPU usage
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            self.metrics["gpu_usage"].append({
                "timestamp": time.time(),
                "memory_percent": gpu_memory * 100
            })

        # Keep only last hour of system metrics
        cutoff_time = time.time() - 3600
        self.metrics["memory_usage"] = [
            m for m in self.metrics["memory_usage"]
            if m["timestamp"] > cutoff_time
        ]
        self.metrics["gpu_usage"] = [
            g for g in self.metrics["gpu_usage"]
            if g["timestamp"] > cutoff_time
        ]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""

        request_times = self.metrics["request_times"]

        if request_times:
            sorted_times = sorted(request_times)
            n = len(sorted_times)

            percentiles = {
                "p50": sorted_times[int(n * 0.5)],
                "p90": sorted_times[int(n * 0.9)],
                "p95": sorted_times[int(n * 0.95)],
                "p99": sorted_times[int(n * 0.99)] if n > 100 else sorted_times[-1]
            }

            avg_response_time = self.metrics["total_request_time"] / self.metrics["request_count"]
        else:
            percentiles = {}
            avg_response_time = 0

        uptime = time.time() - self.start_time
        error_rate = self.metrics["error_count"] / max(1, self.metrics["request_count"])

        return {
            "uptime_seconds": uptime,
            "request_count": self.metrics["request_count"],
            "error_count": self.metrics["error_count"],
            "error_rate": error_rate,
            "avg_response_time": avg_response_time,
            "response_time_percentiles": percentiles,
            "requests_per_second": self.metrics["request_count"] / uptime if uptime > 0 else 0
        }

# Middleware integration
class ObservabilityMiddleware:
    """Middleware for observability and monitoring."""

    def __init__(self, logger: StructuredLogger, performance_monitor: PerformanceMonitor):
        self.logger = logger
        self.performance_monitor = performance_monitor

    async def __call__(self, request: Request, call_next):
        """Process request with full observability."""

        # Generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))

        # Extract request info
        request_info = {
            "correlation_id": correlation_id,
            "method": request.method,
            "path": request.url.path,
            "user_agent": request.headers.get("User-Agent"),
            "remote_addr": request.client.host if request.client else None
        }

        start_time = time.time()

        with self.logger.request_context(**request_info):
            self.logger.info("Request started")

            try:
                with self.performance_monitor.measure_request(request.url.path, request.method):
                    response = await call_next(request)

                # Log successful response
                duration = time.time() - start_time
                self.logger.info(
                    "Request completed",
                    status_code=response.status_code,
                    duration_ms=duration * 1000
                )

                # Add correlation ID to response
                response.headers["X-Correlation-ID"] = correlation_id

                return response

            except Exception as e:
                # Log error
                duration = time.time() - start_time
                self.logger.error(
                    "Request failed",
                    error=e,
                    duration_ms=duration * 1000
                )
                raise

# Usage in Chute
@chute.on_startup()
async def initialize_observability(self):
    """Initialize observability components."""

    # Create structured logger
    self.structured_logger = StructuredLogger(
        f"chute.{self.config.environment}",
        level=self.config.log_level
    )

    # Create performance monitor
    self.performance_monitor = PerformanceMonitor()

    # Add observability middleware
    observability_middleware = ObservabilityMiddleware(
        self.structured_logger,
        self.performance_monitor
    )
    self.app.middleware("http")(observability_middleware)

    # Start background system metrics collection
    asyncio.create_task(self._collect_system_metrics())

async def _collect_system_metrics(self):
    """Background task to collect system metrics."""
    while True:
        try:
            self.performance_monitor.record_system_metrics()
            await asyncio.sleep(30)  # Collect every 30 seconds
        except Exception as e:
            self.structured_logger.error("System metrics collection failed", error=e)
            await asyncio.sleep(60)  # Wait longer on error

@chute.cord(public_api_path="/metrics", method="GET")
async def get_metrics(self):
    """Get application metrics."""
    return self.performance_monitor.get_metrics_summary()
```

## Testing and Quality Assurance

### Comprehensive Testing Strategy

```python
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import tempfile
import json

class ChutesTestSuite:
    """Comprehensive test suite for Chutes applications."""

    @pytest.fixture
    def test_client(self):
        """Create test client."""
        from your_chute_app import chute  # Import your actual chute

        # Override dependencies for testing
        with patch.object(chute, 'model', AsyncMock()):
            client = TestClient(chute.app)
            yield client

    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        model = AsyncMock()
        model.generate = AsyncMock(return_value="Mock generated text")
        return model

    @pytest.fixture
    def sample_request_data(self):
        """Sample request data for testing."""
        return {
            "prompt": "Test prompt",
            "max_tokens": 100,
            "temperature": 0.7
        }

    # Unit Tests
    def test_input_validation(self, test_client, sample_request_data):
        """Test input validation."""

        # Test valid input
        response = test_client.post("/generate", json=sample_request_data)
        assert response.status_code == 200

        # Test missing required field
        invalid_data = sample_request_data.copy()
        del invalid_data["prompt"]
        response = test_client.post("/generate", json=invalid_data)
        assert response.status_code == 422

        # Test invalid data types
        invalid_data = sample_request_data.copy()
        invalid_data["max_tokens"] = "not_a_number"
        response = test_client.post("/generate", json=invalid_data)
        assert response.status_code == 422

    def test_error_handling(self, test_client, sample_request_data):
        """Test error handling."""

        # Mock model to raise exception
        with patch('your_chute_app.model.generate', side_effect=Exception("Model error")):
            response = test_client.post("/generate", json=sample_request_data)
            assert response.status_code == 500
            assert "error" in response.json()

    def test_authentication(self, test_client, sample_request_data):
        """Test authentication mechanisms."""

        # Test without authentication
        response = test_client.post("/protected_generate", json=sample_request_data)
        assert response.status_code == 401

        # Test with valid API key
        headers = {"X-API-Key": "valid_test_key"}
        response = test_client.post("/protected_generate", json=sample_request_data, headers=headers)
        # Should succeed if API key is properly configured

        # Test with invalid API key
        headers = {"X-API-Key": "invalid_key"}
        response = test_client.post("/protected_generate", json=sample_request_data, headers=headers)
        assert response.status_code == 401

    def test_rate_limiting(self, test_client, sample_request_data):
        """Test rate limiting."""

        headers = {"X-API-Key": "test_key"}

        # Make multiple requests rapidly
        responses = []
        for _ in range(10):
            response = test_client.post("/generate", json=sample_request_data, headers=headers)
            responses.append(response.status_code)

        # Should eventually hit rate limit
        assert 429 in responses  # Too Many Requests

    # Integration Tests
    def test_full_text_generation_flow(self, test_client, sample_request_data):
        """Test complete text generation workflow."""

        response = test_client.post("/generate", json=sample_request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "generated_text" in data
        assert isinstance(data["generated_text"], str)
        assert len(data["generated_text"]) > 0

    def test_caching_functionality(self, test_client, sample_request_data):
        """Test caching behavior."""

        # First request
        response1 = test_client.post("/generate_cached", json=sample_request_data)
        assert response1.status_code == 200

        # Second identical request (should be cached)
        response2 = test_client.post("/generate_cached", json=sample_request_data)
        assert response2.status_code == 200

        # Responses should be identical for cached requests
        assert response1.json() == response2.json()

    def test_health_checks(self, test_client):
        """Test health check endpoints."""

        response = test_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "overall_status" in data
        assert data["overall_status"] in ["healthy", "degraded", "critical"]

    # Performance Tests
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, test_client, sample_request_data):
        """Test handling of concurrent requests."""

        async def make_request():
            response = test_client.post("/generate", json=sample_request_data)
            return response.status_code

        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All requests should succeed (or fail gracefully)
        assert all(status in [200, 500, 503] for status in results)

    def test_memory_usage(self, test_client, sample_request_data):
        """Test memory usage during requests."""

        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Make multiple requests
        for _ in range(50):
            response = test_client.post("/generate", json=sample_request_data)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024

    # Security Tests
    def test_input_sanitization(self, test_client):
        """Test input sanitization."""

        malicious_inputs = [
            {"prompt": "<script>alert('xss')</script>"},
            {"prompt": "../../etc/passwd"},
            {"prompt": "'; DROP TABLE users; --"},
            {"prompt": "\x00\x01\x02"},  # Null bytes
        ]

        for malicious_input in malicious_inputs:
            response = test_client.post("/safe_generate", json=malicious_input)
            # Should either sanitize or reject
            assert response.status_code in [200, 400]

            if response.status_code == 200:
                # Check that malicious content was sanitized
                data = response.json()
                assert "<script>" not in data.get("generated_text", "")

    def test_file_upload_security(self, test_client):
        """Test file upload security."""

        # Test malicious file types
        malicious_files = [
            ("test.exe", b"MZ\x90\x00"),  # Executable
            ("test.php", b"<?php echo 'hello'; ?>"),  # PHP script
            ("../../../etc/passwd", b"root:x:0:0:root"),  # Path traversal
        ]

        for filename, content in malicious_files:
            files = {"file": (filename, content, "application/octet-stream")}
            response = test_client.post("/upload", files=files)

            # Should reject malicious files
            assert response.status_code in [400, 415, 422]

    # Load Tests
    def test_sustained_load(self, test_client, sample_request_data):
        """Test sustained load handling."""

        import time

        start_time = time.time()
        request_count = 0
        errors = 0

        # Run for 30 seconds
        while time.time() - start_time < 30:
            response = test_client.post("/generate", json=sample_request_data)
            request_count += 1

            if response.status_code != 200:
                errors += 1

            time.sleep(0.1)  # 10 requests per second

        error_rate = errors / request_count

        # Error rate should be low
        assert error_rate < 0.1  # Less than 10% error rate

    # Edge Case Tests
    def test_edge_cases(self, test_client):
        """Test edge cases and boundary conditions."""

        edge_cases = [
            {"prompt": ""},  # Empty prompt
            {"prompt": "a" * 10000},  # Very long prompt
            {"prompt": "Simple test", "max_tokens": 0},  # Zero tokens
            {"prompt": "Simple test", "max_tokens": 10000},  # Too many tokens
            {"prompt": "Simple test", "temperature": -1},  # Invalid temperature
            {"prompt": "Simple test", "temperature": 3},  # Temperature too high
        ]

        for case in edge_cases:
            response = test_client.post("/generate", json=case)
            # Should handle gracefully (either success or proper error)
            assert response.status_code in [200, 400, 422]

    def test_special_characters(self, test_client):
        """Test handling of special characters."""

        special_prompts = [
            "Testing emoji: 😀🚀🎉",
            "Unicode: こんにちは世界",
            "Math symbols: ∑∫∂∆∇",
            "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?",
            "Mixed: English + 中文 + العربية + русский",
        ]

        for prompt in special_prompts:
            response = test_client.post("/generate", json={"prompt": prompt})
            assert response.status_code in [200, 400]

# Configuration for pytest
class TestConfig:
    """Test configuration."""

    @pytest.fixture(scope="session", autouse=True)
    def setup_test_environment(self):
        """Set up test environment."""

        # Set test environment variables
        import os
        os.environ["APP_ENV"] = "testing"
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ["API_KEY_REQUIRED"] = "false"

        # Create temporary directories for test files
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["TEMP_DIR"] = temp_dir
            yield

    @pytest.fixture(autouse=True)
    def isolate_tests(self):
        """Isolate each test."""

        # Clear any cached data
        # Reset any global state
        yield

        # Cleanup after test

# Run tests with coverage
if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--cov=your_chute_app",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])
```

## Deployment Best Practices

### Production Deployment Checklist

```python
class ProductionDeploymentChecklist:
    """Comprehensive production deployment checklist."""

    CHECKLIST = {
        "Security": [
            "✓ Enable HTTPS/TLS encryption",
            "✓ Configure API authentication",
            "✓ Set up rate limiting",
            "✓ Enable CORS properly",
            "✓ Sanitize all inputs",
            "✓ Use secure headers",
            "✓ Regular security scans",
            "✓ Secrets management",
        ],

        "Performance": [
            "✓ Load testing completed",
            "✓ Memory usage optimized",
            "✓ GPU utilization optimized",
            "✓ Caching implemented",
            "✓ Connection pooling",
            "✓ Request timeouts configured",
            "✓ Auto-scaling rules",
            "✓ CDN for static assets",
        ],

        "Reliability": [
            "✓ Health checks implemented",
            "✓ Error handling comprehensive",
            "✓ Circuit breakers configured",
            "✓ Retry mechanisms",
            "✓ Graceful shutdown",
            "✓ Backup procedures",
            "✓ Disaster recovery plan",
            "✓ Zero-downtime deployments",
        ],

        "Monitoring": [
            "✓ Application metrics",
            "✓ System metrics",
            "✓ Error tracking",
            "✓ Log aggregation",
            "✓ Alert configuration",
            "✓ Dashboard setup",
            "✓ Performance monitoring",
            "✓ User analytics",
        ],

        "Documentation": [
            "✓ API documentation",
            "✓ Deployment guide",
            "✓ Troubleshooting guide",
            "✓ Architecture documentation",
            "✓ Runbooks",
            "✓ Change log",
            "✓ Incident procedures",
            "✓ User guides",
        ]
    }

    @classmethod
    def validate_deployment(cls, chute_instance) -> Dict[str, Any]:
        """Validate deployment readiness."""

        results = {
            "ready_for_production": True,
            "missing_items": [],
            "warnings": [],
            "category_scores": {}
        }

        # Check each category
        for category, items in cls.CHECKLIST.items():
            score = cls._check_category(chute_instance, category, items)
            results["category_scores"][category] = score

            if score < 0.8:  # 80% threshold
                results["ready_for_production"] = False
                results["missing_items"].extend([
                    f"{category}: {item}" for item in items
                    if not cls._check_item(chute_instance, item)
                ])

        return results

    @classmethod
    def _check_category(cls, chute_instance, category: str, items: list) -> float:
        """Check completion percentage for a category."""
        completed = sum(1 for item in items if cls._check_item(chute_instance, item))
        return completed / len(items)

    @classmethod
    def _check_item(cls, chute_instance, item: str) -> bool:
        """Check if a specific item is implemented."""

        # This would implement actual checks for each item
        # For example:
        if "health checks" in item.lower():
            return hasattr(chute_instance, 'health_checker')
        elif "error handling" in item.lower():
            return hasattr(chute_instance, 'error_handler')
        elif "metrics" in item.lower():
            return hasattr(chute_instance, 'performance_monitor')

        # Default to False for demonstration
        return False

class ProductionConfigValidator:
    """Validate production configuration."""

    @staticmethod
    def validate_config(config: ApplicationConfig) -> Dict[str, Any]:
        """Validate configuration for production."""

        issues = []
        warnings = []

        # Security checks
        if not config.api_key_required:
            issues.append("API key authentication should be required in production")

        if config.debug:
            issues.append("Debug mode should be disabled in production")

        if config.log_level == "DEBUG":
            warnings.append("DEBUG log level may impact performance in production")

        # Performance checks
        if config.max_workers < 2:
            warnings.append("Consider using more workers for better performance")

        if config.request_timeout > 60:
            warnings.append("Long request timeouts may cause resource exhaustion")

        # Resource checks
        if config.max_memory_usage > 0.95:
            warnings.append("Very high memory usage limit may cause instability")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }
```

## Summary and Next Steps

This comprehensive best practices guide covers:

1. **Architecture Patterns**: Modular design, dependency injection, configuration management
2. **Performance**: Resource management, caching strategies, load balancing
3. **Security**: Authentication, input validation, content safety
4. **Monitoring**: Structured logging, metrics collection, observability
5. **Testing**: Unit tests, integration tests, security tests, load tests
6. **Deployment**: Production readiness checklist and validation

### Implementation Priority

1. **Start with Security**: Implement authentication and input validation
2. **Add Monitoring**: Set up logging and basic metrics
3. **Optimize Performance**: Add caching and resource management
4. **Comprehensive Testing**: Build a complete test suite
5. **Production Hardening**: Follow deployment checklist

### Continuous Improvement

- **Regular Security Audits**: Update security practices regularly
- **Performance Monitoring**: Continuously optimize based on metrics
- **User Feedback**: Incorporate user feedback into improvements
- **Technology Updates**: Keep dependencies and frameworks updated
- **Team Training**: Ensure team knows best practices

For more specific guides, see:

- [Error Handling Guide](error-handling)
- [Custom Images Guide](custom-images)
- [Streaming Guide](streaming)
- [Templates Guide](templates)
