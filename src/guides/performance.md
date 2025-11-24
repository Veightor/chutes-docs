# Performance Optimization Guide

This comprehensive guide covers performance optimization strategies for Chutes applications, from basic optimizations to advanced scaling techniques.

## Overview

Performance optimization in Chutes involves several key areas:

- **Model Optimization**: Optimizing AI model inference speed and memory usage
- **Resource Management**: Efficient use of GPU, CPU, and memory resources
- **Scaling Strategies**: Horizontal and vertical scaling approaches
- **Caching**: Implementing effective caching strategies
- **Network Optimization**: Reducing latency and improving throughput

## Model Performance Optimization

### Model Quantization

Reduce model size and increase inference speed:

```python
from chutes.image import Image
from chutes.chute import Chute, NodeSelector

# Quantized model image
quantized_image = (
    Image(
        username="myuser",
        name="quantized-model",
        tag="1.0.0",
        python_version="3.11"
    )
    .pip_install([
        "torch==2.1.0",
        "transformers==4.35.0",
        "optimum[onnxruntime]==1.14.0",
        "bitsandbytes==0.41.0"
    ])
)

# Example: 8-bit quantization
def load_quantized_model():
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/DialoGPT-medium",
        quantization_config=quantization_config,
        device_map="auto"
    )

    return model

async def run_quantized_inference(inputs):
    model = load_quantized_model()
    # Inference logic here
    return {"result": "optimized inference"}
```

### Model Compilation with TorchScript

Optimize model execution:

```python
import torch
from transformers import AutoModel, AutoTokenizer

class OptimizedModel:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Compile model for better performance
        self.model = torch.jit.script(self.model)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs

# Use in chute
optimized_model = OptimizedModel("bert-base-uncased")

async def run_optimized(inputs):
    result = optimized_model.predict(inputs["text"])
    return {"embeddings": result.last_hidden_state.mean(dim=1).tolist()}
```

### Batch Processing

Process multiple inputs efficiently:

```python
from typing import List
import torch

class BatchProcessor:
    def __init__(self, model, tokenizer, max_batch_size: int = 32):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size

    async def process_batch(self, texts: List[str]):
        results = []

        # Process in batches
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )

            # Process batch
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Extract results
            batch_results = outputs.last_hidden_state.mean(dim=1)
            results.extend(batch_results.tolist())

        return results

async def run_batch_processing(inputs):
    processor = BatchProcessor(model, tokenizer)
    results = await processor.process_batch(inputs["texts"])
    return {"embeddings": results}
```

## Resource Optimization

### GPU Memory Management

Optimize GPU memory usage:

```python
import torch
import gc

class MemoryOptimizedModel:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model_on_demand(self, model_name: str):
        """Load model only when needed"""
        if self.model is None:
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

    def clear_cache(self):
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def predict(self, inputs):
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        result = self.model(**inputs)

        # Clear intermediate tensors
        del inputs
        self.clear_cache()

        return result

# Configure for memory optimization
torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
```

### CPU Optimization

Optimize CPU-bound operations:

```python
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

class CPUOptimizedProcessor:
    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or mp.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.num_workers)

    async def process_cpu_intensive(self, data_list):
        """Process CPU-intensive tasks in parallel"""
        loop = asyncio.get_event_loop()

        # Use process pool for CPU-bound tasks
        tasks = [
            loop.run_in_executor(self.process_pool, self.cpu_task, data)
            for data in data_list
        ]

        results = await asyncio.gather(*tasks)
        return results

    async def process_io_bound(self, urls):
        """Process I/O-bound tasks in parallel"""
        loop = asyncio.get_event_loop()

        # Use thread pool for I/O-bound tasks
        tasks = [
            loop.run_in_executor(self.thread_pool, self.io_task, url)
            for url in urls
        ]

        results = await asyncio.gather(*tasks)
        return results

    def cpu_task(self, data):
        # CPU-intensive processing
        return data ** 2

    def io_task(self, url):
        # I/O operation
        import requests
        response = requests.get(url)
        return response.status_code
```

## Caching Strategies

### Redis Caching

Implement efficient caching:

```python
import redis
import pickle
import hashlib
from typing import Optional, Any
import asyncio

class CacheManager:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.default_ttl = 3600  # 1 hour

    def generate_cache_key(self, prefix: str, *args) -> str:
        """Generate deterministic cache key"""
        key_data = str(args)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}:{key_hash}"

    async def get_cached(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            print(f"Cache read error: {e}")
        return None

    async def set_cached(self, key: str, value: Any, ttl: int = None):
        """Set value in cache"""
        try:
            ttl = ttl or self.default_ttl
            serialized_data = pickle.dumps(value)
            self.redis_client.setex(key, ttl, serialized_data)
        except Exception as e:
            print(f"Cache write error: {e}")

    async def cached_function(self, func, cache_prefix: str, ttl: int = None):
        """Decorator for caching function results"""
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self.generate_cache_key(cache_prefix, args, kwargs)

            # Try to get from cache
            cached_result = await self.get_cached(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            await self.set_cached(cache_key, result, ttl)

            return result

        return wrapper

# Usage in chute
cache_manager = CacheManager()

@cache_manager.cached_function("model_inference", ttl=1800)
async def cached_inference(text: str):
    # Expensive model inference
    result = model.predict(text)
    return result
```

### In-Memory Caching

Implement local caching:

```python
from functools import lru_cache
import time
from typing import Dict, Any, Tuple

class MemoryCache:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.max_size = max_size
        self.ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                # Expired
                del self.cache[key]
        return None

    def set(self, key: str, value: Any):
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(),
                           key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        self.cache[key] = (value, time.time())

    def cached_method(self, func):
        """Decorator for caching method results"""
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

            # Try cache first
            cached_result = self.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute and cache
            result = func(*args, **kwargs)
            self.set(cache_key, result)
            return result

        return wrapper

# Global cache instance
memory_cache = MemoryCache(max_size=500, ttl=1800)

@memory_cache.cached_method
def expensive_computation(data: str) -> str:
    # Simulate expensive operation
    time.sleep(1)
    return data.upper()
```

## Scaling Strategies

### Auto-scaling Configuration

Configure automatic scaling:

```python
from chutes.chute import Chute, NodeSelector

# Auto-scaling chute configuration
high_performance_chute = Chute(
    username="myuser",
    name="high-performance-service",
    image=optimized_image,
    entry_file="optimized_service.py",
    entry_point="run",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16preferred_provider="runpod"
    ),
    timeout_seconds=300,
    concurrency=20,

    # Auto-scaling settings
    auto_scale=True,
    min_instances=2,
    max_instances=10,
    scale_up_threshold=0.8,    # Scale up when 80% utilized
    scale_down_threshold=0.3,  # Scale down when <30% utilized
    scale_up_cooldown=300,     # Wait 5 min before scaling up again
    scale_down_cooldown=600    # Wait 10 min before scaling down
)
```

### Load Balancing

Implement custom load balancing:

```python
import random
import time
from typing import List, Dict

class LoadBalancer:
    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
        self.health_status = {endpoint: True for endpoint in endpoints}
        self.response_times = {endpoint: 0.0 for endpoint in endpoints}
        self.request_counts = {endpoint: 0 for endpoint in endpoints}

    def get_endpoint(self, strategy: str = "round_robin") -> str:
        """Get next endpoint based on strategy"""
        healthy_endpoints = [
            ep for ep in self.endpoints
            if self.health_status[ep]
        ]

        if not healthy_endpoints:
            raise Exception("No healthy endpoints available")

        if strategy == "round_robin":
            return self._round_robin(healthy_endpoints)
        elif strategy == "least_connections":
            return self._least_connections(healthy_endpoints)
        elif strategy == "fastest_response":
            return self._fastest_response(healthy_endpoints)
        else:
            return random.choice(healthy_endpoints)

    def _round_robin(self, endpoints: List[str]) -> str:
        # Simple round-robin implementation
        min_requests = min(self.request_counts[ep] for ep in endpoints)
        candidates = [ep for ep in endpoints
                     if self.request_counts[ep] == min_requests]
        return candidates[0]

    def _least_connections(self, endpoints: List[str]) -> str:
        return min(endpoints, key=lambda ep: self.request_counts[ep])

    def _fastest_response(self, endpoints: List[str]) -> str:
        return min(endpoints, key=lambda ep: self.response_times[ep])

    def record_request(self, endpoint: str, response_time: float):
        """Record request metrics"""
        self.request_counts[endpoint] += 1
        # Exponential moving average
        self.response_times[endpoint] = (
            0.7 * self.response_times[endpoint] +
            0.3 * response_time
        )

    async def health_check(self):
        """Periodic health check"""
        import httpx

        async with httpx.AsyncClient() as client:
            for endpoint in self.endpoints:
                try:
                    response = await client.get(f"{endpoint}/health", timeout=5.0)
                    self.health_status[endpoint] = response.status_code == 200
                except:
                    self.health_status[endpoint] = False

# Usage with multiple chute instances
load_balancer = LoadBalancer([
    "https://chute1.example.com",
    "https://chute2.example.com",
    "https://chute3.example.com"
])

async def run_with_load_balancing(inputs):
    endpoint = load_balancer.get_endpoint("fastest_response")

    start_time = time.time()

    try:
        # Make request to selected endpoint
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{endpoint}/run", json=inputs)
            result = response.json()

        # Record success
        response_time = time.time() - start_time
        load_balancer.record_request(endpoint, response_time)

        return result

    except Exception as e:
        # Mark endpoint as unhealthy on failure
        load_balancer.health_status[endpoint] = False
        raise e
```

## Network Optimization

### Response Compression

Reduce payload sizes:

```python
import gzip
import json
from typing import Any

class ResponseOptimizer:
    @staticmethod
    def compress_response(data: Any, compression_threshold: int = 1024) -> dict:
        """Compress large responses"""
        json_data = json.dumps(data)

        if len(json_data) > compression_threshold:
            compressed_data = gzip.compress(json_data.encode())

            return {
                "compressed": True,
                "data": compressed_data.hex(),
                "original_size": len(json_data),
                "compressed_size": len(compressed_data)
            }
        else:
            return {
                "compressed": False,
                "data": data,
                "size": len(json_data)
            }

    @staticmethod
    def decompress_response(response: dict) -> Any:
        """Decompress response if needed"""
        if response.get("compressed", False):
            compressed_data = bytes.fromhex(response["data"])
            json_data = gzip.decompress(compressed_data).decode()
            return json.loads(json_data)
        else:
            return response["data"]

# Use in chute
async def run_optimized_response(inputs):
    # Process inputs
    result = await process_large_data(inputs)

    # Optimize response
    optimized_response = ResponseOptimizer.compress_response(result)

    return optimized_response
```

### Streaming Responses

Stream large responses:

```python
import asyncio
from typing import AsyncGenerator

async def stream_large_results(inputs) -> AsyncGenerator[dict, None]:
    """Stream results as they become available"""

    # Process data in chunks
    data_chunks = split_data_into_chunks(inputs["data"])

    for i, chunk in enumerate(data_chunks):
        # Process chunk
        result = await process_chunk(chunk)

        # Yield partial result
        yield {
            "chunk_id": i,
            "total_chunks": len(data_chunks),
            "result": result,
            "is_final": i == len(data_chunks) - 1
        }

        # Allow other tasks to run
        await asyncio.sleep(0)

async def run_streaming(inputs):
    """Handle streaming request"""
    if inputs.get("stream", False):
        # Return async generator for streaming
        return stream_large_results(inputs)
    else:
        # Return complete result
        return await process_all_data(inputs)
```

## Monitoring and Profiling

### Performance Metrics

Track performance metrics:

```python
import time
import psutil
import torch
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests')
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage')
GPU_UTILIZATION = Gauge('gpu_utilization_percent', 'GPU utilization')

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()

        # Start metrics server
        start_http_server(8001)

    def track_request(self, func):
        """Decorator to track request metrics"""
        async def wrapper(*args, **kwargs):
            REQUEST_COUNT.inc()
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                REQUEST_DURATION.observe(duration)

                # Update system metrics
                self.update_system_metrics()

        return wrapper

    def update_system_metrics(self):
        """Update system performance metrics"""
        # Memory usage
        memory_info = psutil.virtual_memory()
        MEMORY_USAGE.set(memory_info.used)

        # GPU utilization
        if torch.cuda.is_available():
            gpu_util = torch.cuda.utilization()
            GPU_UTILIZATION.set(gpu_util)

# Global monitor
monitor = PerformanceMonitor()

@monitor.track_request
async def run_monitored(inputs):
    # Your processing logic
    result = await process_inputs(inputs)
    return result
```

### Profiling Tools

Profile code performance:

```python
import cProfile
import pstats
import io
from contextlib import contextmanager

@contextmanager
def profile_code():
    """Context manager for profiling code blocks"""
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        yield profiler
    finally:
        profiler.disable()

        # Get profile stats
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions

        print("Performance Profile:")
        print(stream.getvalue())

# Usage
async def run_with_profiling(inputs):
    with profile_code():
        result = await expensive_operation(inputs)

    return result

# Memory profiling
from memory_profiler import profile

@profile
def memory_intensive_function(data):
    # Function that uses a lot of memory
    large_list = [data] * 1000000
    processed = [item.upper() for item in large_list]
    return processed[:10]
```

## Best Practices Summary

### Model Optimization

- Use quantization for faster inference
- Implement model compilation where possible
- Process inputs in batches
- Cache model outputs for repeated inputs

### Resource Management

- Monitor GPU memory usage
- Implement proper cleanup
- Use appropriate data types
- Optimize for your specific hardware

### Scaling

- Configure auto-scaling based on your traffic patterns
- Implement health checks
- Use load balancing for high availability
- Monitor performance metrics

### Caching

- Cache expensive computations
- Use appropriate TTL values
- Implement cache invalidation strategies
- Monitor cache hit rates

## Next Steps

- **[Best Practices](best-practices)** - General deployment best practices
- **[Monitoring Guide](../monitoring)** - Advanced monitoring strategies
- **[Cost Optimization](cost-optimization)** - Optimize costs while maintaining performance
- **[Scaling Guide](../scaling)** - Advanced scaling strategies

For enterprise performance optimization, see the [Enterprise Performance Guide](../enterprise/performance).
