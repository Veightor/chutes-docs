# Performance Optimization Guide

This guide focuses on specific performance optimization techniques for maximizing the efficiency of your Chutes applications.

## Overview

Performance optimization covers:

- **Inference Optimization**: Speed up model inference
- **Memory Management**: Efficient use of GPU and system memory
- **Throughput Maximization**: Handle more requests per second
- **Latency Reduction**: Minimize response times
- **Resource Utilization**: Get the most from your hardware

## Model Inference Optimization

### Dynamic Batching

Implement dynamic batching for better GPU utilization:

```python
import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import torch

@dataclass
class BatchRequest:
    data: Dict[str, Any]
    future: asyncio.Future
    timestamp: float

class DynamicBatcher:
    def __init__(self,
                 max_batch_size: int = 32,
                 max_wait_time: float = 0.01,  # 10ms
                 model_inference_fn=None):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.model_inference_fn = model_inference_fn

        self.pending_requests: List[BatchRequest] = []
        self.processing = False
        self.lock = asyncio.Lock()

    async def add_request(self, data: Dict[str, Any]) -> Any:
        """Add request to batch queue"""
        future = asyncio.Future()
        request = BatchRequest(
            data=data,
            future=future,
            timestamp=time.time()
        )

        async with self.lock:
            self.pending_requests.append(request)

            # Start processing if not already running
            if not self.processing:
                asyncio.create_task(self._process_batch())

        return await future

    async def _process_batch(self):
        """Process accumulated requests"""
        async with self.lock:
            if self.processing or not self.pending_requests:
                return
            self.processing = True

        while True:
            # Wait for batch to accumulate or timeout
            start_time = time.time()

            while (len(self.pending_requests) < self.max_batch_size and
                   time.time() - start_time < self.max_wait_time):
                await asyncio.sleep(0.001)  # 1ms

            async with self.lock:
                if not self.pending_requests:
                    break

                # Extract batch
                batch_size = min(len(self.pending_requests), self.max_batch_size)
                batch = self.pending_requests[:batch_size]
                self.pending_requests = self.pending_requests[batch_size:]

            # Process batch
            try:
                batch_data = [req.data for req in batch]
                results = await self._run_inference_batch(batch_data)

                # Return results
                for req, result in zip(batch, results):
                    if not req.future.done():
                        req.future.set_result(result)

            except Exception as e:
                # Handle batch failure
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(e)

        async with self.lock:
            self.processing = False

    async def _run_inference_batch(self, batch_data: List[Dict[str, Any]]) -> List[Any]:
        """Run inference on batch"""
        if self.model_inference_fn:
            return await self.model_inference_fn(batch_data)

        # Default implementation
        return [{"result": f"processed_{i}"} for i in range(len(batch_data))]

# Usage
async def model_inference_batch(batch_data: List[Dict[str, Any]]) -> List[Any]:
    """Your model inference function"""
    # Extract texts
    texts = [item["text"] for item in batch_data]

    # Tokenize batch
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Return results
    results = []
    for i, pred in enumerate(predictions):
        results.append({
            "text": texts[i],
            "prediction": pred.cpu().numpy().tolist(),
            "confidence": float(torch.max(pred))
        })

    return results

# Global batcher
batcher = DynamicBatcher(
    max_batch_size=16,
    max_wait_time=0.01,
    model_inference_fn=model_inference_batch
)

async def run_optimized(inputs: Dict[str, Any]) -> Any:
    """Optimized endpoint using dynamic batching"""
    result = await batcher.add_request(inputs)
    return result
```

### Model Quantization and Optimization

Optimize models for faster inference:

```python
import torch
from transformers import AutoModel, AutoTokenizer
from torch.jit import script
import torch.ao.quantization as quantization

class OptimizedModel:
    def __init__(self, model_name: str, optimization_level: str = "basic"):
        self.model_name = model_name
        self.optimization_level = optimization_level

        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Apply optimizations
        self._optimize_model()

    def _optimize_model(self):
        """Apply various optimizations"""
        self.model.eval()

        if self.optimization_level == "basic":
            self._basic_optimization()
        elif self.optimization_level == "quantized":
            self._quantization_optimization()
        elif self.optimization_level == "compiled":
            self._compilation_optimization()
        elif self.optimization_level == "all":
            self._all_optimizations()

    def _basic_optimization(self):
        """Basic optimizations"""
        # Enable inference mode
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def _quantization_optimization(self):
        """Apply quantization"""
        # Post-training quantization
        self.model = quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

    def _compilation_optimization(self):
        """Apply TorchScript compilation"""
        # Example input for tracing
        example_input = self.tokenizer(
            "Hello world",
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Trace model
        with torch.no_grad():
            self.model = torch.jit.trace(self.model, (example_input))

    def _all_optimizations(self):
        """Apply all optimizations"""
        self._basic_optimization()
        # Note: Quantization and compilation may conflict
        self._quantization_optimization()

    @torch.no_grad()
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Optimized prediction"""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Inference
        outputs = self.model(**inputs)

        # Process results
        embeddings = outputs.last_hidden_state.mean(dim=1)

        results = []
        for i, embedding in enumerate(embeddings):
            results.append({
                "text": texts[i],
                "embedding": embedding.cpu().numpy().tolist()
            })

        return results

# Global optimized model
optimized_model = OptimizedModel("bert-base-uncased", optimization_level="all")
```

## Memory Optimization

### GPU Memory Management

Efficient GPU memory usage:

```python
import torch
import gc
from contextlib import contextmanager
from typing import Iterator

class GPUMemoryManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_threshold = 0.85  # 85% memory usage threshold

    @contextmanager
    def memory_efficient_context(self) -> Iterator[None]:
        """Context manager for memory-efficient operations"""
        initial_memory = self.get_memory_usage()

        try:
            yield
        finally:
            # Clean up after operation
            self.cleanup_memory()

            final_memory = self.get_memory_usage()
            memory_saved = initial_memory - final_memory

            if memory_saved > 0:
                print(f"Memory cleaned up: {memory_saved:.2f}%")

    def get_memory_usage(self) -> float:
        """Get current GPU memory usage percentage"""
        if not torch.cuda.is_available():
            return 0.0

        memory_used = torch.cuda.memory_allocated()
        memory_total = torch.cuda.get_device_properties(0).total_memory
        return (memory_used / memory_total) * 100

    def cleanup_memory(self):
        """Aggressive memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def check_memory_pressure(self) -> bool:
        """Check if we're under memory pressure"""
        return self.get_memory_usage() > self.memory_threshold * 100

    def adaptive_batch_size(self, base_batch_size: int, max_batch_size: int) -> int:
        """Adapt batch size based on memory pressure"""
        memory_usage = self.get_memory_usage()

        if memory_usage > 80:
            return max(1, base_batch_size // 4)
        elif memory_usage > 60:
            return max(1, base_batch_size // 2)
        elif memory_usage < 40:
            return min(max_batch_size, base_batch_size * 2)
        else:
            return base_batch_size

# Global memory manager
memory_manager = GPUMemoryManager()

class MemoryOptimizedProcessor:
    def __init__(self, model):
        self.model = model
        self.base_batch_size = 16
        self.max_batch_size = 64

    async def process_with_memory_management(self, data_list: List[Any]) -> List[Any]:
        """Process data with dynamic memory management"""
        results = []

        i = 0
        while i < len(data_list):
            # Adapt batch size based on memory pressure
            current_batch_size = memory_manager.adaptive_batch_size(
                self.base_batch_size,
                self.max_batch_size
            )

            batch = data_list[i:i + current_batch_size]

            with memory_manager.memory_efficient_context():
                batch_results = await self._process_batch(batch)
                results.extend(batch_results)

            i += current_batch_size

            # Yield control to allow other operations
            await asyncio.sleep(0)

        return results

    async def _process_batch(self, batch: List[Any]) -> List[Any]:
        """Process a single batch"""
        with torch.no_grad():
            # Your processing logic here
            return [{"processed": item} for item in batch]
```

## Caching and Memoization

### Advanced Caching Strategies

Implement intelligent caching:

```python
import hashlib
import pickle
import time
from typing import Any, Optional, Dict, Callable
import redis
import asyncio
from functools import wraps

class AdvancedCache:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.local_cache: Dict[str, tuple] = {}  # (value, timestamp, access_count)
        self.local_cache_size = 1000
        self.default_ttl = 3600

        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.local_hits = 0
        self.redis_hits = 0

    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key"""
        key_data = f"{func_name}:{args}:{sorted(kwargs.items())}"
        return f"cache:{hashlib.md5(key_data.encode()).hexdigest()}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (local first, then Redis)"""
        # Try local cache first
        if key in self.local_cache:
            value, timestamp, access_count = self.local_cache[key]
            if time.time() - timestamp < self.default_ttl:
                # Update access count and timestamp
                self.local_cache[key] = (value, timestamp, access_count + 1)
                self.hits += 1
                self.local_hits += 1
                return value
            else:
                # Expired
                del self.local_cache[key]

        # Try Redis cache
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                value = pickle.loads(cached_data)

                # Store in local cache
                self._store_local(key, value)

                self.hits += 1
                self.redis_hits += 1
                return value
        except Exception:
            pass

        self.misses += 1
        return None

    async def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set value in cache"""
        ttl = ttl or self.default_ttl

        # Store in local cache
        self._store_local(key, value)

        # Store in Redis
        try:
            serialized_value = pickle.dumps(value)
            self.redis_client.setex(key, ttl, serialized_value)
        except Exception:
            pass

    def _store_local(self, key: str, value: Any) -> None:
        """Store value in local cache with LRU eviction"""
        current_time = time.time()

        # Evict if at capacity
        if len(self.local_cache) >= self.local_cache_size:
            # Remove least recently used item
            lru_key = min(
                self.local_cache.keys(),
                key=lambda k: self.local_cache[k][1]  # timestamp
            )
            del self.local_cache[lru_key]

        self.local_cache[key] = (value, current_time, 1)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "local_hits": self.local_hits,
            "redis_hits": self.redis_hits,
            "misses": self.misses,
            "local_cache_size": len(self.local_cache)
        }

    def cached(self, ttl: int = None):
        """Decorator for caching function results"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_key(func.__name__, args, kwargs)

                # Try to get cached result
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # Execute function
                result = await func(*args, **kwargs)

                # Cache result
                await self.set(cache_key, result, ttl)

                return result

            return wrapper
        return decorator

# Global cache instance
advanced_cache = AdvancedCache()

@advanced_cache.cached(ttl=1800)  # 30 minutes
async def expensive_model_inference(text: str, model_params: dict) -> dict:
    """Expensive computation that benefits from caching"""
    # Simulate expensive operation
    await asyncio.sleep(1)
    return {"result": f"processed_{text}", "params": model_params}
```

## Concurrent Processing

### Async Processing Patterns

Optimize concurrent request handling:

```python
import asyncio
from typing import List, Dict, Any, Callable
import aiohttp
from asyncio import Semaphore, Queue
from dataclasses import dataclass

@dataclass
class ProcessingTask:
    id: str
    data: Dict[str, Any]
    priority: int = 0

class ConcurrentProcessor:
    def __init__(self,
                 max_concurrent: int = 10,
                 max_queue_size: int = 1000):
        self.max_concurrent = max_concurrent
        self.semaphore = Semaphore(max_concurrent)
        self.task_queue: Queue = Queue(maxsize=max_queue_size)
        self.workers_running = False

        # Performance metrics
        self.processed_count = 0
        self.failed_count = 0
        self.total_processing_time = 0.0

    async def start_workers(self, num_workers: int = 5):
        """Start background workers"""
        self.workers_running = True
        workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(num_workers)
        ]
        return workers

    async def stop_workers(self):
        """Stop background workers"""
        self.workers_running = False

    async def submit_task(self, task: ProcessingTask) -> Any:
        """Submit task for processing"""
        # Add to queue
        await self.task_queue.put(task)

        # Wait for processing (in real implementation, use futures)
        # This is simplified for example
        return {"task_id": task.id, "status": "queued"}

    async def _worker(self, worker_id: str):
        """Background worker"""
        while self.workers_running:
            try:
                # Get task from queue
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )

                # Process task with concurrency control
                async with self.semaphore:
                    await self._process_task(task)

                # Mark task as done
                self.task_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                self.failed_count += 1

    async def _process_task(self, task: ProcessingTask):
        """Process individual task"""
        start_time = time.time()

        try:
            # Your processing logic here
            await self._simulate_processing(task.data)

            self.processed_count += 1
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time

        except Exception as e:
            self.failed_count += 1
            raise e

    async def _simulate_processing(self, data: Dict[str, Any]):
        """Simulate processing work"""
        # Replace with actual processing
        await asyncio.sleep(0.1)

    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        avg_processing_time = (
            self.total_processing_time / self.processed_count
            if self.processed_count > 0 else 0
        )

        return {
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
            "average_processing_time": avg_processing_time,
            "queue_size": self.task_queue.qsize(),
            "success_rate": (
                self.processed_count / (self.processed_count + self.failed_count)
                if (self.processed_count + self.failed_count) > 0 else 0
            )
        }

# Global processor
concurrent_processor = ConcurrentProcessor(max_concurrent=20)

async def run_concurrent_processing(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Handle requests with concurrent processing"""

    # Create processing task
    task = ProcessingTask(
        id=inputs.get("request_id", "unknown"),
        data=inputs,
        priority=inputs.get("priority", 0)
    )

    # Submit for processing
    result = await concurrent_processor.submit_task(task)

    # Return result with metrics
    return {
        "result": result,
        "metrics": concurrent_processor.get_metrics()
    }

# Initialize workers on startup
async def initialize_concurrent_processing():
    """Initialize concurrent processing system"""
    workers = await concurrent_processor.start_workers(num_workers=10)
    print(f"Started {len(workers)} concurrent workers")
    return workers
```

## Performance Monitoring

### Real-time Performance Metrics

Monitor performance in real-time:

```python
import time
import psutil
import asyncio
from collections import deque
from typing import Dict, Any, List
from dataclasses import dataclass
from prometheus_client import Counter, Histogram, Gauge, start_http_server

@dataclass
class PerformanceMetric:
    timestamp: float
    request_count: int
    avg_response_time: float
    cpu_usage: float
    memory_usage: float
    gpu_utilization: float

class PerformanceMonitor:
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)

        # Prometheus metrics
        self.request_counter = Counter('requests_total', 'Total requests')
        self.response_time_histogram = Histogram('response_time_seconds', 'Response time')
        self.cpu_gauge = Gauge('cpu_usage_percent', 'CPU usage')
        self.memory_gauge = Gauge('memory_usage_percent', 'Memory usage')
        self.gpu_gauge = Gauge('gpu_utilization_percent', 'GPU utilization')

        # Performance tracking
        self.request_times = deque(maxlen=1000)
        self.start_time = time.time()

        # Start metrics server
        start_http_server(8001)

    def record_request(self, response_time: float):
        """Record a request"""
        self.request_counter.inc()
        self.response_time_histogram.observe(response_time)
        self.request_times.append(response_time)

    def update_system_metrics(self):
        """Update system performance metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        self.cpu_gauge.set(cpu_percent)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        self.memory_gauge.set(memory_percent)

        # GPU utilization (if available)
        gpu_util = self._get_gpu_utilization()
        self.gpu_gauge.set(gpu_util)

        # Create metric snapshot
        current_metric = PerformanceMetric(
            timestamp=time.time(),
            request_count=len(self.request_times),
            avg_response_time=sum(self.request_times) / len(self.request_times) if self.request_times else 0,
            cpu_usage=cpu_percent,
            memory_usage=memory_percent,
            gpu_utilization=gpu_util
        )

        self.metrics_history.append(current_metric)

    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.utilization()
        except:
            pass
        return 0.0

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics_history:
            return {"status": "no_data"}

        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 metrics

        return {
            "uptime_seconds": time.time() - self.start_time,
            "total_requests": self.request_counter._value._value,
            "current_rps": len(self.request_times),
            "avg_response_time": sum(m.avg_response_time for m in recent_metrics) / len(recent_metrics),
            "cpu_usage": recent_metrics[-1].cpu_usage,
            "memory_usage": recent_metrics[-1].memory_usage,
            "gpu_utilization": recent_metrics[-1].gpu_utilization,
            "performance_trend": self._calculate_performance_trend()
        }

    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend"""
        if len(self.metrics_history) < 10:
            return "insufficient_data"

        recent = list(self.metrics_history)[-5:]
        older = list(self.metrics_history)[-10:-5]

        recent_avg_response = sum(m.avg_response_time for m in recent) / len(recent)
        older_avg_response = sum(m.avg_response_time for m in older) / len(older)

        if recent_avg_response > older_avg_response * 1.1:
            return "degrading"
        elif recent_avg_response < older_avg_response * 0.9:
            return "improving"
        else:
            return "stable"

    def track_request(self, func):
        """Decorator to track request performance"""
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                response_time = time.time() - start_time
                self.record_request(response_time)
                self.update_system_metrics()

        return wrapper

# Global performance monitor
performance_monitor = PerformanceMonitor()

# Auto-update metrics
async def metrics_updater():
    """Background task to update metrics"""
    while True:
        performance_monitor.update_system_metrics()
        await asyncio.sleep(10)  # Update every 10 seconds

# Start metrics updater
asyncio.create_task(metrics_updater())

@performance_monitor.track_request
async def run_with_monitoring(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Run with performance monitoring"""
    # Your processing logic
    result = await process_request(inputs)

    # Include performance summary
    return {
        "result": result,
        "performance": performance_monitor.get_performance_summary()
    }
```

## Next Steps

- **[Performance Guide](performance)** - General performance optimization
- **[Cost Optimization](cost-optimization)** - Balance performance and cost
- **[Best Practices](best-practices)** - Deployment best practices

For advanced performance tuning, see the [Advanced Performance Guide](../advanced/performance-tuning).
