# Cost Optimization Guide

This guide provides strategies to optimize costs while maintaining performance and reliability for your Chutes applications.

## Overview

Cost optimization in Chutes involves:

- **Resource Right-sizing**: Choose appropriate hardware configurations
- **Auto-scaling**: Scale resources based on demand
- **Spot Instances**: Use cost-effective computing options
- **Efficient Scheduling**: Optimize when workloads run
- **Model Optimization**: Reduce computational requirements

## Resource Right-sizing

### Choose Appropriate Hardware

Select the right GPU and memory configuration:

```python
from chutes.chute import Chute, NodeSelector

# Cost-optimized for inference
inference_chute = Chute(
    username="myuser",
    name="cost-optimized-inference",
    image=your_image,
    entry_file="app.py",
    entry_point="run",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=8,    # Right-size for your model# Minimal RAM requirements
        preferred_provider="vast"  # Often more cost-effective
    ),
    timeout_seconds=300,
    concurrency=8
)

# For batch processing with higher throughput needs
batch_chute = Chute(
    username="myuser",
    name="batch-processing",
    image=your_image,
    entry_file="batch_app.py",
    entry_point="run",
    node_selector=NodeSelector(
        gpu_count=2,
        min_vram_gb_per_gpu=24),
    timeout_seconds=1800,
    concurrency=4
)
```

## Spot Instance Strategy

### Using Spot Instances

Leverage spot instances for significant cost savings:

```python
from chutes.chute import Chute, NodeSelector

# Spot instance configuration
spot_chute = Chute(
    username="myuser",
    name="spot-training",
    image=training_image,
    entry_file="training.py",
    entry_point="run",
    node_selector=NodeSelector(
        gpu_count=4,
        min_vram_gb_per_gpu=16max_spot_price=0.50  # Set maximum price you're willing to pay
    ),
    timeout_seconds=7200,  # Longer timeout for training
    concurrency=1,
    auto_scale=False
)

# Fault-tolerant batch processing with spot instances
class SpotInstanceManager:
    def __init__(self, chute_config):
        self.chute_config = chute_config
        self.retry_count = 3

    async def run_with_retry(self, inputs):
        """Run job with automatic retry on spot interruption"""
        for attempt in range(self.retry_count):
            try:
                # Create chute with spot instance
                chute = Chute(**self.chute_config)
                result = chute.run(inputs)
                return result

            except Exception as e:
                if attempt == self.retry_count - 1:
                    # All retries exhausted
                    raise e

                # Wait before retry
                await asyncio.sleep(30)

        raise Exception("Failed after all retry attempts")
```

## Smart Scaling Strategies

### Time-based Scaling

Scale based on predictable usage patterns:

```python
import schedule
import time
from datetime import datetime

class TimeBasedScaler:
    def __init__(self, chute_name):
        self.chute_name = chute_name
        self.setup_schedule()

    def setup_schedule(self):
        """Set up scaling schedule based on usage patterns"""
        # Scale up during business hours
        schedule.every().monday.at("08:00").do(self.scale_up)
        schedule.every().tuesday.at("08:00").do(self.scale_up)
        schedule.every().wednesday.at("08:00").do(self.scale_up)
        schedule.every().thursday.at("08:00").do(self.scale_up)
        schedule.every().friday.at("08:00").do(self.scale_up)

        # Scale down after hours
        schedule.every().monday.at("18:00").do(self.scale_down)
        schedule.every().tuesday.at("18:00").do(self.scale_down)
        schedule.every().wednesday.at("18:00").do(self.scale_down)
        schedule.every().thursday.at("18:00").do(self.scale_down)
        schedule.every().friday.at("18:00").do(self.scale_down)

        # Minimal scaling on weekends
        schedule.every().saturday.at("00:00").do(self.scale_minimal)
        schedule.every().sunday.at("00:00").do(self.scale_minimal)

    def scale_up(self):
        """Scale up for peak hours"""
        self.update_chute_config({
            "min_instances": 3,
            "max_instances": 10,
            "concurrency": 20
        })

    def scale_down(self):
        """Scale down for off-peak hours"""
        self.update_chute_config({
            "min_instances": 1,
            "max_instances": 3,
            "concurrency": 8
        })

    def scale_minimal(self):
        """Minimal scaling for weekends"""
        self.update_chute_config({
            "min_instances": 0,
            "max_instances": 2,
            "concurrency": 4
        })

    def update_chute_config(self, config):
        """Update chute configuration"""
        # Implementation to update chute scaling settings
        pass

    def run(self):
        """Run the scheduler"""
        while True:
            schedule.run_pending()
            time.sleep(60)
```

### Demand-based Auto-scaling

Implement intelligent auto-scaling:

```python
class DemandBasedScaler:
    def __init__(self, chute, target_utilization=0.7):
        self.chute = chute
        self.target_utilization = target_utilization
        self.metrics_history = []
        self.scale_cooldown = 300  # 5 minutes
        self.last_scale_time = 0

    async def monitor_and_scale(self):
        """Monitor metrics and scale accordingly"""
        current_metrics = await self.get_current_metrics()
        self.metrics_history.append(current_metrics)

        # Keep only last 10 minutes of metrics
        if len(self.metrics_history) > 10:
            self.metrics_history.pop(0)

        # Calculate average utilization
        avg_utilization = sum(m['utilization'] for m in self.metrics_history) / len(self.metrics_history)

        current_time = time.time()
        time_since_last_scale = current_time - self.last_scale_time

        # Only scale if cooldown period has passed
        if time_since_last_scale < self.scale_cooldown:
            return

        if avg_utilization > self.target_utilization + 0.1:  # Scale up
            await self.scale_up()
            self.last_scale_time = current_time
        elif avg_utilization < self.target_utilization - 0.2:  # Scale down
            await self.scale_down()
            self.last_scale_time = current_time

    async def get_current_metrics(self):
        """Get current performance metrics"""
        # Implementation to get actual metrics
        return {
            'utilization': 0.8,
            'response_time': 200,
            'queue_length': 5
        }

    async def scale_up(self):
        """Scale up instances"""
        current_instances = await self.get_current_instance_count()
        new_count = min(current_instances + 1, self.chute.max_instances)
        await self.set_instance_count(new_count)

    async def scale_down(self):
        """Scale down instances"""
        current_instances = await self.get_current_instance_count()
        new_count = max(current_instances - 1, self.chute.min_instances)
        await self.set_instance_count(new_count)
```

## Workload Optimization

### Batch Processing for Cost Efficiency

Process multiple requests together:

```python
import asyncio
from typing import List, Dict, Any

class CostOptimizedBatchProcessor:
    def __init__(self, max_batch_size=32, max_wait_time=5.0):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.processing = False

    async def add_request(self, request_data: Dict[str, Any]) -> Any:
        """Add request to batch queue"""
        future = asyncio.Future()
        self.pending_requests.append({
            'data': request_data,
            'future': future
        })

        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self.process_batch())

        return await future

    async def process_batch(self):
        """Process accumulated requests as a batch"""
        if self.processing:
            return

        self.processing = True

        # Wait for batch to fill up or timeout
        start_time = time.time()
        while (len(self.pending_requests) < self.max_batch_size and
               time.time() - start_time < self.max_wait_time):
            await asyncio.sleep(0.1)

        if not self.pending_requests:
            self.processing = False
            return

        # Extract batch
        batch = self.pending_requests[:self.max_batch_size]
        self.pending_requests = self.pending_requests[self.max_batch_size:]

        try:
            # Process batch
            batch_data = [req['data'] for req in batch]
            results = await self.process_batch_data(batch_data)

            # Return results to futures
            for req, result in zip(batch, results):
                req['future'].set_result(result)

        except Exception as e:
            # Handle batch errors
            for req in batch:
                req['future'].set_exception(e)

        finally:
            self.processing = False

            # Process remaining requests if any
            if self.pending_requests:
                asyncio.create_task(self.process_batch())

    async def process_batch_data(self, batch_data: List[Dict[str, Any]]) -> List[Any]:
        """Process the actual batch - implement your logic here"""
        # Example: AI model inference on batch
        results = []
        for data in batch_data:
            # Process individual item
            result = await self.model_inference(data)
            results.append(result)
        return results

# Usage in chute
batch_processor = CostOptimizedBatchProcessor(max_batch_size=16, max_wait_time=2.0)

async def run_cost_optimized(inputs: Dict[str, Any]) -> Any:
    """Cost-optimized endpoint using batching"""
    result = await batch_processor.add_request(inputs)
    return result
```

## Model Optimization for Cost

### Model Quantization

Reduce computational costs through quantization:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class QuantizedModelForCost:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with 8-bit quantization
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            load_in_8bit=True,  # Reduces memory usage by ~50%
            device_map="auto"
        )

    async def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch prediction with quantized model"""
        # Process in batches for efficiency
        batch_size = 16
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Extract results
            for j, prediction in enumerate(predictions):
                results.append({
                    'text': batch[j],
                    'prediction': prediction.cpu().numpy().tolist(),
                    'confidence': float(torch.max(prediction))
                })

        return results

# Deploy with cost-optimized settings
cost_optimized_chute = Chute(
    username="myuser",
    name="quantized-inference",
    image=quantized_image,
    entry_file="quantized_model.py",
    entry_point="run",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=8,  # Reduced from 16GB due to quantization),
    concurrency=12,  # Higher concurrency due to reduced memory usage
    timeout_seconds=120
)
```

### Model Caching Strategy

Implement intelligent caching to reduce compute costs:

```python
import hashlib
import pickle
import redis
from typing import Optional, Dict, Any

class CostOptimizedCache:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.hit_count = 0
        self.miss_count = 0

    def get_cache_key(self, inputs: Dict[str, Any]) -> str:
        """Generate cache key from inputs"""
        # Create deterministic hash of inputs
        input_str = str(sorted(inputs.items()))
        return f"model_cache:{hashlib.md5(input_str.encode()).hexdigest()}"

    async def get_cached_result(self, inputs: Dict[str, Any]) -> Optional[Any]:
        """Get cached result if available"""
        cache_key = self.get_cache_key(inputs)

        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                self.hit_count += 1
                return pickle.loads(cached_data)
        except Exception:
            pass

        self.miss_count += 1
        return None

    async def cache_result(self, inputs: Dict[str, Any], result: Any, ttl: int = 3600):
        """Cache computation result"""
        cache_key = self.get_cache_key(inputs)

        try:
            serialized_result = pickle.dumps(result)
            self.redis_client.setex(cache_key, ttl, serialized_result)
        except Exception:
            pass

    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache performance statistics"""
        total_requests = self.hit_count + self.miss_count
        if total_requests == 0:
            return {"hit_rate": 0.0, "miss_rate": 0.0}

        return {
            "hit_rate": self.hit_count / total_requests,
            "miss_rate": self.miss_count / total_requests,
            "total_requests": total_requests
        }

# Global cache instance
cost_cache = CostOptimizedCache()

async def run_with_cost_cache(inputs: Dict[str, Any]) -> Any:
    """Run with intelligent caching for cost optimization"""
    # Try to get cached result first
    cached_result = await cost_cache.get_cached_result(inputs)
    if cached_result is not None:
        return {
            "result": cached_result,
            "cached": True,
            "cache_stats": cost_cache.get_cache_stats()
        }

    # Compute result if not cached
    result = await expensive_computation(inputs)

    # Cache result for future requests
    await cost_cache.cache_result(inputs, result, ttl=1800)  # 30 minutes

    return {
        "result": result,
        "cached": False,
        "cache_stats": cost_cache.get_cache_stats()
    }
```

## Cost Monitoring and Analytics

### Cost Tracking

Monitor and track costs in real-time:

```python
import time
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class CostMetric:
    timestamp: float
    gpu_hours: float
    compute_cost: float
    request_count: int
    cache_hit_rate: float

class CostMonitor:
    def __init__(self):
        self.cost_history: List[CostMetric] = []
        self.hourly_costs: Dict[str, float] = {}
        self.daily_budgets: Dict[str, float] = {}

    def record_usage(self, gpu_hours: float, compute_cost: float,
                    request_count: int, cache_hit_rate: float):
        """Record usage metrics"""
        metric = CostMetric(
            timestamp=time.time(),
            gpu_hours=gpu_hours,
            compute_cost=compute_cost,
            request_count=request_count,
            cache_hit_rate=cache_hit_rate
        )
        self.cost_history.append(metric)

        # Update hourly tracking
        hour_key = datetime.now().strftime("%Y-%m-%d-%H")
        if hour_key not in self.hourly_costs:
            self.hourly_costs[hour_key] = 0
        self.hourly_costs[hour_key] += compute_cost

    def get_daily_cost(self, date: str = None) -> float:
        """Get total cost for a specific day"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        daily_cost = 0
        for hour_key, cost in self.hourly_costs.items():
            if hour_key.startswith(date):
                daily_cost += cost

        return daily_cost

    def check_budget_alert(self, daily_budget: float) -> Dict[str, Any]:
        """Check if approaching budget limits"""
        current_cost = self.get_daily_cost()
        budget_usage = current_cost / daily_budget

        alert_level = "green"
        if budget_usage > 0.9:
            alert_level = "red"
        elif budget_usage > 0.7:
            alert_level = "yellow"

        return {
            "current_cost": current_cost,
            "daily_budget": daily_budget,
            "budget_usage": budget_usage,
            "alert_level": alert_level,
            "remaining_budget": daily_budget - current_cost
        }

    def get_cost_optimization_suggestions(self) -> List[str]:
        """Generate cost optimization suggestions"""
        suggestions = []

        # Analyze recent metrics
        recent_metrics = self.cost_history[-10:] if len(self.cost_history) >= 10 else self.cost_history

        if recent_metrics:
            avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
            avg_cost_per_request = sum(m.compute_cost / max(m.request_count, 1) for m in recent_metrics) / len(recent_metrics)

            if avg_cache_hit_rate < 0.5:
                suggestions.append("Consider increasing cache TTL or implementing better caching strategy")

            if avg_cost_per_request > 0.01:  # Threshold for expensive requests
                suggestions.append("Consider using smaller models or batch processing")

            # Check for usage patterns
            hourly_usage = {}
            for metric in recent_metrics:
                hour = datetime.fromtimestamp(metric.timestamp).hour
                if hour not in hourly_usage:
                    hourly_usage[hour] = []
                hourly_usage[hour].append(metric.compute_cost)

            # Suggest time-based scaling if usage varies significantly
            if len(hourly_usage) > 3:
                costs = [sum(costs) for costs in hourly_usage.values()]
                if max(costs) / min(costs) > 3:
                    suggestions.append("Consider time-based scaling to reduce costs during low-usage periods")

        return suggestions

# Global cost monitor
cost_monitor = CostMonitor()

async def run_with_cost_monitoring(inputs: Dict[str, Any]) -> Any:
    """Run with cost monitoring"""
    start_time = time.time()

    # Execute request
    result = await process_request(inputs)

    # Calculate metrics
    execution_time = time.time() - start_time
    gpu_hours = execution_time / 3600  # Convert to hours
    estimated_cost = gpu_hours * 0.50  # $0.50 per GPU hour (example rate)

    # Record usage
    cost_monitor.record_usage(
        gpu_hours=gpu_hours,
        compute_cost=estimated_cost,
        request_count=1,
        cache_hit_rate=0.8  # From cache system
    )

    # Check budget
    budget_status = cost_monitor.check_budget_alert(daily_budget=50.0)

    return {
        "result": result,
        "cost_info": {
            "execution_time": execution_time,
            "estimated_cost": estimated_cost,
            "budget_status": budget_status
        }
    }
```

## Cost Optimization Best Practices

### 1. Resource Selection

- Choose the smallest GPU that meets your performance requirements
- Use CPU-only instances for non-AI workloads
- Consider memory requirements carefully

### 2. Scaling Strategy

- Implement auto-scaling based on actual demand
- Use time-based scaling for predictable patterns
- Set appropriate scale-down policies

### 3. Workload Optimization

- Batch requests when possible
- Implement intelligent caching
- Use model quantization for inference workloads

### 4. Monitoring and Alerts

- Set up budget alerts and monitoring
- Track cost per request and optimization opportunities
- Regular review of usage patterns

## Next Steps

- **[Performance Guide](performance)** - Optimize performance while controlling costs
- **[Best Practices](best-practices)** - General optimization strategies
- **[Monitoring](../monitoring)** - Advanced cost and performance monitoring

For enterprise cost optimization, see the [Enterprise Cost Management Guide](../enterprise/cost-management).
