# Batch Processing

This example shows how to efficiently process multiple inputs in a single request, optimizing GPU utilization and reducing API overhead for high-throughput scenarios.

## What We'll Build

A batch text processing service that:

- 📊 **Processes multiple texts** in a single request
- ⚡ **Optimizes GPU utilization** with efficient batching
- 🔄 **Handles variable input sizes** with dynamic padding
- 📈 **Provides performance metrics** and timing information
- 🛡️ **Validates batch constraints** for stability

## Complete Example

### `batch_processor.py`

````python
import torch
import time
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydantic import BaseModel, Field, validator
from fastapi import HTTPException

from chutes.chute import Chute, NodeSelector
from chutes.image import Image

# === INPUT/OUTPUT SCHEMAS ===

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to process")
    max_length: int = Field(512, ge=50, le=1024, description="Maximum token length")
    batch_size: int = Field(16, ge=1, le=32, description="Processing batch size")

    @validator('texts')
    def validate_texts(cls, v):
        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty')
            if len(text) > 10000:
                raise ValueError(f'Text at index {i} is too long (max 10000 chars)')
        return [text.strip() for text in v]

class TextResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    token_count: int
    processing_order: int

class BatchResult(BaseModel):
    results: List[TextResult]
    total_texts: int
    processing_time: float
    average_time_per_text: float
    batch_info: dict
    performance_metrics: dict

# === CUSTOM IMAGE ===

image = (
    Image(username="myuser", name="batch-processor", tag="1.0")
    .from_base("nvidia/cuda:12.2-runtime-ubuntu22.04")
    .with_python("3.11")
    .run_command("pip install torch==2.1.0 transformers==4.30.0 accelerate==0.20.0 numpy>=1.24.0")
    .with_env("TRANSFORMERS_CACHE", "/app/models")
    .with_env("TOKENIZERS_PARALLELISM", "false")  # Avoid warnings
)

# === CHUTE DEFINITION ===

chute = Chute(
    username="myuser",
    name="batch-processor",
    image=image,
    tagline="High-throughput batch text processing",
    readme="""
# Batch Text Processor

Efficiently process multiple texts in a single request with optimized GPU utilization.

## Usage

```bash
curl -X POST https://myuser-batch-processor.chutes.ai/process-batch \\
  -H "Content-Type: application/json" \\
  -d '{
    "texts": [
      "I love this product!",
      "This is terrible quality.",
      "Amazing service and support!"
    ],
    "batch_size": 8
  }'
```

## Features

- Process up to 100 texts per request
- Automatic batching for GPU optimization
- Dynamic padding for efficient processing
- Comprehensive performance metrics
  """,
  node_selector=NodeSelector(
  gpu_count=1,
  min_vram_gb_per_gpu=12
  ),
  concurrency=4 # Allow multiple concurrent requests
  )

# === MODEL LOADING ===

@chute.on_startup()
async def load_model(self):
"""Load sentiment analysis model optimized for batch processing."""
print("Loading model for batch processing...")

    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    # Load tokenizer and model
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Optimize for batch processing
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model.to(self.device)
    self.model.eval()

    # Enable optimizations
    if torch.cuda.is_available():
        # Enable mixed precision for faster processing
        self.scaler = torch.cuda.amp.GradScaler()
        # Enable TensorCore optimizations where available
        torch.backends.cudnn.benchmark = True

    # Cache for performance tracking
    self.batch_stats = {
        "total_requests": 0,
        "total_texts_processed": 0,
        "average_batch_time": 0.0,
        "peak_batch_size": 0
    }

    print(f"Model loaded on {self.device} with batch optimizations enabled")

# === BATCH PROCESSING ENDPOINTS ===

@chute.cord(
public_api_path="/process-batch",
method="POST",
input_schema=BatchTextInput,
output_content_type="application/json"
)
async def process_batch(self, data: BatchTextInput) -> BatchResult:
"""Process multiple texts efficiently with batching."""

    start_time = time.time()

    # Update statistics
    self.batch_stats["total_requests"] += 1
    self.batch_stats["total_texts_processed"] += len(data.texts)

    try:
        # Process in chunks if batch is too large
        all_results = []
        total_batches = 0

        for chunk_start in range(0, len(data.texts), data.batch_size):
            chunk_end = min(chunk_start + data.batch_size, len(data.texts))
            text_chunk = data.texts[chunk_start:chunk_end]

            # Process this chunk
            chunk_results = await self._process_chunk(
                text_chunk,
                data.max_length,
                chunk_start
            )
            all_results.extend(chunk_results)
            total_batches += 1

        # Calculate performance metrics
        processing_time = time.time() - start_time
        avg_time_per_text = processing_time / len(data.texts)

        # Update global stats
        self.batch_stats["average_batch_time"] = (
            (self.batch_stats["average_batch_time"] * (self.batch_stats["total_requests"] - 1) +
             processing_time) / self.batch_stats["total_requests"]
        )
        self.batch_stats["peak_batch_size"] = max(
            self.batch_stats["peak_batch_size"],
            len(data.texts)
        )

        return BatchResult(
            results=all_results,
            total_texts=len(data.texts),
            processing_time=processing_time,
            average_time_per_text=avg_time_per_text,
            batch_info={
                "requested_batch_size": data.batch_size,
                "actual_batches_used": total_batches,
                "max_length": data.max_length,
                "device": self.device
            },
            performance_metrics={
                "texts_per_second": len(data.texts) / processing_time,
                "gpu_memory_used": self._get_gpu_memory_usage(),
                "total_tokens_processed": sum(r.token_count for r in all_results)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

async def \_process_chunk(self, texts: List[str], max_length: int, start_index: int) -> List[TextResult]:
"""Process a chunk of texts efficiently."""

    # Tokenize all texts in the chunk
    encoded = self.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Move to device
    input_ids = encoded['input_ids'].to(self.device)
    attention_mask = encoded['attention_mask'].to(self.device)

    # Process with mixed precision if available
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

    # Get predictions
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_classes = predictions.argmax(dim=-1)
    confidences = predictions.max(dim=-1).values

    # Convert to results
    labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    results = []

    for i, (text, pred_class, confidence, tokens) in enumerate(
        zip(texts, predicted_classes, confidences, input_ids)
    ):
        results.append(TextResult(
            text=text,
            sentiment=labels[pred_class.item()],
            confidence=confidence.item(),
            token_count=tokens.ne(self.tokenizer.pad_token_id).sum().item(),
            processing_order=start_index + i
        ))

    return results

def \_get_gpu_memory_usage(self) -> Optional[float]:
"""Get current GPU memory usage in GB."""
if torch.cuda.is_available():
return torch.cuda.memory_allocated() / 1024\*\*3
return None

@chute.cord(
public_api_path="/batch-stats",
method="GET",
output_content_type="application/json"
)
async def get_batch_stats(self) -> dict:
"""Get performance statistics for batch processing."""
stats = self.batch_stats.copy()

    # Add current system info
    stats.update({
        "device": self.device,
        "model_loaded": hasattr(self, 'model'),
        "current_gpu_memory": self._get_gpu_memory_usage(),
        "max_gpu_memory": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
    })

    return stats

# === STREAMING BATCH PROCESSING ===

@chute.cord(
public_api_path="/process-batch-stream",
method="POST",
input_schema=BatchTextInput,
stream=True,
output_content_type="application/json"
)
async def process_batch_stream(self, data: BatchTextInput):
"""Process batch with streaming progress updates."""

    start_time = time.time()

    yield {
        "status": "started",
        "total_texts": len(data.texts),
        "batch_size": data.batch_size,
        "estimated_batches": (len(data.texts) + data.batch_size - 1) // data.batch_size
    }

    all_results = []

    for batch_idx, chunk_start in enumerate(range(0, len(data.texts), data.batch_size)):
        chunk_end = min(chunk_start + data.batch_size, len(data.texts))
        text_chunk = data.texts[chunk_start:chunk_end]

        yield {
            "status": "processing_batch",
            "batch_number": batch_idx + 1,
            "batch_size": len(text_chunk),
            "progress": chunk_end / len(data.texts)
        }

        # Process chunk
        batch_start = time.time()
        chunk_results = await self._process_chunk(text_chunk, data.max_length, chunk_start)
        batch_time = time.time() - batch_start

        all_results.extend(chunk_results)

        yield {
            "status": "batch_complete",
            "batch_number": batch_idx + 1,
            "batch_time": batch_time,
            "texts_per_second": len(text_chunk) / batch_time,
            "partial_results": chunk_results
        }

    # Final results
    total_time = time.time() - start_time
    yield {
        "status": "completed",
        "total_time": total_time,
        "average_time_per_text": total_time / len(data.texts),
        "final_results": all_results
    }

# Test locally

if **name** == "**main**":
import asyncio

    async def test_batch_processing():
        # Simulate startup
        await load_model(chute)

        # Test batch
        test_texts = [
            "I love this product!",
            "Terrible quality, very disappointed.",
            "Pretty good, would recommend.",
            "Outstanding service and delivery!",
            "Not worth the money spent.",
            "Amazing features and great design!"
        ]

        test_input = BatchTextInput(
            texts=test_texts,
            batch_size=3
        )

        result = await process_batch(chute, test_input)
        print(f"Processed {result.total_texts} texts in {result.processing_time:.2f}s")
        print(f"Average time per text: {result.average_time_per_text:.3f}s")

        for r in result.results:
            print(f"'{r.text[:30]}...' -> {r.sentiment} ({r.confidence:.2f})")

    asyncio.run(test_batch_processing())

````

## Performance Optimization Techniques

### 1. **Dynamic Batching**

```python
# Automatically adjust batch size based on text lengths
def optimize_batch_size(texts: List[str], max_tokens: int = 8192) -> int:
    avg_length = sum(len(text.split()) for text in texts) / len(texts)
    estimated_tokens_per_text = avg_length * 1.3  # Account for subword tokenization

    optimal_batch_size = max(1, int(max_tokens / estimated_tokens_per_text))
    return min(optimal_batch_size, 32)  # Cap at 32 for memory safety
```

### 2. **Memory-Efficient Processing**

```python
# Process very large batches in chunks
async def process_large_batch(self, texts: List[str], chunk_size: int = 50):
    results = []

    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        chunk_results = await self._process_chunk(chunk, 512, i)
        results.extend(chunk_results)

        # Clear GPU cache between chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results
```

### 3. **Mixed Precision Training**

```python
# Use automatic mixed precision for faster processing
with torch.cuda.amp.autocast():
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
```

## Testing the Batch API

### Simple Batch Test

```python
import requests
import time

# Prepare test data
texts = [
    "I absolutely love this new product!",
    "Worst purchase I've ever made.",
    "It's okay, nothing special.",
    "Fantastic quality and great service!",
    "Complete waste of money.",
    "Highly recommend to everyone!",
    "Poor customer support experience.",
    "Exceeded all my expectations!",
    "Not worth the high price.",
    "Perfect for my needs!"
]

# Test different batch sizes
for batch_size in [2, 5, 10]:
    print(f"\nTesting batch size: {batch_size}")

    start_time = time.time()
    response = requests.post(
        "https://myuser-batch-processor.chutes.ai/process-batch",
        json={
            "texts": texts,
            "batch_size": batch_size,
            "max_length": 256
        }
    )

    result = response.json()
    print(f"Total time: {result['processing_time']:.2f}s")
    print(f"Texts/second: {result['performance_metrics']['texts_per_second']:.1f}")
    print(f"Avg time per text: {result['average_time_per_text']:.3f}s")
```

### Performance Comparison

```python
import asyncio
import aiohttp
import time

async def compare_batch_vs_individual():
    """Compare batch processing vs individual requests."""

    texts = ["Sample text for testing"] * 20

    # Test individual requests
    start_time = time.time()
    individual_results = []

    async with aiohttp.ClientSession() as session:
        tasks = []
        for text in texts:
            task = session.post(
                "https://myuser-batch-processor.chutes.ai/analyze-single",
                json={"text": text}
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        for resp in responses:
            result = await resp.json()
            individual_results.append(result)

    individual_time = time.time() - start_time

    # Test batch processing
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://myuser-batch-processor.chutes.ai/process-batch",
            json={"texts": texts, "batch_size": 10}
        ) as resp:
            batch_result = await resp.json()

    batch_time = time.time() - start_time

    print(f"Individual requests: {individual_time:.2f}s")
    print(f"Batch processing: {batch_time:.2f}s")
    print(f"Speedup: {individual_time / batch_time:.1f}x")

asyncio.run(compare_batch_vs_individual())
```

### Streaming Batch Processing

```python
import asyncio
import aiohttp
import json

async def test_streaming_batch():
    """Test streaming batch processing with progress updates."""

    texts = [f"Test message number {i} for batch processing" for i in range(25)]

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://myuser-batch-processor.chutes.ai/process-batch-stream",
            json={"texts": texts, "batch_size": 5}
        ) as response:

            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode())
                        if data['status'] == 'processing_batch':
                            print(f"Processing batch {data['batch_number']} ({data['progress']:.1%} complete)")
                        elif data['status'] == 'batch_complete':
                            print(f"Batch {data['batch_number']} completed in {data['batch_time']:.2f}s")
                        elif data['status'] == 'completed':
                            print(f"All processing completed in {data['total_time']:.2f}s")
                    except json.JSONDecodeError:
                        continue

asyncio.run(test_streaming_batch())
```

## Key Performance Concepts

### 1. **Batch Size Optimization**

```python
# Find optimal batch size for your hardware
def find_optimal_batch_size(model, tokenizer, device, max_length=512):
    batch_sizes = [1, 2, 4, 8, 16, 32]
    test_texts = ["Sample text for testing"] * 32

    best_throughput = 0
    best_batch_size = 1

    for batch_size in batch_sizes:
        try:
            start_time = time.time()

            # Test processing
            for i in range(0, len(test_texts), batch_size):
                batch = test_texts[i:i + batch_size]
                encoded = tokenizer(batch, padding=True, truncation=True,
                                  max_length=max_length, return_tensors="pt")

                with torch.no_grad():
                    _ = model(**encoded.to(device))

            total_time = time.time() - start_time
            throughput = len(test_texts) / total_time

            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = batch_size

        except RuntimeError as e:
            if "out of memory" in str(e):
                break

    return best_batch_size, best_throughput
```

### 2. **Memory Management**

```python
# Monitor and manage GPU memory
def manage_gpu_memory():
    if torch.cuda.is_available():
        # Clear cache between large batches
        torch.cuda.empty_cache()

        # Get memory usage
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3

        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

        # Set memory fraction if needed
        torch.cuda.set_per_process_memory_fraction(0.8)
```

### 3. **Padding Optimization**

```python
# Minimize padding for better efficiency
def optimize_padding(texts, tokenizer, max_length):
    # Sort by length to minimize padding
    text_lengths = [(len(text), i, text) for i, text in enumerate(texts)]
    text_lengths.sort()

    batches = []
    current_batch = []

    for length, original_idx, text in text_lengths:
        current_batch.append((original_idx, text))

        # Create batch when we have enough similar-length texts
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []

    if current_batch:
        batches.append(current_batch)

    return batches
```

## Common Batch Processing Patterns

### 1. **Classification Tasks**

```python
# Sentiment analysis batch processing
async def batch_sentiment_analysis(texts: List[str]) -> List[dict]:
    results = []
    batch_size = 16

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = await process_sentiment_batch(batch)
        results.extend(batch_results)

    return results
```

### 2. **Text Generation**

```python
# Batch text generation with different prompts
async def batch_text_generation(prompts: List[str]) -> List[str]:
    generated_texts = []

    # Process prompts in batches
    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start:batch_start + batch_size]

        # Generate for batch
        batch_outputs = model.generate(
            **tokenizer(batch_prompts, return_tensors="pt", padding=True),
            max_length=100,
            num_return_sequences=1
        )

        batch_texts = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        generated_texts.extend(batch_texts)

    return generated_texts
```

### 3. **Embedding Generation**

```python
# Batch embedding generation
async def batch_embeddings(texts: List[str]) -> List[List[float]]:
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Tokenize batch
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**encoded.to(device))
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)

        embeddings.extend(batch_embeddings.cpu().tolist())

    return embeddings
```

## Next Steps

- **[Multi-Model Analysis](/docs/examples/multi-model-analysis)** - Combine multiple AI models
- **[Performance Optimization](/docs/guides/performance)** - Advanced speed optimization
- **[Production Deployment](/docs/guides/production)** - Scale to production workloads
- **[Cost Optimization](/docs/guides/cost-optimization)** - Manage processing costs
