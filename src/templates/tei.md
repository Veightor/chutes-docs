# TEI Template

The **TEI (Text Embeddings Inference) template** provides optimized text embedding generation using Hugging Face's high-performance inference server. Perfect for semantic search, similarity detection, and RAG applications.

## What is TEI?

Text Embeddings Inference (TEI) is a specialized inference server for embedding models that provides:

- ⚡ **Optimized performance** with Rust-based implementation
- 📊 **Batch processing** for efficient throughput
- 🔄 **Automatic batching** and request queuing
- 📏 **Embedding normalization** and pooling options
- 🎯 **Production-ready** with health checks and metrics

## Quick Start

```python
from chutes.chute import NodeSelector
from chutes.chute.template.tei import build_tei_chute

chute = build_tei_chute(
    username="myuser",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    revision="main",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=8
    )
)
```

This creates a complete TEI deployment with:

- ✅ Optimized embedding inference server
- ✅ OpenAI-compatible embeddings API
- ✅ Automatic request batching
- ✅ Built-in normalization
- ✅ Auto-scaling based on demand

## Function Reference

### `build_tei_chute()`

```python
def build_tei_chute(
    username: str,
    model_name: str,
    revision: str = "main",
    node_selector: NodeSelector = None,
    image: str | Image = None,
    tagline: str = "",
    readme: str = "",
    concurrency: int = 1,

    # TEI-specific parameters
    max_batch_tokens: int = 16384,
    max_batch_requests: int = 512,
    max_concurrent_requests: int = 512,
    pooling: str = "mean",
    normalize: bool = True,
    trust_remote_code: bool = False,
    **kwargs
) -> Chute:
```

#### Required Parameters

- **`username`**: Your Chutes username
- **`model_name`**: HuggingFace embedding model identifier

#### TEI Configuration

- **`max_batch_tokens`**: Maximum tokens per batch (default: 16384)
- **`max_batch_requests`**: Maximum requests per batch (default: 512)
- **`max_concurrent_requests`**: Maximum concurrent requests (default: 512)
- **`pooling`**: Pooling strategy - "mean", "cls", or "max" (default: "mean")
- **`normalize`**: Whether to normalize embeddings (default: True)
- **`trust_remote_code`**: Allow custom model code execution (default: False)

## Complete Example

```python
from chutes.chute import NodeSelector
from chutes.chute.template.tei import build_tei_chute

# Build TEI chute for embedding generation
chute = build_tei_chute(
    username="myuser",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    revision="main",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=8
    ),
    tagline="High-performance text embeddings",
    readme="""
# Text Embeddings Service

Fast and efficient text embedding generation using TEI.

## Features
- OpenAI-compatible embeddings API
- Automatic batching and optimization
- Normalized embeddings for similarity search
- Production-ready performance

## API Endpoints
- `/v1/embeddings` - Generate embeddings
- `/embed` - Alternative embedding endpoint
- `/health` - Health check
    """,

    # TEI optimization
    max_batch_tokens=32768,
    max_batch_requests=256,
    pooling="mean",
    normalize=True
)
```

## API Endpoints

### OpenAI-Compatible Embeddings

```bash
curl -X POST https://myuser-tei-chute.chutes.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": [
      "The quick brown fox jumps over the lazy dog",
      "Machine learning is transforming technology"
    ]
  }'
```

### Single Text Embedding

```bash
curl -X POST https://myuser-tei-chute.chutes.ai/embed \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": "This is a sample text for embedding generation"
  }'
```

### Batch Processing

```bash
curl -X POST https://myuser-tei-chute.chutes.ai/embed \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      "First document to embed",
      "Second document for embedding",
      "Third text for similarity search"
    ]
  }'
```

## Model Recommendations

### Small & Fast Models

```python
# Lightweight, fast inference
NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=4,
    include=["rtx3090", "rtx4090"]
)

# Recommended models:
# - sentence-transformers/all-MiniLM-L6-v2 (384 dim)
# - sentence-transformers/all-MiniLM-L12-v2 (384 dim)
# - microsoft/codebert-base (768 dim)
```

### Balanced Performance Models

```python
# Good balance of speed and quality
NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=8,
    include=["rtx4090", "a100"]
)

# Recommended models:
# - sentence-transformers/all-mpnet-base-v2 (768 dim)
# - sentence-transformers/multi-qa-mpnet-base-dot-v1 (768 dim)
# - thenlper/gte-base (768 dim)
```

### High-Quality Models

```python
# Best embedding quality
NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=12,
    include=["a100", "h100"]
)

# Recommended models:
# - sentence-transformers/all-mpnet-base-v2 (768 dim)
# - intfloat/e5-large-v2 (1024 dim)
# - BAAI/bge-large-en-v1.5 (1024 dim)
```

## Use Cases

### 1. **Semantic Search**

```python
search_chute = build_tei_chute(
    username="myuser",
    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    tagline="Semantic search embeddings",
    max_batch_tokens=32768,  # Handle large documents
    normalize=True  # Important for similarity search
)
```

### 2. **Document Similarity**

```python
similarity_chute = build_tei_chute(
    username="myuser",
    model_name="sentence-transformers/all-mpnet-base-v2",
    tagline="Document similarity service",
    pooling="mean",
    normalize=True
)
```

### 3. **Code Embeddings**

```python
code_chute = build_tei_chute(
    username="myuser",
    model_name="microsoft/codebert-base",
    tagline="Code similarity and search",
    max_batch_tokens=16384,  # Typical code snippet length
    trust_remote_code=True   # May be needed for code models
)
```

### 4. **Multilingual Embeddings**

```python
multilingual_chute = build_tei_chute(
    username="myuser",
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    tagline="Multilingual text embeddings",
    max_batch_requests=1024  # Handle diverse languages efficiently
)
```

## Performance Optimization

### Throughput Optimization

```python
# Maximize throughput for batch processing
chute = build_tei_chute(
    username="myuser",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    max_batch_tokens=65536,      # Large batches
    max_batch_requests=1024,     # Many requests
    max_concurrent_requests=2048, # High concurrency
    concurrency=8                # Multiple chute instances
)
```

### Latency Optimization

```python
# Minimize latency for real-time applications
chute = build_tei_chute(
    username="myuser",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    max_batch_tokens=4096,       # Smaller batches
    max_batch_requests=32,       # Fewer requests per batch
    max_concurrent_requests=128  # Lower concurrency
)
```

### Memory Optimization

```python
# Optimize for memory usage
chute = build_tei_chute(
    username="myuser",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    max_batch_tokens=8192,       # Moderate batch size
    max_batch_requests=256,      # Moderate requests
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=6    # Conservative memory
    )
)
```

## Testing Your TEI Chute

### Python Client

```python
import requests
import numpy as np

# Generate embeddings
response = requests.post(
    "https://myuser-tei-chute.chutes.ai/v1/embeddings",
    json={
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "input": [
            "The quick brown fox",
            "A fast brown animal",
            "The weather is nice today"
        ]
    }
)

result = response.json()
embeddings = [item["embedding"] for item in result["data"]]

# Calculate similarity
emb1 = np.array(embeddings[0])
emb2 = np.array(embeddings[1])
emb3 = np.array(embeddings[2])

similarity_1_2 = np.dot(emb1, emb2)  # Should be high
similarity_1_3 = np.dot(emb1, emb3)  # Should be low

print(f"Similarity fox vs animal: {similarity_1_2:.3f}")
print(f"Similarity fox vs weather: {similarity_1_3:.3f}")
```

### OpenAI Client

```python
from openai import OpenAI

# Use OpenAI client with your chute
client = OpenAI(
    api_key="dummy",  # Not needed for Chutes
    base_url="https://myuser-tei-chute.chutes.ai/v1"
)

# Generate embeddings
response = client.embeddings.create(
    model="sentence-transformers/all-MiniLM-L6-v2",
    input=[
        "Document for semantic search",
        "Query for finding similar content"
    ]
)

for i, item in enumerate(response.data):
    print(f"Embedding {i}: {len(item.embedding)} dimensions")
```

### Batch Processing Test

```python
import asyncio
import aiohttp
import time

async def test_batch_performance():
    """Test batch processing performance."""

    # Generate test texts
    texts = [f"This is test document number {i} for embedding generation."
             for i in range(100)]

    # Test batch processing
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://myuser-tei-chute.chutes.ai/embed",
            json={"inputs": texts}
        ) as response:
            result = await response.json()

    batch_time = time.time() - start_time

    print(f"Batch processing:")
    print(f"  Texts: {len(texts)}")
    print(f"  Time: {batch_time:.2f}s")
    print(f"  Throughput: {len(texts)/batch_time:.1f} texts/sec")

    # Test individual requests
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = []
        for text in texts[:10]:  # Test subset for fairness
            task = session.post(
                "https://myuser-tei-chute.chutes.ai/embed",
                json={"inputs": text}
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

    individual_time = time.time() - start_time

    print(f"\nIndividual requests:")
    print(f"  Texts: 10")
    print(f"  Time: {individual_time:.2f}s")
    print(f"  Throughput: {10/individual_time:.1f} texts/sec")
    print(f"  Speedup: {(individual_time*10)/(batch_time):.1f}x")

asyncio.run(test_batch_performance())
```

## Integration Examples

### Semantic Search with Vector Database

```python
import requests
import numpy as np
from pinecone import Pinecone

# Initialize vector database
pc = Pinecone(api_key="your-api-key")
index = pc.Index("semantic-search")

def embed_text(text):
    """Generate embedding for text."""
    response = requests.post(
        "https://myuser-tei-chute.chutes.ai/v1/embeddings",
        json={
            "model": "sentence-transformers/all-mpnet-base-v2",
            "input": text
        }
    )
    return response.json()["data"][0]["embedding"]

def index_documents(documents):
    """Index documents for search."""
    vectors = []

    for i, doc in enumerate(documents):
        embedding = embed_text(doc)
        vectors.append({
            "id": str(i),
            "values": embedding,
            "metadata": {"text": doc}
        })

    index.upsert(vectors)

def search_documents(query, top_k=5):
    """Search for similar documents."""
    query_embedding = embed_text(query)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    return [(match.score, match.metadata["text"])
            for match in results.matches]

# Example usage
documents = [
    "Python is a programming language",
    "Machine learning uses algorithms",
    "The weather is sunny today",
    "Neural networks are inspired by the brain"
]

index_documents(documents)
results = search_documents("What is artificial intelligence?")

for score, text in results:
    print(f"Score: {score:.3f} - {text}")
```

### Document Clustering

```python
import requests
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def embed_documents(documents):
    """Generate embeddings for multiple documents."""
    response = requests.post(
        "https://myuser-tei-chute.chutes.ai/v1/embeddings",
        json={
            "model": "sentence-transformers/all-mpnet-base-v2",
            "input": documents
        }
    )
    return [item["embedding"] for item in response.json()["data"]]

def cluster_documents(documents, n_clusters=3):
    """Cluster documents based on embeddings."""
    # Generate embeddings
    embeddings = embed_documents(documents)
    embeddings_array = np.array(embeddings)

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings_array)

    # Visualize with PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_array)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=clusters, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('Document Clustering')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    # Add document labels
    for i, doc in enumerate(documents):
        plt.annotate(f"Doc {i}", (embeddings_2d[i, 0], embeddings_2d[i, 1]))

    plt.show()

    return clusters

# Example usage
documents = [
    "Python programming language tutorial",
    "JavaScript web development guide",
    "Machine learning with neural networks",
    "Deep learning and artificial intelligence",
    "HTML and CSS for beginners",
    "React framework for web apps",
    "Natural language processing techniques",
    "Computer vision and image recognition"
]

clusters = cluster_documents(documents)

# Group documents by cluster
for cluster_id in range(max(clusters) + 1):
    print(f"\nCluster {cluster_id}:")
    for i, doc in enumerate(documents):
        if clusters[i] == cluster_id:
            print(f"  - {doc}")
```

## Troubleshooting

### Common Issues

**Slow embedding generation?**

- Increase `max_batch_tokens` for better throughput
- Use a smaller/faster model
- Optimize hardware with more GPU memory

**Out of memory errors?**

- Reduce `max_batch_tokens`
- Decrease `max_batch_requests`
- Use a smaller model
- Increase GPU VRAM requirements

**Poor embedding quality?**

- Use a larger, more sophisticated model
- Ensure proper text preprocessing
- Check if the model matches your domain

**High latency?**

- Reduce batch sizes for faster response
- Use a smaller/faster model
- Consider multiple smaller instances

### Performance Monitoring

```python
import requests
import time

def monitor_performance():
    """Monitor TEI chute performance."""

    # Test different batch sizes
    batch_sizes = [1, 5, 10, 25, 50]
    test_text = "This is a test document for performance monitoring."

    for batch_size in batch_sizes:
        texts = [test_text] * batch_size

        start_time = time.time()
        response = requests.post(
            "https://myuser-tei-chute.chutes.ai/embed",
            json={"inputs": texts}
        )
        end_time = time.time()

        if response.status_code == 200:
            throughput = batch_size / (end_time - start_time)
            print(f"Batch size {batch_size}: {throughput:.1f} texts/sec")
        else:
            print(f"Batch size {batch_size}: Error {response.status_code}")

monitor_performance()
```

## Best Practices

### 1. **Model Selection**

```python
# For general text similarity
model_name = "sentence-transformers/all-mpnet-base-v2"

# For search applications
model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# For code similarity
model_name = "microsoft/codebert-base"

# For multilingual applications
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
```

### 2. **Batch Size Tuning**

```python
# For real-time applications (low latency)
max_batch_tokens = 4096
max_batch_requests = 32

# For bulk processing (high throughput)
max_batch_tokens = 32768
max_batch_requests = 512

# For balanced performance
max_batch_tokens = 16384
max_batch_requests = 256
```

### 3. **Text Preprocessing**

```python
def preprocess_text(text):
    """Preprocess text for better embeddings."""
    # Remove excessive whitespace
    text = " ".join(text.split())

    # Normalize length (very long texts may be truncated)
    if len(text) > 5000:  # Adjust based on model's max length
        text = text[:5000]

    return text.strip()

# Apply preprocessing before embedding
texts = [preprocess_text(text) for text in raw_texts]
```

### 4. **Error Handling**

```python
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_embeddings(texts):
    """Generate embeddings with retry logic."""
    try:
        response = requests.post(
            "https://myuser-tei-chute.chutes.ai/v1/embeddings",
            json={
                "model": "sentence-transformers/all-mpnet-base-v2",
                "input": texts
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        raise
```

## Next Steps

- **[VLLM Template](/docs/templates/vllm)** - High-performance language model serving
- **[Diffusion Template](/docs/templates/diffusion)** - Image generation capabilities
- **[Vector Databases Guide](/docs/guides/vector-databases)** - Integration with vector stores
- **[Semantic Search Example](/docs/examples/semantic-search)** - Complete search application
