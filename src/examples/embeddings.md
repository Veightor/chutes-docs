# Text Embeddings with TEI

This guide demonstrates how to build powerful text embedding services using Text Embeddings Inference (TEI), enabling semantic search, similarity analysis, and retrieval-augmented generation (RAG) applications.

## Overview

Text Embeddings Inference (TEI) is a high-performance embedding server that provides:

- **Fast Inference**: Optimized for batch processing and low latency
- **Multiple Models**: Support for various embedding architectures
- **Similarity Search**: Built-in similarity and ranking capabilities
- **Pooling Strategies**: Multiple pooling methods for optimal embeddings
- **Batch Processing**: Efficient handling of multiple texts
- **Production Ready**: Auto-scaling and error handling

## Complete Implementation

### Input Schema Design

Define comprehensive input validation for embedding operations:

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from enum import Enum

class PoolingStrategy(str, Enum):
    CLS = "cls"                    # Use [CLS] token
    MEAN = "mean"                  # Mean pooling
    MAX = "max"                    # Max pooling
    MEAN_SQRT_LEN = "mean_sqrt_len" # Mean pooling with sqrt normalization

class EmbeddingInput(BaseModel):
    inputs: Union[str, List[str]]  # Single text or batch
    normalize: bool = Field(default=True)
    truncate: bool = Field(default=True)
    pooling: Optional[PoolingStrategy] = PoolingStrategy.MEAN

class SimilarityInput(BaseModel):
    source_text: str
    target_texts: List[str] = Field(max_items=100)
    normalize: bool = Field(default=True)

class RerankInput(BaseModel):
    query: str
    texts: List[str] = Field(max_items=50)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)

class SearchInput(BaseModel):
    query: str
    corpus: List[str] = Field(max_items=1000)
    top_k: int = Field(default=10, ge=1, le=100)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
```

### Custom Image with TEI

Build a custom image with Text Embeddings Inference:

```python
from chutes.image import Image
from chutes.chute import Chute, NodeSelector

image = (
    Image(
        username="myuser",
        name="text-embeddings",
        tag="0.0.1",
        readme="High-performance text embeddings with TEI")
    .from_base("parachutes/base-python:3.11")
    .run_command("pip install --upgrade pip")
    .run_command("pip install text-embeddings-inference-client")
    .run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    .run_command("pip install transformers sentence-transformers")
    .run_command("pip install numpy scikit-learn faiss-cpu")
    .run_command("pip install loguru pydantic fastapi")
    # Install TEI server
    .run_command(
        "wget https://github.com/huggingface/text-embeddings-inference/releases/download/v1.2.3/text-embeddings-inference-1.2.3-x86_64-unknown-linux-gnu.tar.gz && "
        "tar -xzf text-embeddings-inference-1.2.3-x86_64-unknown-linux-gnu.tar.gz && "
        "chmod +x text-embeddings-inference && "
        "mv text-embeddings-inference /usr/local/bin/"
    )
)
```

### Chute Configuration

Configure the service with appropriate GPU and memory requirements:

```python
chute = Chute(
    username="myuser",
    name="text-embeddings-service",
    tagline="High-performance text embeddings and similarity search",
    readme="Production-ready text embedding service with similarity search, reranking, and semantic analysis capabilities",
    image=image,
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16,  # Sufficient for most embedding models
    ),
    concurrency=8,  # Handle multiple concurrent requests
)
```

### Model Initialization

Initialize the embedding model and TEI server:

```python
import subprocess
import time
import requests
from loguru import logger

@chute.on_startup()
async def initialize_embeddings(self):
    """
    Initialize TEI server and embedding capabilities.
    """
    import torch
    import numpy as np
    from sentence_transformers import SentenceTransformer

    # Model configuration
    self.model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Default model
    self.tei_port = 8080
    self.tei_url = f"http://localhost:{self.tei_port}"

    # Start TEI server in background
    logger.info("Starting TEI server...")
    self.tei_process = subprocess.Popen([
        "text-embeddings-inference",
        "--model-id", self.model_name,
        "--port", str(self.tei_port),
        "--max-concurrent-requests", "32",
        "--max-batch-tokens", "16384",
        "--max-batch-requests", "16"
    ])

    # Wait for server to start
    max_wait = 120
    for i in range(max_wait):
        try:
            response = requests.get(f"{self.tei_url}/health", timeout=5)
            if response.status_code == 200:
                logger.success("TEI server started successfully")
                break
        except requests.exceptions.RequestException:
            if i < max_wait - 1:
                time.sleep(1)
            else:
                raise Exception("TEI server failed to start")

    # Initialize fallback model for local processing
    logger.info("Loading fallback sentence transformer...")
    self.sentence_transformer = SentenceTransformer(self.model_name)

    # Store utilities
    self.torch = torch
    self.numpy = np

    # Initialize vector storage (in-memory for this example)
    self.vector_store = {}
    self.text_store = {}

    # Warmup
    await self._warmup_model()

async def _warmup_model(self):
    """Perform warmup embedding generation."""
    warmup_texts = [
        "This is a warmup sentence to initialize the embedding model.",
        "Another test sentence for model warming.",
        "Final warmup text to ensure optimal performance."
    ]

    try:
        # Warmup TEI server
        response = requests.post(
            f"{self.tei_url}/embed",
            json={"inputs": warmup_texts},
            timeout=30
        )
        if response.status_code == 200:
            logger.info("TEI server warmed up successfully")
        else:
            logger.warning("TEI warmup failed, using fallback model")
            # Warmup fallback model
            _ = self.sentence_transformer.encode(warmup_texts)

    except Exception as e:
        logger.warning(f"Warmup failed: {e}, using fallback model")
        _ = self.sentence_transformer.encode(warmup_texts)
```

### Core Embedding Functions

Implement core embedding functionality:

```python
import hashlib
from typing import List, Dict, Tuple

async def get_embeddings(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
    """
    Get embeddings for text(s) using TEI server or fallback.
    """
    if isinstance(texts, str):
        texts = [texts]

    try:
        # Try TEI server first
        response = requests.post(
            f"{self.tei_url}/embed",
            json={
                "inputs": texts,
                "normalize": normalize,
                "truncate": True
            },
            timeout=30
        )

        if response.status_code == 200:
            embeddings = self.numpy.array(response.json())
            return embeddings
        else:
            logger.warning(f"TEI server error: {response.status_code}, using fallback")

    except Exception as e:
        logger.warning(f"TEI server failed: {e}, using fallback")

    # Fallback to local model
    embeddings = self.sentence_transformer.encode(
        texts,
        normalize_embeddings=normalize,
        convert_to_numpy=True
    )
    return embeddings

def compute_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between embeddings."""

    # Normalize if not already normalized
    if embeddings1.ndim == 1:
        embeddings1 = embeddings1.reshape(1, -1)
    if embeddings2.ndim == 1:
        embeddings2 = embeddings2.reshape(1, -1)

    # Compute cosine similarity
    dot_product = self.numpy.dot(embeddings1, embeddings2.T)
    norms1 = self.numpy.linalg.norm(embeddings1, axis=1, keepdims=True)
    norms2 = self.numpy.linalg.norm(embeddings2, axis=1, keepdims=True)

    similarities = dot_product / (norms1 * norms2.T)
    return similarities

def add_to_vector_store(self, texts: List[str], embeddings: np.ndarray, collection: str = "default"):
    """Add texts and embeddings to vector store."""

    if collection not in self.vector_store:
        self.vector_store[collection] = []
        self.text_store[collection] = []

    for text, embedding in zip(texts, embeddings):
        text_id = hashlib.md5(text.encode()).hexdigest()

        self.vector_store[collection].append({
            "id": text_id,
            "embedding": embedding,
            "text": text
        })
        self.text_store[collection].append(text)
```

### Embedding Generation Endpoints

Create endpoints for different embedding operations:

```python
from fastapi import HTTPException

@chute.cord(
    public_api_path="/embed",
    public_api_method="POST",
    stream=False)
async def generate_embeddings(self, args: EmbeddingInput) -> Dict:
    """
    Generate embeddings for input text(s).
    """
    try:
        embeddings = await get_embeddings(self, args.inputs, args.normalize)

        # Convert to list for JSON serialization
        embeddings_list = embeddings.tolist()

        if isinstance(args.inputs, str):
            return {
                "embeddings": embeddings_list[0],
                "model": self.model_name,
                "dimension": len(embeddings_list[0])
            }
        else:
            return {
                "embeddings": embeddings_list,
                "model": self.model_name,
                "dimension": len(embeddings_list[0]),
                "count": len(embeddings_list)
            }

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@chute.cord(
    public_api_path="/similarity",
    public_api_method="POST",
    stream=False)
async def compute_text_similarity(self, args: SimilarityInput) -> Dict:
    """
    Compute similarity between source text and target texts.
    """
    try:
        # Get embeddings for all texts
        all_texts = [args.source_text] + args.target_texts
        embeddings = await get_embeddings(self, all_texts, args.normalize)

        # Separate source and target embeddings
        source_embedding = embeddings[0:1]
        target_embeddings = embeddings[1:]

        # Compute similarities
        similarities = compute_similarity(self, source_embedding, target_embeddings)
        similarity_scores = similarities[0].tolist()

        # Create results with metadata
        results = []
        for i, (text, score) in enumerate(zip(args.target_texts, similarity_scores)):
            results.append({
                "text": text,
                "similarity": float(score),
                "rank": i + 1
            })

        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)

        # Update ranks
        for i, result in enumerate(results):
            result["rank"] = i + 1

        return {
            "source_text": args.source_text,
            "results": results,
            "model": self.model_name
        }

    except Exception as e:
        logger.error(f"Similarity computation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity computation failed: {str(e)}")

@chute.cord(
    public_api_path="/rerank",
    public_api_method="POST",
    stream=False)
async def rerank_texts(self, args: RerankInput) -> Dict:
    """
    Rerank texts based on relevance to query.
    """
    try:
        # Get embeddings
        query_embedding = await get_embeddings(self, args.query, normalize=True)
        text_embeddings = await get_embeddings(self, args.texts, normalize=True)

        # Compute similarities
        similarities = compute_similarity(self, query_embedding, text_embeddings)
        scores = similarities[0].tolist()

        # Create scored results
        scored_texts = [
            {
                "text": text,
                "score": float(score),
                "index": i
            }
            for i, (text, score) in enumerate(zip(args.texts, scores))
        ]

        # Sort by score (descending)
        scored_texts.sort(key=lambda x: x["score"], reverse=True)

        # Apply top_k limit if specified
        if args.top_k:
            scored_texts = scored_texts[:args.top_k]

        # Add ranks
        for rank, item in enumerate(scored_texts):
            item["rank"] = rank + 1

        return {
            "query": args.query,
            "results": scored_texts,
            "total_results": len(scored_texts),
            "model": self.model_name
        }

    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")
```

### Semantic Search Implementation

Build a complete semantic search system:

```python
@chute.cord(
    public_api_path="/search",
    public_api_method="POST",
    stream=False)
async def semantic_search(self, args: SearchInput) -> Dict:
    """
    Perform semantic search over a corpus of texts.
    """
    try:
        # Get query embedding
        query_embedding = await get_embeddings(self, args.query, normalize=True)

        # Get corpus embeddings (batch processing for efficiency)
        corpus_embeddings = await get_embeddings(self, args.corpus, normalize=True)

        # Compute similarities
        similarities = compute_similarity(self, query_embedding, corpus_embeddings)
        scores = similarities[0]

        # Create results with scores
        results = []
        for i, (text, score) in enumerate(zip(args.corpus, scores)):
            if args.threshold is None or score >= args.threshold:
                results.append({
                    "text": text,
                    "score": float(score),
                    "corpus_index": i
                })

        # Sort by score (descending) and take top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:args.top_k]

        # Add ranks
        for rank, result in enumerate(results):
            result["rank"] = rank + 1

        return {
            "query": args.query,
            "results": results,
            "total_corpus_size": len(args.corpus),
            "results_returned": len(results),
            "model": self.model_name,
            "threshold": args.threshold
        }

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")
```

## Advanced Features

### Vector Store Management

Implement persistent vector storage:

```python
class VectorStoreInput(BaseModel):
    collection: str = "default"
    texts: List[str]
    metadata: Optional[Dict] = None

class SearchStoreInput(BaseModel):
    collection: str = "default"
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    filter_metadata: Optional[Dict] = None

@chute.cord(public_api_path="/store/add", method="POST")
async def add_to_store(self, args: VectorStoreInput) -> Dict:
    """Add texts to persistent vector store."""

    try:
        # Generate embeddings
        embeddings = await get_embeddings(self, args.texts, normalize=True)

        # Add to store
        add_to_vector_store(self, args.texts, embeddings, args.collection)

        return {
            "collection": args.collection,
            "added_count": len(args.texts),
            "total_in_collection": len(self.text_store.get(args.collection, []))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add to store: {str(e)}")

@chute.cord(public_api_path="/store/search", method="POST")
async def search_store(self, args: SearchStoreInput) -> Dict:
    """Search within a specific collection."""

    if args.collection not in self.vector_store:
        raise HTTPException(status_code=404, detail=f"Collection '{args.collection}' not found")

    try:
        # Get query embedding
        query_embedding = await get_embeddings(self, args.query, normalize=True)

        # Get stored embeddings
        stored_items = self.vector_store[args.collection]
        stored_embeddings = self.numpy.array([item["embedding"] for item in stored_items])

        # Compute similarities
        similarities = compute_similarity(self, query_embedding, stored_embeddings)
        scores = similarities[0]

        # Create results
        results = []
        for i, (item, score) in enumerate(zip(stored_items, scores)):
            results.append({
                "text": item["text"],
                "score": float(score),
                "id": item["id"]
            })

        # Sort and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:args.top_k]

        # Add ranks
        for rank, result in enumerate(results):
            result["rank"] = rank + 1

        return {
            "collection": args.collection,
            "query": args.query,
            "results": results,
            "total_in_collection": len(stored_items)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Store search failed: {str(e)}")

@chute.cord(public_api_path="/store/collections", method="GET")
async def list_collections(self) -> Dict:
    """List all available collections."""

    collections = []
    for name, texts in self.text_store.items():
        collections.append({
            "name": name,
            "size": len(texts),
            "sample_texts": texts[:3] if texts else []
        })

    return {"collections": collections}
```

### Batch Processing Optimization

Optimize for large-scale batch operations:

```python
class BatchEmbeddingInput(BaseModel):
    texts: List[str] = Field(max_items=1000)
    batch_size: int = Field(default=32, ge=1, le=128)
    normalize: bool = True

@chute.cord(public_api_path="/embed/batch", method="POST")
async def batch_embeddings(self, args: BatchEmbeddingInput) -> Dict:
    """Process large batches of texts efficiently."""

    try:
        all_embeddings = []
        processed_count = 0

        # Process in batches
        for i in range(0, len(args.texts), args.batch_size):
            batch_texts = args.texts[i:i + args.batch_size]
            batch_embeddings = await get_embeddings(self, batch_texts, args.normalize)
            all_embeddings.extend(batch_embeddings.tolist())
            processed_count += len(batch_texts)

            # Optional: yield progress for very large batches
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count}/{len(args.texts)} texts")

        return {
            "embeddings": all_embeddings,
            "processed_count": processed_count,
            "batch_size": args.batch_size,
            "model": self.model_name,
            "dimension": len(all_embeddings[0]) if all_embeddings else 0
        }

    except Exception as e:
        logger.error(f"Batch embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
```

### Clustering and Analysis

Add text clustering capabilities:

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class ClusterInput(BaseModel):
    texts: List[str] = Field(min_items=2, max_items=500)
    n_clusters: int = Field(default=5, ge=2, le=20)
    method: str = Field(default="kmeans")

@chute.cord(public_api_path="/cluster", method="POST")
async def cluster_texts(self, args: ClusterInput) -> Dict:
    """Cluster texts based on semantic similarity."""

    try:
        # Get embeddings
        embeddings = await get_embeddings(self, args.texts, normalize=True)

        # Perform clustering
        if args.method == "kmeans":
            # Adjust number of clusters if needed
            n_clusters = min(args.n_clusters, len(args.texts))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Get cluster centers
            cluster_centers = kmeans.cluster_centers_

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported clustering method: {args.method}")

        # Organize results by cluster
        clusters = {}
        for i, (text, label) in enumerate(zip(args.texts, cluster_labels)):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                "text": text,
                "index": i
            })

        # Calculate cluster statistics
        cluster_stats = []
        for label, items in clusters.items():
            # Find centroid text (closest to cluster center)
            cluster_embeddings = embeddings[[item["index"] for item in items]]
            center = cluster_centers[label]

            # Compute distances to center
            distances = self.numpy.linalg.norm(cluster_embeddings - center, axis=1)
            centroid_idx = self.numpy.argmin(distances)

            cluster_stats.append({
                "cluster_id": label,
                "size": len(items),
                "centroid_text": items[centroid_idx]["text"],
                "texts": [item["text"] for item in items]
            })

        return {
            "clusters": cluster_stats,
            "n_clusters": len(clusters),
            "method": args.method,
            "total_texts": len(args.texts)
        }

    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")
```

## Deployment and Usage

### Deploy the Service

```bash
# Build and deploy the embeddings service
chutes deploy my_embeddings:chute

# Monitor the deployment
chutes chutes get my-embeddings
```

### Using the API

#### Basic Embedding Generation

```bash
curl -X POST "https://myuser-my-embeddings.chutes.ai/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": "This is a sample text for embedding generation",
    "normalize": true
  }'
```

#### Similarity Search

```bash
curl -X POST "https://myuser-my-embeddings.chutes.ai/similarity" \
  -H "Content-Type: application/json" \
  -d '{
    "source_text": "machine learning algorithms",
    "target_texts": [
      "artificial intelligence techniques",
      "cooking recipes",
      "neural network models",
      "gardening tips",
      "deep learning frameworks"
    ],
    "normalize": true
  }'
```

#### Python Client Example

```python
import requests
from typing import List, Dict, Optional

class EmbeddingsClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    def embed(self, texts: Union[str, List[str]], normalize: bool = True) -> Dict:
        """Generate embeddings for text(s)."""
        response = requests.post(
            f"{self.base_url}/embed",
            json={
                "inputs": texts,
                "normalize": normalize
            }
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Embedding failed: {response.status_code} - {response.text}")

    def similarity(self, source_text: str, target_texts: List[str]) -> Dict:
        """Compute similarity between source and target texts."""
        response = requests.post(
            f"{self.base_url}/similarity",
            json={
                "source_text": source_text,
                "target_texts": target_texts,
                "normalize": True
            }
        )
        return response.json()

    def search(self, query: str, corpus: List[str], top_k: int = 10) -> Dict:
        """Perform semantic search over corpus."""
        response = requests.post(
            f"{self.base_url}/search",
            json={
                "query": query,
                "corpus": corpus,
                "top_k": top_k
            }
        )
        return response.json()

    def rerank(self, query: str, texts: List[str], top_k: Optional[int] = None) -> Dict:
        """Rerank texts by relevance to query."""
        payload = {
            "query": query,
            "texts": texts
        }
        if top_k:
            payload["top_k"] = top_k

        response = requests.post(
            f"{self.base_url}/rerank",
            json=payload
        )
        return response.json()

    def add_to_store(self, texts: List[str], collection: str = "default") -> Dict:
        """Add texts to vector store."""
        response = requests.post(
            f"{self.base_url}/store/add",
            json={
                "texts": texts,
                "collection": collection
            }
        )
        return response.json()

    def search_store(self, query: str, collection: str = "default", top_k: int = 10) -> Dict:
        """Search within stored collection."""
        response = requests.post(
            f"{self.base_url}/store/search",
            json={
                "query": query,
                "collection": collection,
                "top_k": top_k
            }
        )
        return response.json()

    def cluster(self, texts: List[str], n_clusters: int = 5) -> Dict:
        """Cluster texts by semantic similarity."""
        response = requests.post(
            f"{self.base_url}/cluster",
            json={
                "texts": texts,
                "n_clusters": n_clusters,
                "method": "kmeans"
            }
        )
        return response.json()

# Usage examples
client = EmbeddingsClient("https://myuser-my-embeddings.chutes.ai")

# Generate embeddings
result = client.embed("This is a test sentence")
embedding = result["embeddings"]
print(f"Embedding dimension: {result['dimension']}")

# Batch embeddings
batch_result = client.embed([
    "First document about machine learning",
    "Second document about cooking",
    "Third document about artificial intelligence"
])

# Find similar texts
similarity_result = client.similarity(
    source_text="artificial intelligence research",
    target_texts=[
        "machine learning algorithms",
        "cooking recipes",
        "neural networks",
        "gardening techniques"
    ]
)

print("Most similar texts:")
for result in similarity_result["results"][:3]:
    print(f"- {result['text']} (similarity: {result['similarity']:.3f})")

# Build a knowledge base
documents = [
    "Python is a programming language",
    "Machine learning uses algorithms to learn patterns",
    "Deep learning is a subset of machine learning",
    "Natural language processing analyzes text",
    "Computer vision processes images",
    "Reinforcement learning learns through trial and error"
]

# Add to vector store
client.add_to_store(documents, collection="ai_knowledge")

# Search the knowledge base
search_result = client.search_store(
    query="algorithms for learning",
    collection="ai_knowledge",
    top_k=3
)

print("Knowledge base search results:")
for result in search_result["results"]:
    print(f"- {result['text']} (score: {result['score']:.3f})")

# Cluster documents
cluster_result = client.cluster(documents, n_clusters=3)
print(f"Clustered into {cluster_result['n_clusters']} groups:")
for cluster in cluster_result["clusters"]:
    print(f"Cluster {cluster['cluster_id']} ({cluster['size']} items):")
    print(f"  Centroid: {cluster['centroid_text']}")
```

## Best Practices

### 1. Model Selection

```python
# Different models for different use cases
model_recommendations = {
    "general_purpose": "sentence-transformers/all-MiniLM-L6-v2",  # Fast, good quality
    "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "high_quality": "sentence-transformers/all-mpnet-base-v2",  # Best quality
    "domain_specific": "sentence-transformers/allenai-specter",  # Scientific papers
    "code": "microsoft/codebert-base",  # Code similarity
}

def select_model_for_use_case(use_case: str) -> str:
    """Select optimal model based on use case."""
    return model_recommendations.get(use_case, model_recommendations["general_purpose"])
```

### 2. Text Preprocessing

```python
import re
from typing import List

def preprocess_text(text: str) -> str:
    """Clean and prepare text for embedding."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters if needed
    text = re.sub(r'[^\w\s\-\.]', '', text)

    # Normalize case (optional, depends on model)
    # text = text.lower()

    # Remove very short texts
    if len(text.strip()) < 3:
        return ""

    return text.strip()

def batch_preprocess(texts: List[str]) -> List[str]:
    """Preprocess batch of texts."""
    processed = []
    for text in texts:
        cleaned = preprocess_text(text)
        if cleaned:  # Only add non-empty texts
            processed.append(cleaned)

    return processed
```

### 3. Caching and Performance

```python
import hashlib
from typing import Dict
import pickle

class EmbeddingCache:
    """Simple LRU cache for embeddings."""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, np.ndarray] = {}
        self.access_order = []
        self.max_size = max_size

    def get_key(self, text: str, model: str) -> str:
        """Generate cache key."""
        content = f"{text}_{model}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, model: str) -> Optional[np.ndarray]:
        """Get cached embedding."""
        key = self.get_key(text, model)
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def set(self, text: str, model: str, embedding: np.ndarray):
        """Cache embedding."""
        key = self.get_key(text, model)

        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]

        self.cache[key] = embedding
        if key not in self.access_order:
            self.access_order.append(key)

# Add to chute initialization
@chute.on_startup()
async def initialize_with_cache(self):
    # ... existing initialization ...
    self.embedding_cache = EmbeddingCache(max_size=2000)

async def get_embeddings_cached(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
    """Get embeddings with caching."""
    if isinstance(texts, str):
        texts = [texts]

    cached_embeddings = []
    uncached_texts = []
    uncached_indices = []

    # Check cache
    for i, text in enumerate(texts):
        cached = self.embedding_cache.get(text, self.model_name)
        if cached is not None:
            cached_embeddings.append((i, cached))
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)

    # Generate uncached embeddings
    if uncached_texts:
        new_embeddings = await get_embeddings(self, uncached_texts, normalize)

        # Cache new embeddings
        for text, embedding in zip(uncached_texts, new_embeddings):
            self.embedding_cache.set(text, self.model_name, embedding)

        # Combine cached and new embeddings
        all_embeddings = [None] * len(texts)

        # Place cached embeddings
        for orig_idx, embedding in cached_embeddings:
            all_embeddings[orig_idx] = embedding

        # Place new embeddings
        for new_idx, orig_idx in enumerate(uncached_indices):
            all_embeddings[orig_idx] = new_embeddings[new_idx]

        return self.numpy.array(all_embeddings)

    else:
        # All cached
        return self.numpy.array([emb for _, emb in sorted(cached_embeddings)])
```

### 4. Error Handling and Monitoring

```python
import time
from loguru import logger

@chute.cord(public_api_path="/robust_embed", method="POST")
async def robust_embeddings(self, args: EmbeddingInput) -> Dict:
    """Embeddings with comprehensive error handling."""

    start_time = time.time()

    try:
        # Validate input
        if isinstance(args.inputs, list) and len(args.inputs) > 1000:
            raise HTTPException(
                status_code=400,
                detail="Batch size too large. Maximum 1000 texts allowed."
            )

        # Preprocess texts
        if isinstance(args.inputs, str):
            processed_texts = preprocess_text(args.inputs)
            if not processed_texts:
                raise HTTPException(status_code=400, detail="Text too short after preprocessing")
        else:
            processed_texts = batch_preprocess(args.inputs)
            if not processed_texts:
                raise HTTPException(status_code=400, detail="No valid texts after preprocessing")

        # Generate embeddings with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                embeddings = await get_embeddings_cached(self, processed_texts, args.normalize)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                time.sleep(1)

        generation_time = time.time() - start_time
        logger.info(f"Embedding generation completed in {generation_time:.2f}s")

        # Return results
        embeddings_list = embeddings.tolist()
        return {
            "embeddings": embeddings_list if isinstance(args.inputs, list) else embeddings_list[0],
            "model": self.model_name,
            "dimension": len(embeddings_list[0]),
            "generation_time": generation_time,
            "processed_count": len(processed_texts)
        }

    except HTTPException:
        raise
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"Embedding generation failed after {error_time:.2f}s: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Embedding generation failed: {str(e)}"
        )
```

## Performance Optimization

### Batch Size Tuning

```python
def get_optimal_batch_size(text_lengths: List[int], max_tokens: int = 16384) -> int:
    """Calculate optimal batch size based on text lengths."""

    # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
    estimated_tokens = [length // 4 for length in text_lengths]

    # Calculate how many texts can fit in max_tokens
    cumulative_tokens = 0
    optimal_batch = 0

    for tokens in estimated_tokens:
        if cumulative_tokens + tokens <= max_tokens:
            cumulative_tokens += tokens
            optimal_batch += 1
        else:
            break

    return max(1, optimal_batch)
```

### Memory Management

```python
async def memory_efficient_embeddings(self, texts: List[str], max_batch_size: int = 32) -> np.ndarray:
    """Generate embeddings with memory management."""

    all_embeddings = []

    for i in range(0, len(texts), max_batch_size):
        batch = texts[i:i + max_batch_size]

        # Clear cache before each batch
        if hasattr(self, 'torch'):
            self.torch.cuda.empty_cache()

        batch_embeddings = await get_embeddings(self, batch, normalize=True)
        all_embeddings.extend(batch_embeddings)

        # Optional: yield progress
        if (i + max_batch_size) % 100 == 0:
            logger.info(f"Processed {min(i + max_batch_size, len(texts))}/{len(texts)} texts")

    return self.numpy.array(all_embeddings)
```

## Next Steps

- **Fine-tuning**: Train custom embedding models on domain-specific data
- **Advanced Search**: Implement hybrid search (dense + sparse)
- **Real-time Updates**: Build dynamic vector databases
- **Multimodal**: Extend to image and audio embeddings

For more advanced examples, see:

- [Custom Training](/docs/examples/custom-training)
- [Vector Databases](/docs/examples/vector-databases)
- [RAG Applications](/docs/examples/rag-applications)
