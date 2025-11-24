# Semantic Search with Text Embeddings

This guide demonstrates how to build a complete semantic search application using text embeddings with Chutes. We'll create a search system that understands meaning, not just keywords.

## Overview

Semantic search enables:

- **Meaning-based Search**: Find documents based on meaning, not just exact keywords
- **Similarity Matching**: Discover related content even with different wording
- **Multi-language Support**: Search across different languages
- **Contextual Understanding**: Understand context and intent in queries
- **Scalable Indexing**: Handle large document collections efficiently

## Quick Start

### Basic Semantic Search Service

```python
from chutes.chute import Chute, NodeSelector
from chutes.chute.template.tei import build_tei_chute

# Create text embedding service
embedding_chute = build_tei_chute(
    username="myuser",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=4
    ),
    concurrency=8
)

print("Deploying embedding service...")
result = embedding_chute.deploy()
print(f"✅ Embedding service deployed: {result}")
```

### Search Application

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

class Document(BaseModel):
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

class SearchQuery(BaseModel):
    query: str
    max_results: int = Field(default=10, le=100)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)

class SearchResult(BaseModel):
    document: Document
    similarity_score: float
    rank: int

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_matches: int
    search_time_ms: float

class SemanticSearchEngine:
    def __init__(self, embedding_chute_url: str):
        self.embedding_chute_url = embedding_chute_url
        self.documents: List[Document] = []
        self.embeddings_matrix = None

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using TEI service"""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.embedding_chute_url}/embed",
                json={"inputs": text}
            )
            response.raise_for_status()
            return response.json()[0]

    async def add_document(self, document: Document) -> None:
        """Add document to search index"""
        if document.embedding is None:
            document.embedding = await self.embed_text(document.content)

        self.documents.append(document)
        self._update_embeddings_matrix()

    async def add_documents(self, documents: List[Document]) -> None:
        """Add multiple documents efficiently"""
        # Generate embeddings for documents without them
        texts_to_embed = []
        doc_indices = []

        for i, doc in enumerate(documents):
            if doc.embedding is None:
                texts_to_embed.append(doc.content)
                doc_indices.append(i)

        if texts_to_embed:
            embeddings = await self._embed_batch(texts_to_embed)
            for idx, embedding in zip(doc_indices, embeddings):
                documents[idx].embedding = embedding

        self.documents.extend(documents)
        self._update_embeddings_matrix()

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.embedding_chute_url}/embed",
                json={"inputs": texts}
            )
            response.raise_for_status()
            return response.json()

    def _update_embeddings_matrix(self):
        """Update the embeddings matrix for similarity search"""
        if self.documents:
            embeddings = [doc.embedding for doc in self.documents]
            self.embeddings_matrix = np.array(embeddings)

    async def search(self, query: SearchQuery) -> SearchResponse:
        """Perform semantic search"""
        import time
        start_time = time.time()

        # Generate query embedding
        query_embedding = await self.embed_text(query.query)
        query_vector = np.array(query_embedding).reshape(1, -1)

        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.embeddings_matrix)[0]

        # Apply similarity threshold
        valid_indices = np.where(similarities >= query.similarity_threshold)[0]
        valid_similarities = similarities[valid_indices]

        # Sort by similarity (descending)
        sorted_indices = valid_indices[np.argsort(valid_similarities)[::-1]]

        # Apply filters and limit results
        results = []
        for rank, idx in enumerate(sorted_indices[:query.max_results]):
            document = self.documents[idx]

            # Apply filters if specified
            if query.filters and not self._apply_filters(document, query.filters):
                continue

            results.append(SearchResult(
                document=document,
                similarity_score=float(similarities[idx]),
                rank=rank + 1
            ))

        search_time = (time.time() - start_time) * 1000

        return SearchResponse(
            query=query.query,
            results=results,
            total_matches=len(results),
            search_time_ms=search_time
        )

    def _apply_filters(self, document: Document, filters: Dict[str, Any]) -> bool:
        """Apply metadata filters to document"""
        for key, value in filters.items():
            if key not in document.metadata:
                return False
            if document.metadata[key] != value:
                return False
        return True

# Global search engine instance
search_engine = None

async def initialize_search_engine(embedding_url: str, documents_data: List[Dict] = None):
    """Initialize the search engine with documents"""
    global search_engine
    search_engine = SemanticSearchEngine(embedding_url)

    if documents_data:
        documents = [Document(**doc_data) for doc_data in documents_data]
        await search_engine.add_documents(documents)

async def run(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Main search service entry point"""
    global search_engine

    action = inputs.get("action", "search")

    if action == "initialize":
        embedding_url = inputs["embedding_service_url"]
        documents_data = inputs.get("documents", [])
        await initialize_search_engine(embedding_url, documents_data)
        return {"status": "initialized", "document_count": len(documents_data)}

    elif action == "add_document":
        document_data = inputs["document"]
        document = Document(**document_data)
        await search_engine.add_document(document)
        return {"status": "added", "document_id": document.id}

    elif action == "add_documents":
        documents_data = inputs["documents"]
        documents = [Document(**doc_data) for doc_data in documents_data]
        await search_engine.add_documents(documents)
        return {"status": "added", "count": len(documents)}

    elif action == "search":
        query_data = inputs["query"]
        query = SearchQuery(**query_data)
        response = await search_engine.search(query)
        return response.dict()

    else:
        raise ValueError(f"Unknown action: {action}")
```

## Complete Example Implementation

### Document Indexing Service

```python
from chutes.image import Image
from chutes.chute import Chute, NodeSelector

# Custom image with search dependencies
search_image = (
    Image(
        username="myuser",
        name="semantic-search",
        tag="1.0.0",
        python_version="3.11"
    )
    .pip_install([
        "scikit-learn==1.3.0",
        "numpy==1.24.3",
        "httpx==0.25.0",
        "pydantic==2.4.2",
        "fastapi==0.104.1",
        "uvicorn==0.24.0"
    ])
)

# Deploy search service
search_chute = Chute(
    username="myuser",
    name="semantic-search-service",
    image=search_image,
    entry_file="search_engine.py",
    entry_point="run",
    node_selector=NodeSelector(
        gpu_count=0,  # CPU-only for search logic),
    timeout_seconds=300,
    concurrency=10
)

result = search_chute.deploy()
print(f"Search service deployed: {result}")
```

### Usage Examples

#### Initialize with Documents

```python
# Sample documents
documents = [
    {
        "id": "doc1",
        "content": "Artificial intelligence is transforming healthcare through machine learning algorithms.",
        "metadata": {"category": "healthcare", "author": "Dr. Smith", "year": 2024}
    },
    {
        "id": "doc2",
        "content": "Machine learning models can predict patient outcomes with high accuracy.",
        "metadata": {"category": "healthcare", "author": "Dr. Johnson", "year": 2024}
    },
    {
        "id": "doc3",
        "content": "Climate change affects global weather patterns and ocean temperatures.",
        "metadata": {"category": "environment", "author": "Prof. Green", "year": 2023}
    }
]

# Initialize search service
response = search_chute.run({
    "action": "initialize",
    "embedding_service_url": "https://your-embedding-service.com",
    "documents": documents
})
print(f"Initialized: {response}")
```

#### Perform Searches

```python
# Search for healthcare AI content
search_response = search_chute.run({
    "action": "search",
    "query": {
        "query": "AI in medical diagnosis",
        "max_results": 5,
        "similarity_threshold": 0.6,
        "filters": {"category": "healthcare"}
    }
})

print(f"Found {search_response['total_matches']} results:")
for result in search_response['results']:
    print(f"- {result['document']['id']}: {result['similarity_score']:.3f}")
```

#### Add New Documents

```python
# Add new document to index
new_doc = {
    "id": "doc4",
    "content": "Deep learning networks excel at image recognition tasks in medical imaging.",
    "metadata": {"category": "healthcare", "author": "Dr. Lee", "year": 2024}
}

response = search_chute.run({
    "action": "add_document",
    "document": new_doc
})
print(f"Added document: {response}")
```

## Advanced Features

### Multi-Modal Search

```python
class MultiModalDocument(BaseModel):
    id: str
    text_content: str
    image_path: Optional[str] = None
    text_embedding: Optional[List[float]] = None
    image_embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MultiModalSearchEngine(SemanticSearchEngine):
    def __init__(self, text_embedding_url: str, image_embedding_url: str):
        super().__init__(text_embedding_url)
        self.image_embedding_url = image_embedding_url

    async def embed_image(self, image_path: str) -> List[float]:
        """Generate embedding for image using CLIP or similar"""
        import httpx

        async with httpx.AsyncClient() as client:
            with open(image_path, "rb") as f:
                files = {"image": f}
                response = await client.post(
                    f"{self.image_embedding_url}/embed",
                    files=files
                )
            response.raise_for_status()
            return response.json()

    async def hybrid_search(self, text_query: str, image_query: str = None,
                          text_weight: float = 0.7) -> SearchResponse:
        """Perform hybrid text + image search"""
        text_embedding = await self.embed_text(text_query)

        if image_query:
            image_embedding = await self.embed_image(image_query)
            # Combine embeddings with weights
            combined_embedding = (
                np.array(text_embedding) * text_weight +
                np.array(image_embedding) * (1 - text_weight)
            )
        else:
            combined_embedding = np.array(text_embedding)

        # Perform similarity search with combined embedding
        # Implementation similar to regular search...
```

### Real-time Search API

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Semantic Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

@app.post("/search")
async def search_documents(query: SearchQuery) -> SearchResponse:
    """Search documents endpoint"""
    try:
        return await search_engine.search(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents")
async def add_document(document: Document) -> Dict[str, str]:
    """Add document endpoint"""
    try:
        await search_engine.add_document(document)
        return {"status": "success", "document_id": document.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "documents": len(search_engine.documents)}

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
```

### Vector Database Integration

```python
import chromadb
from chromadb.config import Settings

class VectorDBSearchEngine:
    def __init__(self, embedding_service_url: str):
        self.embedding_service_url = embedding_service_url
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    async def add_documents(self, documents: List[Document]):
        """Add documents to vector database"""
        # Generate embeddings
        texts = [doc.content for doc in documents]
        embeddings = await self._embed_batch(texts)

        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=[doc.metadata for doc in documents],
            ids=[doc.id for doc in documents]
        )

    async def search(self, query: SearchQuery) -> SearchResponse:
        """Search using vector database"""
        query_embedding = await self.embed_text(query.query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=query.max_results,
            where=query.filters if query.filters else None
        )

        # Format response
        search_results = []
        for i, (doc_id, content, metadata, distance) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            similarity = 1 - distance  # Convert distance to similarity
            if similarity >= query.similarity_threshold:
                search_results.append(SearchResult(
                    document=Document(
                        id=doc_id,
                        content=content,
                        metadata=metadata
                    ),
                    similarity_score=similarity,
                    rank=i + 1
                ))

        return SearchResponse(
            query=query.query,
            results=search_results,
            total_matches=len(search_results),
            search_time_ms=0  # ChromaDB handles timing
        )
```

## Production Deployment

### Scalable Architecture

```python
# High-performance embedding service
embedding_chute = build_tei_chute(
    username="mycompany",
    model_name="sentence-transformers/all-mpnet-base-v2",
    node_selector=NodeSelector(
        gpu_count=2,
        min_vram_gb_per_gpu=16,
        preferred_provider="runpod"
    ),
    concurrency=16,
    auto_scale=True,
    min_instances=2,
    max_instances=8
)

# Search service with caching
search_chute = Chute(
    username="mycompany",
    name="search-prod",
    image=search_image,
    entry_file="search_api.py",
    entry_point="app",
    node_selector=NodeSelector(
        gpu_count=0),
    environment={
        "REDIS_URL": "redis://cache.example.com:6379",
        "VECTOR_DB_PATH": "/data/chroma",
        "EMBEDDING_SERVICE_URL": embedding_chute.url
    },
    timeout_seconds=300,
    concurrency=20
)
```

### Performance Monitoring

```python
from prometheus_client import Counter, Histogram, start_http_server
import time

# Metrics
SEARCH_REQUESTS = Counter('search_requests_total', 'Total search requests')
SEARCH_DURATION = Histogram('search_duration_seconds', 'Search duration')
EMBEDDING_CACHE_HITS = Counter('embedding_cache_hits_total', 'Cache hits')

class MonitoredSearchEngine(SemanticSearchEngine):
    async def search(self, query: SearchQuery) -> SearchResponse:
        SEARCH_REQUESTS.inc()

        with SEARCH_DURATION.time():
            return await super().search(query)

    async def embed_text(self, text: str) -> List[float]:
        # Check cache first
        cache_key = f"embed:{hash(text)}"
        cached = await self._get_from_cache(cache_key)

        if cached:
            EMBEDDING_CACHE_HITS.inc()
            return cached

        # Generate new embedding
        embedding = await super().embed_text(text)
        await self._store_in_cache(cache_key, embedding)
        return embedding

# Start metrics server
start_http_server(8001)
```

## Next Steps

- **[Text Embeddings Guide](../templates/tei)** - Deep dive into embedding models
- **[Vector Databases](vector-databases)** - Scalable vector storage solutions
- **[RAG Applications](rag-applications)** - Retrieval-Augmented Generation
- **[Performance Optimization](../guides/performance)** - Scale your search service

For enterprise-scale deployments, see the [Production Search Architecture](../guides/production-search) guide.
