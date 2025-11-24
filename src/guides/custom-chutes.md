# Building Custom Chutes

This guide walks you through creating custom Chutes from scratch, covering everything from basic setup to advanced patterns for production applications.

## Overview

Custom Chutes give you complete control over your AI application architecture, allowing you to:

- **Build Complex Logic**: Implement sophisticated AI pipelines
- **Custom Dependencies**: Use any Python packages or system libraries
- **Multiple Models**: Combine different AI models in a single service
- **Advanced Processing**: Add preprocessing, postprocessing, and business logic
- **Custom APIs**: Design exactly the endpoints you need

## Basic Custom Chute Structure

### Minimal Example

Here's the simplest possible custom Chute:

```python
from chutes.chute import Chute
from chutes.image import Image

# Create custom image
image = (
    Image(username="myuser", name="my-custom-app", tag="1.0")
    .from_base("python:3.11-slim")
    .run_command("pip install numpy pandas")
)

# Create chute
chute = Chute(
    username="myuser",
    name="my-custom-app",
    image=image
)

@chute.on_startup()
async def initialize(self):
    """Initialize any resources needed by your app."""
    self.message = "Hello from custom chute!"

@chute.cord(public_api_path="/hello", method="GET")
async def hello(self):
    """Simple endpoint that returns a greeting."""
    return {"message": self.message}
```

### Adding Dependencies and Models

```python
from chutes.chute import Chute, NodeSelector
from chutes.image import Image
from pydantic import BaseModel
from typing import List, Optional

# Define input/output schemas
class AnalysisInput(BaseModel):
    text: str
    options: Optional[List[str]] = []

class AnalysisOutput(BaseModel):
    result: str
    confidence: float
    metadata: dict

# Create custom image with AI dependencies
image = (
    Image(username="myuser", name="text-analyzer", tag="1.0")
    .from_base("nvidia/cuda:11.8-devel-ubuntu22.04")
    .run_command("apt update && apt install -y python3 python3-pip")
    .run_command("pip3 install torch transformers tokenizers")
    .run_command("pip3 install numpy pandas scikit-learn")
    .run_command("pip3 install fastapi uvicorn pydantic")
    .set_workdir("/app")
)

# Create chute with GPU support
chute = Chute(
    username="myuser",
    name="text-analyzer",
    image=image,
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=8
    ),
    concurrency=4
)

@chute.on_startup()
async def initialize_models(self):
    """Load AI models during startup."""
    from transformers import pipeline
    import torch

    # Load sentiment analysis model
    self.sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=0 if torch.cuda.is_available() else -1
    )

    # Load text classification model
    self.classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if torch.cuda.is_available() else -1
    )

@chute.cord(
    public_api_path="/analyze",
    method="POST",
    input_schema=AnalysisInput,
    output_schema=AnalysisOutput
)
async def analyze_text(self, input_data: AnalysisInput) -> AnalysisOutput:
    """Analyze text with multiple AI models."""

    # Sentiment analysis
    sentiment_result = self.sentiment_analyzer(input_data.text)[0]

    # Classification (if options provided)
    classification_result = None
    if input_data.options:
        classification_result = self.classifier(
            input_data.text,
            input_data.options
        )

    # Combine results
    result = f"Sentiment: {sentiment_result['label']}"
    if classification_result:
        result += f", Category: {classification_result['labels'][0]}"

    return AnalysisOutput(
        result=result,
        confidence=sentiment_result['score'],
        metadata={
            "sentiment": sentiment_result,
            "classification": classification_result
        }
    )
```

## Advanced Patterns

### Multi-Model Pipeline

```python
from chutes.chute import Chute, NodeSelector
from chutes.image import Image
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import asyncio

class DocumentInput(BaseModel):
    text: str
    analyze_sentiment: bool = True
    extract_entities: bool = True
    summarize: bool = False
    max_summary_length: int = Field(default=150, ge=50, le=500)

class DocumentOutput(BaseModel):
    original_text: str
    sentiment: Optional[Dict[str, Any]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    summary: Optional[str] = None
    processing_time: float

# Advanced image with multiple AI libraries
image = (
    Image(username="myuser", name="document-processor", tag="2.0")
    .from_base("nvidia/cuda:11.8-devel-ubuntu22.04")
    .run_command("apt update && apt install -y python3 python3-pip git")
    .run_command("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    .run_command("pip3 install transformers tokenizers")
    .run_command("pip3 install spacy")
    .run_command("python3 -m spacy download en_core_web_sm")
    .run_command("pip3 install sumy nltk")
    .run_command("pip3 install asyncio aiofiles")
    .set_workdir("/app")
)

chute = Chute(
    username="myuser",
    name="document-processor",
    image=image,
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16
    ),
    concurrency=6
)

@chute.on_startup()
async def initialize_pipeline(self):
    """Initialize multiple AI models for document processing."""
    from transformers import pipeline
    import spacy
    import torch
    import time

    self.device = 0 if torch.cuda.is_available() else -1

    # Load models
    print("Loading sentiment analyzer...")
    self.sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=self.device
    )

    print("Loading NER model...")
    self.ner_model = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        device=self.device,
        aggregation_strategy="simple"
    )

    print("Loading summarization model...")
    self.summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=self.device
    )

    print("Loading spaCy model...")
    self.nlp = spacy.load("en_core_web_sm")

    print("All models loaded successfully!")

async def analyze_sentiment_async(self, text: str) -> Dict[str, Any]:
    """Asynchronous sentiment analysis."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: self.sentiment_analyzer(text)[0]
    )
    return result

async def extract_entities_async(self, text: str) -> List[Dict[str, Any]]:
    """Asynchronous named entity recognition."""
    loop = asyncio.get_event_loop()

    # Use transformers NER
    ner_results = await loop.run_in_executor(
        None,
        lambda: self.ner_model(text)
    )

    # Also use spaCy for additional entity types
    spacy_results = await loop.run_in_executor(
        None,
        lambda: [(ent.text, ent.label_, ent.start_char, ent.end_char)
                for ent in self.nlp(text).ents]
    )

    # Combine results
    entities = []

    # Add transformer results
    for entity in ner_results:
        entities.append({
            "text": entity["word"],
            "label": entity["entity_group"],
            "confidence": entity["score"],
            "start": entity["start"],
            "end": entity["end"],
            "source": "transformers"
        })

    # Add spaCy results
    for text_span, label, start, end in spacy_results:
        entities.append({
            "text": text_span,
            "label": label,
            "confidence": 1.0,  # spaCy doesn't provide confidence
            "start": start,
            "end": end,
            "source": "spacy"
        })

    return entities

async def summarize_async(self, text: str, max_length: int = 150) -> str:
    """Asynchronous text summarization."""
    if len(text.split()) < 50:
        return text  # Too short to summarize

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: self.summarizer(
            text,
            max_length=max_length,
            min_length=30,
            do_sample=False
        )[0]
    )
    return result["summary_text"]

@chute.cord(
    public_api_path="/process",
    method="POST",
    input_schema=DocumentInput,
    output_schema=DocumentOutput
)
async def process_document(self, input_data: DocumentInput) -> DocumentOutput:
    """Process document with multiple AI models in parallel."""
    import time

    start_time = time.time()

    # Create tasks for parallel processing
    tasks = []

    if input_data.analyze_sentiment:
        tasks.append(analyze_sentiment_async(self, input_data.text))
    else:
        tasks.append(asyncio.create_task(asyncio.sleep(0, result=None)))

    if input_data.extract_entities:
        tasks.append(extract_entities_async(self, input_data.text))
    else:
        tasks.append(asyncio.create_task(asyncio.sleep(0, result=None)))

    if input_data.summarize:
        tasks.append(summarize_async(self, input_data.text, input_data.max_summary_length))
    else:
        tasks.append(asyncio.create_task(asyncio.sleep(0, result=None)))

    # Execute all tasks in parallel
    sentiment_result, entities_result, summary_result = await asyncio.gather(*tasks)

    processing_time = time.time() - start_time

    return DocumentOutput(
        original_text=input_data.text,
        sentiment=sentiment_result,
        entities=entities_result,
        summary=summary_result,
        processing_time=processing_time
    )
```

### State Management and Caching

```python
from chutes.chute import Chute
from chutes.image import Image
import asyncio
from typing import Dict, Any, Optional
import hashlib
import json
import time

class StatefulChute(Chute):
    """Custom chute with built-in state management."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
        self.session_data = {}
        self.request_history = []

# Create image with caching dependencies
image = (
    Image(username="myuser", name="stateful-app", tag="1.0")
    .from_base("python:3.11-slim")
    .run_command("pip install redis aioredis")
    .run_command("pip install sqlalchemy aiosqlite")
    .run_command("pip install fastapi uvicorn pydantic")
)

chute = StatefulChute(
    username="myuser",
    name="stateful-app",
    image=image
)

@chute.on_startup()
async def initialize_storage(self):
    """Initialize storage systems."""
    import aioredis

    # In-memory cache
    self.memory_cache = {}
    self.cache_ttl = {}

    # Try to connect to Redis (optional)
    try:
        self.redis = await aioredis.create_redis_pool('redis://localhost')
        self.has_redis = True
    except:
        self.redis = None
        self.has_redis = False
        print("Redis not available, using memory cache only")

    # Session storage
    self.sessions = {}

    # Request tracking
    self.request_count = 0
    self.last_requests = []

async def get_cached(self, key: str) -> Optional[Any]:
    """Get value from cache (Redis or memory)."""

    # Check memory cache first
    if key in self.memory_cache:
        if key in self.cache_ttl and time.time() > self.cache_ttl[key]:
            del self.memory_cache[key]
            del self.cache_ttl[key]
        else:
            return self.memory_cache[key]

    # Check Redis if available
    if self.has_redis:
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
        except:
            pass

    return None

async def set_cached(self, key: str, value: Any, ttl: int = 3600):
    """Set value in cache with TTL."""

    # Store in memory cache
    self.memory_cache[key] = value
    self.cache_ttl[key] = time.time() + ttl

    # Store in Redis if available
    if self.has_redis:
        try:
            await self.redis.setex(key, ttl, json.dumps(value))
        except:
            pass

def get_cache_key(self, data: str, operation: str) -> str:
    """Generate cache key from data and operation."""
    content = f"{operation}:{data}"
    return hashlib.md5(content.encode()).hexdigest()

class ProcessingRequest(BaseModel):
    text: str
    operation: str = "analyze"
    use_cache: bool = True
    session_id: Optional[str] = None

@chute.cord(
    public_api_path="/process_cached",
    method="POST",
    input_schema=ProcessingRequest
)
async def process_with_caching(self, input_data: ProcessingRequest) -> Dict[str, Any]:
    """Process request with caching and session management."""

    # Track request
    self.request_count += 1
    request_info = {
        "timestamp": time.time(),
        "operation": input_data.operation,
        "session_id": input_data.session_id
    }
    self.last_requests.append(request_info)

    # Keep only last 100 requests
    if len(self.last_requests) > 100:
        self.last_requests = self.last_requests[-100:]

    # Check cache
    cache_key = get_cache_key(self, input_data.text, input_data.operation)
    if input_data.use_cache:
        cached_result = await get_cached(self, cache_key)
        if cached_result:
            cached_result["from_cache"] = True
            cached_result["request_id"] = self.request_count
            return cached_result

    # Process request (simulate AI processing)
    await asyncio.sleep(0.1)  # Simulate processing time

    result = {
        "text": input_data.text,
        "operation": input_data.operation,
        "result": f"Processed: {input_data.text[:50]}...",
        "timestamp": time.time(),
        "request_id": self.request_count,
        "from_cache": False
    }

    # Store in cache
    if input_data.use_cache:
        await set_cached(self, cache_key, result, ttl=1800)  # 30 minutes

    # Update session data
    if input_data.session_id:
        if input_data.session_id not in self.sessions:
            self.sessions[input_data.session_id] = {
                "created": time.time(),
                "requests": []
            }

        self.sessions[input_data.session_id]["requests"].append({
            "request_id": self.request_count,
            "operation": input_data.operation,
            "timestamp": time.time()
        })

    return result

@chute.cord(public_api_path="/stats", method="GET")
async def get_stats(self) -> Dict[str, Any]:
    """Get service statistics."""

    cache_size = len(self.memory_cache)
    session_count = len(self.sessions)

    # Recent request stats
    recent_requests = [r for r in self.last_requests
                      if time.time() - r["timestamp"] < 3600]  # Last hour

    operation_counts = {}
    for req in recent_requests:
        op = req["operation"]
        operation_counts[op] = operation_counts.get(op, 0) + 1

    return {
        "total_requests": self.request_count,
        "cache_size": cache_size,
        "session_count": session_count,
        "recent_requests_1h": len(recent_requests),
        "operation_counts": operation_counts,
        "has_redis": self.has_redis
    }
```

### Background Jobs and Queues

```python
from chutes.chute import Chute
from chutes.image import Image
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import uuid
import time
from enum import Enum

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class JobRequest(BaseModel):
    task_type: str
    data: Dict
    priority: int = Field(default=1, ge=1, le=5)

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict] = None
    error: Optional[str] = None

# Create image with job processing capabilities
image = (
    Image(username="myuser", name="job-processor", tag="1.0")
    .from_base("python:3.11-slim")
    .run_command("pip install asyncio aiofiles")
    .run_command("pip install celery redis")  # For advanced job queues
    .run_command("pip install fastapi uvicorn pydantic")
)

chute = Chute(
    username="myuser",
    name="job-processor",
    image=image,
    concurrency=8
)

@chute.on_startup()
async def initialize_job_system(self):
    """Initialize job processing system."""

    # Job storage
    self.jobs = {}
    self.job_queue = asyncio.Queue()

    # Job processing
    self.workers = []
    self.max_workers = 4

    # Start background workers
    for i in range(self.max_workers):
        worker = asyncio.create_task(self.job_worker(f"worker-{i}"))
        self.workers.append(worker)

    print(f"Started {self.max_workers} job workers")

async def job_worker(self, worker_name: str):
    """Background worker to process jobs."""

    while True:
        try:
            # Get job from queue
            job_id = await self.job_queue.get()

            if job_id not in self.jobs:
                continue

            job = self.jobs[job_id]

            # Update job status
            job["status"] = JobStatus.RUNNING
            job["started_at"] = time.time()
            job["worker"] = worker_name

            print(f"{worker_name} processing job {job_id}")

            # Process job based on type
            try:
                if job["task_type"] == "text_analysis":
                    result = await self.process_text_analysis(job["data"])
                elif job["task_type"] == "data_processing":
                    result = await self.process_data(job["data"])
                elif job["task_type"] == "file_conversion":
                    result = await self.process_file_conversion(job["data"])
                else:
                    raise ValueError(f"Unknown task type: {job['task_type']}")

                # Job completed successfully
                job["status"] = JobStatus.COMPLETED
                job["completed_at"] = time.time()
                job["result"] = result

            except Exception as e:
                # Job failed
                job["status"] = JobStatus.FAILED
                job["completed_at"] = time.time()
                job["error"] = str(e)
                print(f"Job {job_id} failed: {e}")

            # Mark task as done
            self.job_queue.task_done()

        except Exception as e:
            print(f"Worker {worker_name} error: {e}")
            await asyncio.sleep(1)

async def process_text_analysis(self, data: Dict) -> Dict:
    """Process text analysis job."""
    text = data.get("text", "")

    # Simulate AI processing
    await asyncio.sleep(2)  # Simulate processing time

    return {
        "text": text,
        "length": len(text),
        "word_count": len(text.split()),
        "analysis": "Text analysis completed"
    }

async def process_data(self, data: Dict) -> Dict:
    """Process data processing job."""
    items = data.get("items", [])

    # Simulate data processing
    await asyncio.sleep(len(items) * 0.1)

    return {
        "processed_items": len(items),
        "total_value": sum(item.get("value", 0) for item in items)
    }

async def process_file_conversion(self, data: Dict) -> Dict:
    """Process file conversion job."""
    file_type = data.get("file_type", "")
    target_type = data.get("target_type", "")

    # Simulate file conversion
    await asyncio.sleep(3)

    return {
        "source_type": file_type,
        "target_type": target_type,
        "status": "converted",
        "file_size": "1.2MB"
    }

@chute.cord(
    public_api_path="/jobs",
    method="POST",
    input_schema=JobRequest
)
async def submit_job(self, job_request: JobRequest) -> Dict[str, str]:
    """Submit a new job for processing."""

    job_id = str(uuid.uuid4())

    # Create job record
    job = {
        "id": job_id,
        "task_type": job_request.task_type,
        "data": job_request.data,
        "priority": job_request.priority,
        "status": JobStatus.PENDING,
        "created_at": time.time(),
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error": None,
        "worker": None
    }

    self.jobs[job_id] = job

    # Add to queue
    await self.job_queue.put(job_id)

    return {"job_id": job_id, "status": "submitted"}

@chute.cord(public_api_path="/jobs/{job_id}", method="GET")
async def get_job_status(self, job_id: str) -> JobResponse:
    """Get status of a specific job."""

    if job_id not in self.jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = self.jobs[job_id]

    return JobResponse(
        job_id=job["id"],
        status=job["status"],
        created_at=job["created_at"],
        started_at=job["started_at"],
        completed_at=job["completed_at"],
        result=job["result"],
        error=job["error"]
    )

@chute.cord(public_api_path="/jobs", method="GET")
async def list_jobs(self, status: Optional[JobStatus] = None, limit: int = 50) -> Dict:
    """List jobs with optional filtering."""

    jobs = list(self.jobs.values())

    # Filter by status if specified
    if status:
        jobs = [job for job in jobs if job["status"] == status]

    # Sort by creation time (newest first)
    jobs.sort(key=lambda x: x["created_at"], reverse=True)

    # Limit results
    jobs = jobs[:limit]

    # Convert to response format
    job_list = []
    for job in jobs:
        job_list.append(JobResponse(
            job_id=job["id"],
            status=job["status"],
            created_at=job["created_at"],
            started_at=job["started_at"],
            completed_at=job["completed_at"],
            result=job["result"],
            error=job["error"]
        ))

    return {
        "jobs": job_list,
        "total": len(job_list),
        "queue_size": self.job_queue.qsize()
    }

# Background job decorator
@chute.job()
async def cleanup_old_jobs(self):
    """Clean up completed jobs older than 24 hours."""

    cutoff_time = time.time() - (24 * 60 * 60)  # 24 hours ago

    jobs_to_remove = []
    for job_id, job in self.jobs.items():
        if (job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED] and
            job["completed_at"] and job["completed_at"] < cutoff_time):
            jobs_to_remove.append(job_id)

    for job_id in jobs_to_remove:
        del self.jobs[job_id]

    if jobs_to_remove:
        print(f"Cleaned up {len(jobs_to_remove)} old jobs")
```

## Best Practices

### 1. Error Handling

```python
from fastapi import HTTPException
import traceback
from loguru import logger

@chute.cord(public_api_path="/robust", method="POST")
async def robust_endpoint(self, input_data: Dict) -> Dict:
    """Endpoint with comprehensive error handling."""

    try:
        # Validate input
        if not input_data.get("text"):
            raise HTTPException(
                status_code=400,
                detail="Missing required field: text"
            )

        # Process with timeout
        result = await asyncio.wait_for(
            self.process_text(input_data["text"]),
            timeout=30.0
        )

        return {"result": result, "status": "success"}

    except asyncio.TimeoutError:
        logger.error("Processing timeout")
        raise HTTPException(
            status_code=408,
            detail="Processing timeout - request took too long"
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )

    except Exception as e:
        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
```

### 2. Resource Management

```python
@chute.on_startup()
async def initialize_with_resource_management(self):
    """Initialize with proper resource management."""
    import torch

    # GPU memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        self.device = torch.device("cuda")

        # Monitor GPU memory
        self.gpu_memory_threshold = 0.9  # 90% usage threshold
    else:
        self.device = torch.device("cpu")

    # Connection pools
    self.session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=100)
    )

    # Resource cleanup tracking
    self.cleanup_tasks = []

@chute.on_shutdown()
async def cleanup_resources(self):
    """Clean up resources on shutdown."""

    # Close HTTP session
    if hasattr(self, 'session'):
        await self.session.close()

    # Cancel background tasks
    for task in self.cleanup_tasks:
        task.cancel()

    # Clear GPU memory
    if hasattr(self, 'device') and self.device.type == 'cuda':
        torch.cuda.empty_cache()

    print("Resources cleaned up successfully")
```

### 3. Monitoring and Metrics

```python
import time
from collections import defaultdict

@chute.on_startup()
async def initialize_metrics(self):
    """Initialize metrics collection."""

    self.metrics = {
        "request_count": 0,
        "error_count": 0,
        "response_times": [],
        "endpoint_usage": defaultdict(int)
    }

    # Start metrics collection task
    self.metrics_task = asyncio.create_task(self.collect_metrics())

async def collect_metrics(self):
    """Background task to collect and log metrics."""

    while True:
        try:
            await asyncio.sleep(60)  # Collect every minute

            if self.metrics["response_times"]:
                avg_response_time = sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
                self.metrics["response_times"] = []  # Reset
            else:
                avg_response_time = 0

            logger.info(f"Metrics - Requests: {self.metrics['request_count']}, "
                       f"Errors: {self.metrics['error_count']}, "
                       f"Avg Response Time: {avg_response_time:.2f}s")

        except Exception as e:
            logger.error(f"Metrics collection error: {e}")

# Decorator for automatic metrics collection
def with_metrics(func):
    """Decorator to automatically collect metrics."""

    async def wrapper(self, *args, **kwargs):
        start_time = time.time()

        try:
            self.metrics["request_count"] += 1
            self.metrics["endpoint_usage"][func.__name__] += 1

            result = await func(self, *args, **kwargs)

            response_time = time.time() - start_time
            self.metrics["response_times"].append(response_time)

            return result

        except Exception as e:
            self.metrics["error_count"] += 1
            raise

    return wrapper

@chute.cord(public_api_path="/monitored", method="POST")
@with_metrics
async def monitored_endpoint(self, input_data: Dict) -> Dict:
    """Endpoint with automatic metrics collection."""

    # Your processing logic here
    await asyncio.sleep(0.1)  # Simulate work

    return {"result": "processed", "input": input_data}
```

## Testing and Development

### Local Testing

```python
# test_custom_chute.py
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

@pytest.mark.asyncio
async def test_chute_initialization():
    """Test chute startup."""

    # Mock the chute
    chute_mock = Mock()
    chute_mock.initialize_models = AsyncMock()

    # Test initialization
    await chute_mock.initialize_models()

    assert chute_mock.initialize_models.called

@pytest.mark.asyncio
async def test_endpoint_functionality():
    """Test endpoint logic."""

    # Create test instance
    chute_instance = Mock()
    chute_instance.process_text = AsyncMock(return_value="processed result")

    # Test data
    test_input = {"text": "test input"}

    # Call function
    result = await chute_instance.process_text(test_input["text"])

    assert result == "processed result"

# Run tests
# pytest test_custom_chute.py -v
```

### Development Workflow

```bash
# 1. Create and test locally
python my_chute.py  # Test locally first

# 2. Build image
chutes build my-custom-app:chute --wait

# 3. Deploy to staging
chutes deploy my-custom-app:chute --wait

# 4. Test deployed service
curl https://myuser-my-custom-app.chutes.ai/hello

# 5. Monitor and iterate
chutes chutes logs my-custom-app
chutes chutes metrics my-custom-app
```

## Advanced Topics

### 1. Custom Middleware

```python
from fastapi import Request, Response
import time

@chute.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to all responses."""

    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    response.headers["X-Process-Time"] = str(process_time)
    return response
```

### 2. Custom Dependencies

```python
from fastapi import Depends, HTTPException

async def verify_api_key(api_key: str = Header(None)) -> str:
    """Verify API key dependency."""

    if not api_key or api_key != "your-secret-key":
        raise HTTPException(status_code=401, detail="Invalid API key")

    return api_key

@chute.cord(public_api_path="/secure", method="POST")
async def secure_endpoint(
    self,
    input_data: Dict,
    api_key: str = Depends(verify_api_key)
) -> Dict:
    """Secure endpoint requiring API key."""

    return {"message": "Access granted", "data": input_data}
```

### 3. WebSocket Support

```python
from fastapi import WebSocket

@chute.websocket("/ws")
async def websocket_endpoint(self, websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""

    await websocket.accept()

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()

            # Process message
            response = await self.process_message(data)

            # Send response
            await websocket.send_text(response)

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
```

## Next Steps

- **Production Deployment**: Scale and monitor custom chutes
- **Advanced Patterns**: Implement microservices architectures
- **Integration**: Connect with external APIs and databases
- **Optimization**: Profile and optimize performance

For more advanced topics, see:

- [Error Handling Guide](error-handling)
- [Best Practices](best-practices)
- [Performance Optimization](performance-optimization)
