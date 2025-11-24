# Complete Text Analysis Service

This guide demonstrates building a comprehensive text analysis service that combines multiple AI models for sentiment analysis, entity recognition, text classification, and content moderation.

## Overview

This complete example showcases:

- **Multi-model Architecture**: Combining different AI models in a single service
- **Sentiment Analysis**: Understanding emotional tone of text
- **Named Entity Recognition**: Extracting people, places, organizations
- **Text Classification**: Categorizing content by topic or intent
- **Content Moderation**: Detecting inappropriate or harmful content
- **Batch Processing**: Handling multiple texts efficiently
- **Error Handling**: Robust error management across models
- **Monitoring**: Built-in metrics and health checks
- **Caching**: Performance optimization for repeated queries

## Complete Implementation

### Input Schema Design

Define comprehensive input validation for text analysis:

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class AnalysisType(str, Enum):
    SENTIMENT = "sentiment"
    ENTITIES = "entities"
    CLASSIFICATION = "classification"
    MODERATION = "moderation"
    ALL = "all"

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    id: Optional[str] = Field(None, description="Optional identifier for tracking")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class InputArgs(BaseModel):
    texts: List[TextInput] = Field(..., min_items=1, max_items=100)
    analysis_types: List[AnalysisType] = Field(default=[AnalysisType.ALL])
    include_confidence: bool = Field(default=True)
    language: Optional[str] = Field(default="en", description="ISO language code")
```

### Custom Image with Multiple Models

Build a comprehensive image with all required AI models:

```python
from chutes.image import Image
from chutes.chute import Chute, NodeSelector

image = (
    Image(
        username="myuser",
        name="text-analysis-complete",
        tag="1.0.0",
        python_version="3.11"
    )
    .run_command("pip install transformers==4.35.0 torch==2.1.0 spacy==3.7.2 scikit-learn==1.3.0 numpy==1.24.3 pandas==2.0.3 redis==5.0.0 prometheus-client==0.18.0")
    .run_command("python -m spacy download en_core_web_sm")
    .run_command("python -m spacy download en_core_web_lg")
    .add("./models", "/app/models")
    .add("./config", "/app/config")
)
```

### Multi-Model Service Implementation

Create a comprehensive service that orchestrates multiple AI models:

```python
import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

import torch
import spacy
import redis
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline
)
from prometheus_client import Counter, Histogram, start_http_server
import numpy as np

# Metrics
REQUEST_COUNT = Counter('analysis_requests_total', 'Total analysis requests', ['type'])
REQUEST_DURATION = Histogram('analysis_duration_seconds', 'Request duration', ['type'])
ERROR_COUNT = Counter('analysis_errors_total', 'Total errors', ['type', 'error'])

@dataclass
class AnalysisResult:
    text_id: Optional[str]
    sentiment: Optional[Dict[str, Any]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    classification: Optional[Dict[str, Any]] = None
    moderation: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class TextAnalysisService:
    def __init__(self, cache_enabled: bool = True):
        self.logger = logging.getLogger(__name__)
        self.cache_enabled = cache_enabled

        # Initialize Redis cache
        if cache_enabled:
            try:
                self.cache = redis.Redis(host='localhost', port=6379, db=0)
                self.cache.ping()
                self.logger.info("Cache connection established")
            except Exception as e:
                self.logger.warning(f"Cache disabled: {e}")
                self.cache_enabled = False

        # Load models
        self._load_models()

        # Start metrics server
        start_http_server(8001)
        self.logger.info("Metrics server started on port 8001")

    def _load_models(self):
        """Load all AI models with proper error handling"""
        self.logger.info("Loading AI models...")

        try:
            # Sentiment Analysis Model
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
                "cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                "cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            self.logger.info("✓ Sentiment model loaded")

            # Text Classification Model
            self.classifier = pipeline(
                "text-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            self.logger.info("✓ Classification model loaded")

            # Content Moderation Model
            self.moderation_pipeline = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=0 if torch.cuda.is_available() else -1
            )
            self.logger.info("✓ Moderation model loaded")

            # Named Entity Recognition
            self.nlp = spacy.load("en_core_web_lg")
            self.logger.info("✓ NER model loaded")

        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise

    def _get_cache_key(self, text: str, analysis_type: str) -> str:
        """Generate cache key for text and analysis type"""
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"analysis:{analysis_type}:{text_hash}"

    def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Retrieve cached analysis result"""
        if not self.cache_enabled:
            return None

        try:
            cached = self.cache.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            self.logger.warning(f"Cache read error: {e}")

        return None

    def _cache_result(self, cache_key: str, result: Dict, ttl: int = 3600):
        """Cache analysis result with TTL"""
        if not self.cache_enabled:
            return

        try:
            self.cache.setex(
                cache_key,
                ttl,
                json.dumps(result, default=str)
            )
        except Exception as e:
            self.logger.warning(f"Cache write error: {e}")

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Perform sentiment analysis with caching"""
        cache_key = self._get_cache_key(text, "sentiment")
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached

        with REQUEST_DURATION.labels(type='sentiment').time():
            try:
                inputs = self.sentiment_tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )

                with torch.no_grad():
                    outputs = self.sentiment_model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

                labels = ['negative', 'neutral', 'positive']
                scores = predictions[0].tolist()

                result = {
                    'label': labels[np.argmax(scores)],
                    'confidence': float(max(scores)),
                    'scores': {label: float(score) for label, score in zip(labels, scores)}
                }

                self._cache_result(cache_key, result)
                REQUEST_COUNT.labels(type='sentiment').inc()
                return result

            except Exception as e:
                ERROR_COUNT.labels(type='sentiment', error=type(e).__name__).inc()
                raise Exception(f"Sentiment analysis failed: {e}")

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities with caching"""
        cache_key = self._get_cache_key(text, "entities")
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached

        with REQUEST_DURATION.labels(type='entities').time():
            try:
                doc = self.nlp(text)
                entities = []

                for ent in doc.ents:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'description': spacy.explain(ent.label_),
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': float(ent.kb_id_) if ent.kb_id_ else 0.9
                    })

                self._cache_result(cache_key, entities)
                REQUEST_COUNT.labels(type='entities').inc()
                return entities

            except Exception as e:
                ERROR_COUNT.labels(type='entities', error=type(e).__name__).inc()
                raise Exception(f"Entity extraction failed: {e}")

    async def classify_text(self, text: str, categories: List[str] = None) -> Dict[str, Any]:
        """Classify text into categories"""
        if categories is None:
            categories = [
                "technology", "business", "health", "sports",
                "entertainment", "politics", "science", "education"
            ]

        cache_key = self._get_cache_key(f"{text}:{','.join(categories)}", "classification")
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached

        with REQUEST_DURATION.labels(type='classification').time():
            try:
                # Use zero-shot classification
                candidate_labels = categories
                result = self.classifier(text, candidate_labels)

                classification_result = {
                    'predicted_category': result['labels'][0],
                    'confidence': float(result['scores'][0]),
                    'all_scores': {
                        label: float(score)
                        for label, score in zip(result['labels'], result['scores'])
                    }
                }

                self._cache_result(cache_key, classification_result)
                REQUEST_COUNT.labels(type='classification').inc()
                return classification_result

            except Exception as e:
                ERROR_COUNT.labels(type='classification', error=type(e).__name__).inc()
                raise Exception(f"Text classification failed: {e}")

    async def moderate_content(self, text: str) -> Dict[str, Any]:
        """Detect inappropriate content"""
        cache_key = self._get_cache_key(text, "moderation")
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached

        with REQUEST_DURATION.labels(type='moderation').time():
            try:
                result = self.moderation_pipeline(text)

                # Process toxicity detection result
                is_toxic = any(item['label'] == 'TOXIC' and item['score'] > 0.7 for item in result)
                max_toxicity_score = max((item['score'] for item in result if item['label'] == 'TOXIC'), default=0.0)

                moderation_result = {
                    'is_inappropriate': is_toxic,
                    'toxicity_score': float(max_toxicity_score),
                    'categories': result,
                    'action_required': is_toxic
                }

                self._cache_result(cache_key, moderation_result)
                REQUEST_COUNT.labels(type='moderation').inc()
                return moderation_result

            except Exception as e:
                ERROR_COUNT.labels(type='moderation', error=type(e).__name__).inc()
                raise Exception(f"Content moderation failed: {e}")

    async def analyze_single_text(
        self,
        text_input: TextInput,
        analysis_types: List[AnalysisType],
        include_confidence: bool = True
    ) -> AnalysisResult:
        """Analyze a single text with specified analysis types"""
        start_time = time.time()
        result = AnalysisResult(text_id=text_input.id, metadata=text_input.metadata)

        try:
            # Determine which analyses to run
            run_all = AnalysisType.ALL in analysis_types

            tasks = []

            if run_all or AnalysisType.SENTIMENT in analysis_types:
                tasks.append(("sentiment", self.analyze_sentiment(text_input.text)))

            if run_all or AnalysisType.ENTITIES in analysis_types:
                tasks.append(("entities", self.extract_entities(text_input.text)))

            if run_all or AnalysisType.CLASSIFICATION in analysis_types:
                tasks.append(("classification", self.classify_text(text_input.text)))

            if run_all or AnalysisType.MODERATION in analysis_types:
                tasks.append(("moderation", self.moderate_content(text_input.text)))

            # Run analyses concurrently
            if tasks:
                task_names, task_coroutines = zip(*tasks)
                results = await asyncio.gather(*task_coroutines, return_exceptions=True)

                for name, task_result in zip(task_names, results):
                    if isinstance(task_result, Exception):
                        self.logger.error(f"Analysis {name} failed: {task_result}")
                    else:
                        setattr(result, name, task_result)

            result.processing_time_ms = (time.time() - start_time) * 1000
            return result

        except Exception as e:
            self.logger.error(f"Text analysis failed: {e}")
            result.processing_time_ms = (time.time() - start_time) * 1000
            raise Exception(f"Analysis failed: {e}")

    async def analyze_batch(self, inputs: InputArgs) -> List[AnalysisResult]:
        """Analyze multiple texts concurrently"""
        self.logger.info(f"Processing batch of {len(inputs.texts)} texts")

        # Process texts concurrently with controlled concurrency
        semaphore = asyncio.Semaphore(10)  # Limit concurrent analyses

        async def analyze_with_semaphore(text_input):
            async with semaphore:
                return await self.analyze_single_text(
                    text_input,
                    inputs.analysis_types,
                    inputs.include_confidence
                )

        tasks = [analyze_with_semaphore(text_input) for text_input in inputs.texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = AnalysisResult(
                    text_id=inputs.texts[i].id,
                    metadata={"error": str(result)}
                )
                final_results.append(error_result)
            else:
                final_results.append(result)

        return final_results

# Global service instance
service = None

def get_service() -> TextAnalysisService:
    """Get or create the global service instance"""
    global service
    if service is None:
        service = TextAnalysisService()
    return service

async def run(inputs: InputArgs) -> List[Dict[str, Any]]:
    """Main entry point for the chute"""
    analysis_service = get_service()

    try:
        results = await analysis_service.analyze_batch(inputs)

        # Convert results to serializable format
        output = []
        for result in results:
            result_dict = {
                'text_id': result.text_id,
                'processing_time_ms': result.processing_time_ms,
                'metadata': result.metadata
            }

            if result.sentiment:
                result_dict['sentiment'] = result.sentiment
            if result.entities:
                result_dict['entities'] = result.entities
            if result.classification:
                result_dict['classification'] = result.classification
            if result.moderation:
                result_dict['moderation'] = result.moderation

            output.append(result_dict)

        return output

    except Exception as e:
        logging.error(f"Batch processing failed: {e}")
        raise Exception(f"Analysis service error: {e}")
```

### Creating the Complete Chute

Deploy the comprehensive text analysis service:

```python
from chutes.chute import Chute, NodeSelector

# Create the complete text analysis chute
chute = Chute(
    username="myuser",
    name="text-analysis-complete",
    image=image,
    entry_file="analysis_service.py",
    entry_point="run",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16),
    timeout_seconds=300,
    concurrency=5
)

# Deploy the service
print("Deploying comprehensive text analysis service...")
result = chute.deploy()
print(f"✅ Service deployed: {result}")
```

## Usage Examples

### Basic Text Analysis

```python
# Analyze a single text with all models
response = chute.run({
    "texts": [
        {
            "text": "I absolutely love this new AI technology! It's revolutionary and will change everything.",
            "id": "text_1"
        }
    ],
    "analysis_types": ["all"],
    "include_confidence": True
})

# Response includes all analysis types
result = response[0]
print(f"Sentiment: {result['sentiment']['label']} ({result['sentiment']['confidence']:.2f})")
print(f"Category: {result['classification']['predicted_category']}")
print(f"Entities: {[ent['text'] for ent in result['entities']]}")
print(f"Content Safe: {not result['moderation']['is_inappropriate']}")
```

### Batch Processing

```python
# Analyze multiple texts efficiently
texts = [
    {"text": "This product is amazing!", "id": "review_1"},
    {"text": "The service was terrible and slow.", "id": "review_2"},
    {"text": "Apple Inc. reported strong quarterly earnings.", "id": "news_1"},
    {"text": "The new iPhone features advanced AI capabilities.", "id": "tech_1"}
]

response = chute.run({
    "texts": texts,
    "analysis_types": ["sentiment", "entities", "classification"],
    "include_confidence": True
})

# Process results
for result in response:
    print(f"\nText ID: {result['text_id']}")
    print(f"Processing time: {result['processing_time_ms']:.2f}ms")
    if 'sentiment' in result:
        print(f"Sentiment: {result['sentiment']['label']}")
    if 'entities' in result:
        print(f"Entities: {[ent['text'] for ent in result['entities']]}")
```

### Selective Analysis

```python
# Run only specific analysis types
response = chute.run({
    "texts": [
        {"text": "Breaking: Tech giant announces major acquisition", "id": "headline_1"}
    ],
    "analysis_types": ["entities", "classification"],  # Only NER and classification
    "include_confidence": True
})
```

### Content Moderation Focus

```python
# Focus on content safety
user_comments = [
    {"text": "This is a great discussion!", "id": "comment_1"},
    {"text": "I disagree but respect your opinion.", "id": "comment_2"},
    {"text": "This platform needs better moderation.", "id": "comment_3"}
]

response = chute.run({
    "texts": user_comments,
    "analysis_types": ["moderation", "sentiment"],
    "include_confidence": True
})

# Filter inappropriate content
safe_comments = [
    result for result in response
    if not result['moderation']['is_inappropriate']
]
```

## Performance Optimization

### Caching Strategy

The service implements intelligent caching:

- **Redis-based caching** for repeated text analyses
- **1-hour TTL** for cached results
- **Cache keys** based on text content and analysis type
- **Graceful degradation** when cache is unavailable

### Concurrent Processing

- **Semaphore-controlled concurrency** (max 10 concurrent analyses)
- **Async/await patterns** for non-blocking operations
- **Batch processing** for multiple texts
- **Error isolation** prevents single failures from affecting the batch

### Resource Management

```python
# Optimized node selection for production
chute = Chute(
    username="myuser",
    name="text-analysis-production",
    image=image,
    entry_file="analysis_service.py",
    entry_point="run",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=24,  # Larger VRAM for complex models# More RAM for caching
        preferred_provider="runpod"  # Specify provider if needed
    ),
    timeout_seconds=600,         # Longer timeout for large batches
    concurrency=10               # Higher concurrency for production
)
```

## Monitoring and Observability

### Built-in Metrics

The service exposes Prometheus metrics on port 8001:

- `analysis_requests_total` - Total requests by analysis type
- `analysis_duration_seconds` - Request duration histograms
- `analysis_errors_total` - Error counts by type

### Health Checks

```python
# Health check endpoint
async def health_check():
    service = get_service()

    # Test all models with sample text
    test_text = "Hello world"

    try:
        await service.analyze_sentiment(test_text)
        await service.extract_entities(test_text)
        await service.classify_text(test_text)
        await service.moderate_content(test_text)

        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### Logging Configuration

```python
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/analysis.log')
    ]
)
```

## Error Handling and Recovery

### Graceful Degradation

```python
async def robust_analysis(text_input: TextInput) -> AnalysisResult:
    """Analysis with fallback strategies"""
    result = AnalysisResult(text_id=text_input.id)

    # Try sentiment analysis with fallback
    try:
        result.sentiment = await analyze_sentiment(text_input.text)
    except Exception as e:
        result.sentiment = {"error": "Sentiment analysis unavailable", "fallback": True}
        logger.warning(f"Sentiment analysis failed: {e}")

    # Continue with other analyses even if one fails
    try:
        result.entities = await extract_entities(text_input.text)
    except Exception as e:
        result.entities = []
        logger.warning(f"Entity extraction failed: {e}")

    return result
```

### Circuit Breaker Pattern

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"

            raise e
```

## Advanced Features

### Custom Model Integration

```python
# Add custom models to the service
class CustomTextAnalysisService(TextAnalysisService):
    def _load_models(self):
        super()._load_models()

        # Load custom domain-specific model
        self.custom_classifier = pipeline(
            "text-classification",
            model="/app/models/custom-domain-classifier",
            device=0 if torch.cuda.is_available() else -1
        )

    async def custom_classification(self, text: str) -> Dict[str, Any]:
        """Domain-specific classification"""
        result = self.custom_classifier(text)
        return {
            'custom_category': result[0]['label'],
            'confidence': result[0]['score']
        }
```

### Multi-language Support

```python
# Language detection and processing
from langdetect import detect

async def analyze_multilingual_text(self, text: str, language: str = None) -> Dict:
    """Analyze text with language-specific models"""

    # Auto-detect language if not provided
    if language is None:
        language = detect(text)

    # Load language-specific models
    if language == "es":
        nlp = spacy.load("es_core_news_sm")
    elif language == "fr":
        nlp = spacy.load("fr_core_news_sm")
    else:
        nlp = self.nlp  # Default English model

    # Process with appropriate model
    doc = nlp(text)
    return self._extract_entities_from_doc(doc)
```

## Deployment Best Practices

### Production Configuration

```python
# Production-ready deployment
production_chute = Chute(
    username="mycompany",
    name="text-analysis-prod",
    image=image,
    entry_file="analysis_service.py",
    entry_point="run",
    node_selector=NodeSelector(
        gpu_count=2,
        min_vram_gb_per_gpu=24preferred_provider="runpod",
        instance_type="RTX A6000"
    ),
    environment={
        "REDIS_URL": "redis://cache.example.com:6379",
        "LOG_LEVEL": "INFO",
        "CACHE_TTL": "3600",
        "MAX_BATCH_SIZE": "100"
    },
    timeout_seconds=900,
    concurrency=20,
    auto_scale=True,
    min_instances=2,
    max_instances=10
)
```

### Cost Optimization

```python
# Cost-optimized configuration for development
dev_chute = Chute(
    username="myuser",
    name="text-analysis-dev",
    image=image,
    entry_file="analysis_service.py",
    entry_point="run",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=8),
    timeout_seconds=300,
    concurrency=3,
    auto_scale=False
)
```

This comprehensive example demonstrates how to build a production-ready text analysis service that combines multiple AI models, implements proper error handling, includes monitoring and caching, and provides a robust API for various text analysis tasks.
