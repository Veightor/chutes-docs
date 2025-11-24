# Multi-Model Analysis with Chutes

This guide demonstrates how to build sophisticated analysis systems that combine multiple AI models to provide comprehensive insights from text, images, audio, and other data types.

## Overview

Multi-model analysis enables:

- **Comprehensive Understanding**: Combine different AI models for deeper insights
- **Cross-Modal Analysis**: Analyze relationships between text, images, and audio
- **Ensemble Predictions**: Improve accuracy by combining multiple model outputs
- **Specialized Processing**: Use domain-specific models for different aspects
- **Robust Error Handling**: Graceful degradation when individual models fail

## Architecture Patterns

### Sequential Processing Pipeline

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import asyncio
from dataclasses import dataclass
import logging

@dataclass
class ModelResult:
    model_name: str
    result: Dict[str, Any]
    confidence: float
    processing_time_ms: float
    status: str = "success"
    error: Optional[str] = None

class MultiModelRequest(BaseModel):
    text: Optional[str] = None
    image_base64: Optional[str] = None
    audio_base64: Optional[str] = None
    analysis_types: List[str] = Field(default=["sentiment", "entities", "classification"])
    combine_results: bool = True
    confidence_threshold: float = 0.5

class MultiModelResponse(BaseModel):
    individual_results: List[ModelResult]
    combined_analysis: Optional[Dict[str, Any]] = None
    overall_confidence: float
    total_processing_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MultiModelAnalyzer:
    def __init__(self):
        self.models = {}
        self.logger = logging.getLogger(__name__)

        # Initialize individual model services
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all AI model services"""
        # Text analysis models
        self.models["sentiment"] = SentimentAnalyzer()
        self.models["entities"] = EntityExtractor()
        self.models["classification"] = TextClassifier()
        self.models["summarization"] = TextSummarizer()

        # Image analysis models
        self.models["image_classification"] = ImageClassifier()
        self.models["object_detection"] = ObjectDetector()
        self.models["ocr"] = OpticalCharacterRecognition()

        # Audio analysis models
        self.models["speech_recognition"] = SpeechRecognizer()
        self.models["audio_classification"] = AudioClassifier()

        # Cross-modal models
        self.models["image_captioning"] = ImageCaptioner()
        self.models["visual_qa"] = VisualQuestionAnswering()

    async def analyze(self, request: MultiModelRequest) -> MultiModelResponse:
        """Perform multi-model analysis"""
        start_time = time.time()
        results = []

        # Determine which models to run based on available inputs
        models_to_run = self._select_models(request)

        # Run models in parallel where possible
        tasks = []
        for model_name in models_to_run:
            task = self._run_model_safe(model_name, request)
            tasks.append(task)

        # Execute all tasks
        model_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for model_name, result in zip(models_to_run, model_results):
            if isinstance(result, Exception):
                results.append(ModelResult(
                    model_name=model_name,
                    result={},
                    confidence=0.0,
                    processing_time_ms=0.0,
                    status="error",
                    error=str(result)
                ))
            else:
                results.append(result)

        # Combine results if requested
        combined_analysis = None
        if request.combine_results:
            combined_analysis = self._combine_results(results, request)

        # Calculate overall metrics
        successful_results = [r for r in results if r.status == "success"]
        overall_confidence = (
            sum(r.confidence for r in successful_results) / len(successful_results)
            if successful_results else 0.0
        )

        total_time = (time.time() - start_time) * 1000

        return MultiModelResponse(
            individual_results=results,
            combined_analysis=combined_analysis,
            overall_confidence=overall_confidence,
            total_processing_time_ms=total_time,
            metadata={
                "models_run": len(models_to_run),
                "successful_models": len(successful_results),
                "failed_models": len(results) - len(successful_results)
            }
        )

    def _select_models(self, request: MultiModelRequest) -> List[str]:
        """Select which models to run based on available inputs and analysis types"""
        models_to_run = []

        # Text-based models
        if request.text:
            if "sentiment" in request.analysis_types:
                models_to_run.append("sentiment")
            if "entities" in request.analysis_types:
                models_to_run.append("entities")
            if "classification" in request.analysis_types:
                models_to_run.append("classification")
            if "summarization" in request.analysis_types:
                models_to_run.append("summarization")

        # Image-based models
        if request.image_base64:
            if "image_classification" in request.analysis_types:
                models_to_run.append("image_classification")
            if "object_detection" in request.analysis_types:
                models_to_run.append("object_detection")
            if "ocr" in request.analysis_types:
                models_to_run.append("ocr")
            if "image_captioning" in request.analysis_types:
                models_to_run.append("image_captioning")

        # Audio-based models
        if request.audio_base64:
            if "speech_recognition" in request.analysis_types:
                models_to_run.append("speech_recognition")
            if "audio_classification" in request.analysis_types:
                models_to_run.append("audio_classification")

        # Cross-modal models
        if request.text and request.image_base64:
            if "visual_qa" in request.analysis_types:
                models_to_run.append("visual_qa")

        return models_to_run

    async def _run_model_safe(self, model_name: str, request: MultiModelRequest) -> ModelResult:
        """Safely run a model with error handling"""
        start_time = time.time()

        try:
            model = self.models[model_name]
            result = await self._execute_model(model, model_name, request)

            processing_time = (time.time() - start_time) * 1000

            return ModelResult(
                model_name=model_name,
                result=result["output"],
                confidence=result.get("confidence", 0.5),
                processing_time_ms=processing_time
            )

        except Exception as e:
            self.logger.error(f"Model {model_name} failed: {e}")
            processing_time = (time.time() - start_time) * 1000

            return ModelResult(
                model_name=model_name,
                result={},
                confidence=0.0,
                processing_time_ms=processing_time,
                status="error",
                error=str(e)
            )

    async def _execute_model(self, model, model_name: str, request: MultiModelRequest) -> Dict[str, Any]:
        """Execute a specific model based on its type"""
        if model_name in ["sentiment", "entities", "classification", "summarization"]:
            return await model.analyze(request.text)

        elif model_name in ["image_classification", "object_detection", "ocr"]:
            return await model.analyze(request.image_base64)

        elif model_name == "image_captioning":
            return await model.generate_caption(request.image_base64)

        elif model_name in ["speech_recognition", "audio_classification"]:
            return await model.analyze(request.audio_base64)

        elif model_name == "visual_qa":
            return await model.answer(request.text, request.image_base64)

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _combine_results(self, results: List[ModelResult], request: MultiModelRequest) -> Dict[str, Any]:
        """Combine results from multiple models intelligently"""
        combined = {
            "summary": {},
            "confidence_scores": {},
            "cross_modal_insights": {},
            "consensus": {}
        }

        # Extract successful results
        successful_results = [r for r in results if r.status == "success"]

        # Sentiment consensus
        sentiment_results = [r for r in successful_results if r.model_name == "sentiment"]
        if sentiment_results:
            combined["summary"]["sentiment"] = sentiment_results[0].result
            combined["confidence_scores"]["sentiment"] = sentiment_results[0].confidence

        # Entity consolidation
        entity_results = [r for r in successful_results if r.model_name == "entities"]
        if entity_results:
            entities = entity_results[0].result.get("entities", [])
            # Group entities by type
            entity_groups = {}
            for entity in entities:
                entity_type = entity.get("label", "UNKNOWN")
                if entity_type not in entity_groups:
                    entity_groups[entity_type] = []
                entity_groups[entity_type].append(entity["text"])

            combined["summary"]["entities"] = entity_groups
            combined["confidence_scores"]["entities"] = entity_results[0].confidence

        # Cross-modal insights
        if request.text and request.image_base64:
            text_sentiment = next((r.result for r in successful_results if r.model_name == "sentiment"), None)
            image_caption = next((r.result for r in successful_results if r.model_name == "image_captioning"), None)

            if text_sentiment and image_caption:
                combined["cross_modal_insights"]["text_image_alignment"] = self._analyze_text_image_alignment(
                    text_sentiment, image_caption
                )

        # Generate overall consensus
        combined["consensus"] = self._generate_consensus(successful_results)

        return combined

    def _analyze_text_image_alignment(self, text_sentiment: Dict, image_caption: Dict) -> Dict[str, Any]:
        """Analyze alignment between text sentiment and image content"""
        # Simple alignment analysis
        text_polarity = text_sentiment.get("label", "neutral")
        caption_text = image_caption.get("caption", "")

        # Basic keyword matching for alignment
        positive_keywords = ["happy", "smile", "bright", "beautiful", "joy"]
        negative_keywords = ["sad", "dark", "angry", "broken", "disappointed"]

        caption_lower = caption_text.lower()
        positive_matches = sum(1 for word in positive_keywords if word in caption_lower)
        negative_matches = sum(1 for word in negative_keywords if word in caption_lower)

        if positive_matches > negative_matches:
            image_sentiment = "positive"
        elif negative_matches > positive_matches:
            image_sentiment = "negative"
        else:
            image_sentiment = "neutral"

        alignment_score = 1.0 if text_polarity == image_sentiment else 0.5

        return {
            "text_sentiment": text_polarity,
            "inferred_image_sentiment": image_sentiment,
            "alignment_score": alignment_score,
            "caption": caption_text
        }

    def _generate_consensus(self, results: List[ModelResult]) -> Dict[str, Any]:
        """Generate consensus view across all successful models"""
        consensus = {
            "primary_insights": [],
            "confidence_level": "low",
            "recommendation": "further_analysis_needed"
        }

        # Aggregate confidence scores
        avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0.0

        if avg_confidence > 0.8:
            consensus["confidence_level"] = "high"
            consensus["recommendation"] = "results_reliable"
        elif avg_confidence > 0.6:
            consensus["confidence_level"] = "medium"
            consensus["recommendation"] = "results_moderately_reliable"

        # Extract key insights
        for result in results:
            if result.confidence > 0.7:
                if result.model_name == "sentiment":
                    consensus["primary_insights"].append(
                        f"Text sentiment: {result.result.get('label', 'unknown')}"
                    )
                elif result.model_name == "classification":
                    consensus["primary_insights"].append(
                        f"Content category: {result.result.get('predicted_class', 'unknown')}"
                    )
                elif result.model_name == "object_detection":
                    objects = result.result.get("objects", [])
                    if objects:
                        consensus["primary_insights"].append(
                            f"Key objects detected: {', '.join([obj['class'] for obj in objects[:3]])}"
                        )

        return consensus

# Model implementations (simplified interfaces)
class SentimentAnalyzer:
    async def analyze(self, text: str) -> Dict[str, Any]:
        # Implementation would use actual sentiment model
        return {
            "output": {"label": "positive", "score": 0.85},
            "confidence": 0.85
        }

class EntityExtractor:
    async def analyze(self, text: str) -> Dict[str, Any]:
        # Implementation would use actual NER model
        return {
            "output": {
                "entities": [
                    {"text": "Apple", "label": "ORG", "start": 0, "end": 5}
                ]
            },
            "confidence": 0.9
        }

class ImageClassifier:
    async def analyze(self, image_base64: str) -> Dict[str, Any]:
        # Implementation would use actual image classification model
        return {
            "output": {"class": "cat", "score": 0.92},
            "confidence": 0.92
        }

class ImageCaptioner:
    async def generate_caption(self, image_base64: str) -> Dict[str, Any]:
        # Implementation would use actual image captioning model
        return {
            "output": {"caption": "A cat sitting on a windowsill"},
            "confidence": 0.88
        }

# Global analyzer instance
multi_analyzer = None

def initialize_analyzer():
    """Initialize the multi-model analyzer"""
    global multi_analyzer
    multi_analyzer = MultiModelAnalyzer()
    return {"status": "initialized", "models_available": len(multi_analyzer.models)}

async def analyze_multi_modal(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Main multi-model analysis endpoint"""
    request = MultiModelRequest(**inputs)
    result = await multi_analyzer.analyze(request)
    return result.dict()
```

## Production Deployment

### Scalable Multi-Model Service

```python
from chutes.image import Image
from chutes.chute import Chute, NodeSelector

# Comprehensive multi-model image
multi_model_image = (
    Image(
        username="myuser",
        name="multi-model-analysis",
        tag="1.0.0",
        base_image="nvidia/cuda:12.1-devel-ubuntu22.04",
        python_version="3.11"
    )
    .run_command("pip install torch==2.1.0+cu121 transformers==4.35.0 sentence-transformers==2.2.2 opencv-python==4.8.0.76 pillow==10.0.1 ultralytics==8.0.206 librosa==0.10.1 soundfile==0.12.1 whisper==1.1.10 pytesseract==0.3.10 easyocr==1.7.0 numpy==1.24.3 scipy==1.11.4 scikit-learn==1.3.0 redis==5.0.0 --extra-index-url https://download.pytorch.org/whl/cu121")
    .run_command("apt-get update && apt-get install -y tesseract-ocr")
    .add("./models", "/app/models")
    .add("./multi_model", "/app/multi_model")
)

# Deploy multi-model service
multi_model_chute = Chute(
    username="myuser",
    name="multi-model-analysis",
    image=multi_model_image,
    entry_file="multi_model_analyzer.py",
    entry_point="analyze_multi_modal",
    node_selector=NodeSelector(
        gpu_count=2,
        min_vram_gb_per_gpu=16),
    timeout_seconds=600,
    concurrency=5
)

result = multi_model_chute.deploy()
print(f"Multi-model service deployed: {result}")
```

## Advanced Use Cases

### Document Intelligence

```python
class DocumentIntelligenceAnalyzer(MultiModelAnalyzer):
    """Specialized analyzer for document processing"""

    async def analyze_document(self, document_image: str, document_text: str = None) -> Dict[str, Any]:
        """Comprehensive document analysis"""

        # Extract text using OCR if not provided
        if not document_text:
            ocr_result = await self.models["ocr"].analyze(document_image)
            document_text = ocr_result["output"]["text"]

        # Parallel analysis
        tasks = [
            self.models["entities"].analyze(document_text),           # Named entities
            self.models["classification"].analyze(document_text),     # Document type
            self.models["sentiment"].analyze(document_text),          # Sentiment/tone
            self.models["object_detection"].analyze(document_image),  # Layout analysis
            self._extract_document_structure(document_image),         # Structure analysis
            self._detect_signatures_stamps(document_image)           # Signature detection
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine into document intelligence report
        intelligence_report = {
            "document_type": results[1].get("predicted_class") if len(results) > 1 else "unknown",
            "extracted_entities": results[0].get("entities", []) if len(results) > 0 else [],
            "document_sentiment": results[2].get("label") if len(results) > 2 else "neutral",
            "layout_elements": results[3].get("objects", []) if len(results) > 3 else [],
            "structure_analysis": results[4] if len(results) > 4 else {},
            "signature_analysis": results[5] if len(results) > 5 else {},
            "extracted_text": document_text,
            "confidence_score": self._calculate_document_confidence(results)
        }

        return intelligence_report

    async def _extract_document_structure(self, image_base64: str) -> Dict[str, Any]:
        """Analyze document structure and layout"""
        # Implementation would use layout analysis model
        return {
            "sections": ["header", "body", "footer"],
            "tables_detected": 2,
            "figures_detected": 1,
            "text_blocks": 5
        }

    async def _detect_signatures_stamps(self, image_base64: str) -> Dict[str, Any]:
        """Detect signatures and stamps in document"""
        # Implementation would use specialized signature detection
        return {
            "signatures_detected": 1,
            "stamps_detected": 0,
            "signature_locations": [{"x": 450, "y": 600, "width": 150, "height": 50}]
        }

async def analyze_document_intelligence(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Document intelligence analysis endpoint"""
    analyzer = DocumentIntelligenceAnalyzer()

    result = await analyzer.analyze_document(
        document_image=inputs["document_image_base64"],
        document_text=inputs.get("document_text")
    )

    return result
```

### Social Media Content Analysis

```python
class SocialMediaAnalyzer(MultiModelAnalyzer):
    """Specialized analyzer for social media content"""

    async def analyze_social_post(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive social media post analysis"""

        text = post_data.get("text", "")
        images = post_data.get("images", [])
        video = post_data.get("video")
        audio = post_data.get("audio")

        analysis_tasks = []

        # Text analysis
        if text:
            analysis_tasks.extend([
                ("sentiment", self.models["sentiment"].analyze(text)),
                ("entities", self.models["entities"].analyze(text)),
                ("classification", self.models["classification"].analyze(text)),
                ("toxicity", self._analyze_toxicity(text)),
                ("engagement_prediction", self._predict_engagement(text))
            ])

        # Image analysis
        for i, image in enumerate(images):
            analysis_tasks.extend([
                (f"image_{i}_classification", self.models["image_classification"].analyze(image)),
                (f"image_{i}_objects", self.models["object_detection"].analyze(image)),
                (f"image_{i}_caption", self.models["image_captioning"].generate_caption(image)),
                (f"image_{i}_faces", self._detect_faces(image))
            ])

        # Audio analysis (if present)
        if audio:
            analysis_tasks.extend([
                ("speech_to_text", self.models["speech_recognition"].analyze(audio)),
                ("audio_mood", self.models["audio_classification"].analyze(audio))
            ])

        # Execute all analyses
        task_names, tasks = zip(*analysis_tasks) if analysis_tasks else ([], [])
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Compile comprehensive report
        social_analysis = {
            "content_summary": self._generate_content_summary(text, images, audio),
            "engagement_factors": self._analyze_engagement_factors(results, task_names),
            "risk_assessment": self._assess_content_risks(results, task_names),
            "recommendations": self._generate_recommendations(results, task_names),
            "virality_score": self._calculate_virality_score(results, task_names),
            "target_audience": self._identify_target_audience(results, task_names)
        }

        return social_analysis

    async def _analyze_toxicity(self, text: str) -> Dict[str, Any]:
        """Analyze text for toxic content"""
        # Implementation would use toxicity detection model
        return {"toxicity_score": 0.1, "is_toxic": False}

    async def _predict_engagement(self, text: str) -> Dict[str, Any]:
        """Predict engagement potential of text"""
        # Implementation would use engagement prediction model
        return {"predicted_likes": 150, "predicted_shares": 25, "predicted_comments": 10}

    def _calculate_virality_score(self, results: List, task_names: List[str]) -> float:
        """Calculate potential virality score"""
        # Complex scoring algorithm based on multiple factors
        base_score = 0.5

        # Boost for positive sentiment
        sentiment_idx = next((i for i, name in enumerate(task_names) if name == "sentiment"), None)
        if sentiment_idx and not isinstance(results[sentiment_idx], Exception):
            sentiment = results[sentiment_idx].get("label", "neutral")
            if sentiment == "positive":
                base_score += 0.2

        # Boost for visual content
        image_count = sum(1 for name in task_names if "image_" in name and "_classification" in name)
        base_score += min(image_count * 0.1, 0.3)

        return min(base_score, 1.0)

async def analyze_social_media(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Social media analysis endpoint"""
    analyzer = SocialMediaAnalyzer()
    result = await analyzer.analyze_social_post(inputs["post_data"])
    return result
```

## Performance Optimization

### Caching and Load Balancing

```python
import redis
import pickle
import hashlib
from typing import Optional

class CachedMultiModelAnalyzer(MultiModelAnalyzer):
    """Multi-model analyzer with Redis caching"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        super().__init__()
        self.redis_client = redis.from_url(redis_url)
        self.cache_ttl = 3600  # 1 hour

    def _generate_cache_key(self, request: MultiModelRequest) -> str:
        """Generate cache key for request"""
        request_str = f"{request.text or ''}{request.image_base64 or ''}{request.audio_base64 or ''}"
        return f"multi_model:{hashlib.md5(request_str.encode()).hexdigest()}"

    async def analyze(self, request: MultiModelRequest) -> MultiModelResponse:
        """Analyze with caching"""
        cache_key = self._generate_cache_key(request)

        # Try to get from cache
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result

        # Perform analysis
        result = await super().analyze(request)

        # Cache result
        self._store_in_cache(cache_key, result)

        return result

    def _get_from_cache(self, key: str) -> Optional[MultiModelResponse]:
        """Get result from Redis cache"""
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return MultiModelResponse(**pickle.loads(cached_data))
        except Exception as e:
            self.logger.warning(f"Cache read error: {e}")
        return None

    def _store_in_cache(self, key: str, result: MultiModelResponse):
        """Store result in Redis cache"""
        try:
            serialized_data = pickle.dumps(result.dict())
            self.redis_client.setex(key, self.cache_ttl, serialized_data)
        except Exception as e:
            self.logger.warning(f"Cache write error: {e}")

# Model load balancing
class LoadBalancedMultiModelAnalyzer(CachedMultiModelAnalyzer):
    """Multi-model analyzer with load balancing across model instances"""

    def __init__(self, model_endpoints: Dict[str, List[str]], redis_url: str = "redis://localhost:6379"):
        super().__init__(redis_url)
        self.model_endpoints = model_endpoints
        self.current_endpoints = {model: 0 for model in model_endpoints}

    def _get_next_endpoint(self, model_name: str) -> str:
        """Get next endpoint using round-robin load balancing"""
        if model_name not in self.model_endpoints:
            raise ValueError(f"No endpoints configured for model: {model_name}")

        endpoints = self.model_endpoints[model_name]
        current_idx = self.current_endpoints[model_name]
        endpoint = endpoints[current_idx]

        # Update for next request
        self.current_endpoints[model_name] = (current_idx + 1) % len(endpoints)

        return endpoint

    async def _execute_model(self, model, model_name: str, request: MultiModelRequest) -> Dict[str, Any]:
        """Execute model with load balancing"""
        endpoint = self._get_next_endpoint(model_name)

        # Make HTTP request to model endpoint
        import httpx
        async with httpx.AsyncClient() as client:
            if model_name in ["sentiment", "entities", "classification"]:
                response = await client.post(f"{endpoint}/analyze", json={"text": request.text})
            elif model_name in ["image_classification", "object_detection"]:
                response = await client.post(f"{endpoint}/analyze", json={"image": request.image_base64})
            # Add more model types as needed

            response.raise_for_status()
            return response.json()
```

## Monitoring and Observability

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics
MODEL_REQUESTS = Counter('model_requests_total', 'Total model requests', ['model_name', 'status'])
MODEL_DURATION = Histogram('model_duration_seconds', 'Model execution time', ['model_name'])
ACTIVE_ANALYSES = Gauge('active_analyses', 'Number of active analyses')
CACHE_HITS = Counter('cache_hits_total', 'Cache hits', ['type'])

class MonitoredMultiModelAnalyzer(LoadBalancedMultiModelAnalyzer):
    """Multi-model analyzer with comprehensive monitoring"""

    async def analyze(self, request: MultiModelRequest) -> MultiModelResponse:
        """Analyze with monitoring"""
        ACTIVE_ANALYSES.inc()

        try:
            start_time = time.time()
            result = await super().analyze(request)

            # Record success metrics
            MODEL_REQUESTS.labels(model_name='multi_model', status='success').inc()
            MODEL_DURATION.labels(model_name='multi_model').observe(time.time() - start_time)

            return result

        except Exception as e:
            MODEL_REQUESTS.labels(model_name='multi_model', status='error').inc()
            raise
        finally:
            ACTIVE_ANALYSES.dec()

    async def _run_model_safe(self, model_name: str, request: MultiModelRequest) -> ModelResult:
        """Run model with individual monitoring"""
        MODEL_REQUESTS.labels(model_name=model_name, status='started').inc()

        with MODEL_DURATION.labels(model_name=model_name).time():
            result = await super()._run_model_safe(model_name, request)

        status = 'success' if result.status == 'success' else 'error'
        MODEL_REQUESTS.labels(model_name=model_name, status=status).inc()

        return result

# Start metrics server
start_http_server(8001)
```

## Usage Examples

### Comprehensive Content Analysis

```python
# Deploy the multi-model service
comprehensive_result = multi_model_chute.run({
    "text": "Just visited the most amazing restaurant! The food was incredible and the view was breathtaking. Highly recommend!",
    "image_base64": "...",  # Base64 encoded restaurant photo
    "analysis_types": [
        "sentiment", "entities", "classification",
        "image_classification", "object_detection", "image_captioning"
    ],
    "combine_results": True,
    "confidence_threshold": 0.6
})

print("Individual Results:")
for result in comprehensive_result["individual_results"]:
    print(f"- {result['model_name']}: {result['confidence']:.2f} confidence")

print("\nCombined Analysis:")
print(f"Overall sentiment: {comprehensive_result['combined_analysis']['summary']['sentiment']['label']}")
print(f"Entities found: {comprehensive_result['combined_analysis']['summary']['entities']}")
print(f"Cross-modal alignment: {comprehensive_result['combined_analysis']['cross_modal_insights']}")
```

## Next Steps

- **[Custom Training](custom-training)** - Train specialized models for your use case
- **[Performance Optimization](../guides/performance)** - Scale multi-model systems
- **[Production Deployment](../guides/production-deployment)** - Deploy at enterprise scale
- **[Model Orchestration](../guides/model-orchestration)** - Advanced model management

For enterprise multi-model architectures, see the [ML Platform Guide](../guides/ml-platform).
