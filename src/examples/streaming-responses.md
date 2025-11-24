# Streaming Responses

This example shows how to create **streaming API endpoints** that send results in real-time as they become available. Perfect for long-running AI tasks where you want to show progress.

## Real-time Text Streaming {#real-time-text-streaming}

Real-time text streaming allows you to process and return results as they become available, providing immediate feedback to users instead of waiting for all processing to complete. This is especially valuable for:

- **Long-running AI operations** - Show progress during model inference
- **Interactive applications** - Provide immediate feedback as users type
- **Large text processing** - Stream results chunk by chunk
- **Multi-step workflows** - Display each processing step as it completes

## What We'll Build

A text processing service that streams results as they're computed:

- 🌊 **Streaming responses** with real-time updates
- 📊 **Progress tracking** for long operations
- 🔄 **Multiple processing steps** shown incrementally
- 📝 **Chunked text processing** for large inputs

## Complete Example

### `streaming_processor.py`

````python
import asyncio
import time
import json
from typing import AsyncGenerator

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from pydantic import BaseModel, Field
from fastapi import HTTPException

from chutes.chute import Chute, NodeSelector
from chutes.image import Image

# === INPUT SCHEMAS ===

class StreamingTextInput(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000)
    include_sentiment: bool = Field(True, description="Include sentiment analysis")
    include_summary: bool = Field(True, description="Include text summarization")
    include_entities: bool = Field(False, description="Include named entity recognition")
    chunk_size: int = Field(200, ge=50, le=500, description="Text chunk size for processing")

# === CUSTOM IMAGE ===

image = (
    Image(username="myuser", name="streaming-processor", tag="1.0")
    .from_base("nvidia/cuda:12.2-runtime-ubuntu22.04")
    .with_python("3.11")
    .run_command("pip install torch==2.1.0 transformers==4.30.0 accelerate==0.20.0 spacy>=3.6.0")
    .run_command("python -m spacy download en_core_web_sm")
    .with_env("TRANSFORMERS_CACHE", "/app/models")
)

# === CHUTE DEFINITION ===

chute = Chute(
    username="myuser",
    name="streaming-processor",
    image=image,
    tagline="Real-time streaming text processing",
    readme="""
# Streaming Text Processor

Process text with real-time streaming results.

## Usage

```bash
curl -X POST https://myuser-streaming-processor.chutes.ai/process-stream \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Your long text here..."}' \\
  --no-buffer
```

Each response line contains JSON with the current processing step.
""",
node_selector=NodeSelector(
gpu_count=1,
min_vram_gb_per_gpu=12
)
)

# === MODEL LOADING ===

@chute.on_startup()
async def load_models(self):
"""Load all models needed for processing."""
print("Loading models for streaming processing...")

    # Sentiment analysis model
    sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

    # Summarization pipeline
    self.summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=0 if torch.cuda.is_available() else -1
    )

    # Named entity recognition (spaCy)
    import spacy
    self.nlp = spacy.load("en_core_web_sm")

    # Move sentiment model to GPU
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.sentiment_model.to(self.device)

    print(f"All models loaded on device: {self.device}")

# === STREAMING ENDPOINTS ===

@chute.cord(
public_api_path="/process-stream",
method="POST",
input_schema=StreamingTextInput,
stream=True,
output_content_type="application/json"
)
async def process_text_stream(self, data: StreamingTextInput) -> AsyncGenerator[dict, None]:
"""Process text with streaming results."""

    start_time = time.time()

    # Initial status
    yield {
        "status": "started",
        "message": "Beginning text processing...",
        "timestamp": time.time(),
        "text_length": len(data.text)
    }

    # Step 1: Text chunking
    yield {"status": "chunking", "message": "Splitting text into chunks..."}

    chunks = []
    text = data.text
    for i in range(0, len(text), data.chunk_size):
        chunk = text[i:i + data.chunk_size]
        chunks.append(chunk)

    yield {
        "status": "chunked",
        "message": f"Split into {len(chunks)} chunks",
        "chunks": len(chunks)
    }

    # Step 2: Sentiment Analysis (if requested)
    if data.include_sentiment:
        yield {"status": "sentiment_processing", "message": "Analyzing sentiment..."}

        try:
            sentiments = []
            for i, chunk in enumerate(chunks):
                # Process chunk
                inputs = self.sentiment_tokenizer(
                    chunk, return_tensors="pt", truncation=True, max_length=512
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.sentiment_model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

                # Get sentiment
                labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
                predicted_class = predictions.argmax().item()
                confidence = predictions[0][predicted_class].item()

                chunk_sentiment = {
                    "chunk": i + 1,
                    "sentiment": labels[predicted_class],
                    "confidence": confidence
                }
                sentiments.append(chunk_sentiment)

                # Stream progress
                yield {
                    "status": "sentiment_progress",
                    "progress": (i + 1) / len(chunks),
                    "chunk_result": chunk_sentiment
                }

                # Small delay to show streaming effect
                await asyncio.sleep(0.1)

            # Overall sentiment
            positive_chunks = sum(1 for s in sentiments if s["sentiment"] == "POSITIVE")
            negative_chunks = sum(1 for s in sentiments if s["sentiment"] == "NEGATIVE")

            if positive_chunks > negative_chunks:
                overall_sentiment = "POSITIVE"
            elif negative_chunks > positive_chunks:
                overall_sentiment = "NEGATIVE"
            else:
                overall_sentiment = "NEUTRAL"

            yield {
                "status": "sentiment_complete",
                "overall_sentiment": overall_sentiment,
                "chunk_sentiments": sentiments,
                "positive_chunks": positive_chunks,
                "negative_chunks": negative_chunks
            }

        except Exception as e:
            yield {"status": "sentiment_error", "error": str(e)}

    # Step 3: Summarization (if requested)
    if data.include_summary and len(data.text) > 100:
        yield {"status": "summarization_processing", "message": "Generating summary..."}

        try:
            # Summarize the full text
            summary_result = self.summarizer(
                data.text,
                max_length=130,
                min_length=30,
                do_sample=False
            )

            summary = summary_result[0]['summary_text']

            yield {
                "status": "summarization_complete",
                "summary": summary,
                "compression_ratio": len(summary) / len(data.text)
            }

        except Exception as e:
            yield {"status": "summarization_error", "error": str(e)}

    # Step 4: Named Entity Recognition (if requested)
    if data.include_entities:
        yield {"status": "entities_processing", "message": "Extracting entities..."}

        try:
            doc = self.nlp(data.text)
            entities = []

            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "description": spacy.explain(ent.label_),
                    "start": ent.start_char,
                    "end": ent.end_char
                })

            # Group by entity type
            entity_types = {}
            for ent in entities:
                label = ent["label"]
                if label not in entity_types:
                    entity_types[label] = []
                entity_types[label].append(ent)

            yield {
                "status": "entities_complete",
                "entities": entities,
                "entity_types": entity_types,
                "total_entities": len(entities)
            }

        except Exception as e:
            yield {"status": "entities_error", "error": str(e)}

    # Final status
    total_time = time.time() - start_time
    yield {
        "status": "completed",
        "message": "All processing complete!",
        "total_processing_time": total_time,
        "timestamp": time.time()
    }

@chute.cord(
public_api_path="/generate-stream",
method="POST",
stream=True,
output_content_type="text/plain"
)
async def generate_text_stream(self, prompt: str) -> AsyncGenerator[str, None]:
"""Simple text generation with streaming (simulated)."""

    # Simulate text generation word by word
    words = [
        "Artificial", "intelligence", "is", "revolutionizing", "how", "we",
        "process", "and", "understand", "text", "data.", "With", "advanced",
        "models", "like", "transformers,", "we", "can", "perform", "complex",
        "natural", "language", "tasks", "with", "unprecedented", "accuracy."
    ]

    yield f"Prompt: {prompt}\n\nGenerated text: "

    for word in words:
        yield word + " "
        await asyncio.sleep(0.2)  # Simulate processing time

    yield "\n\n[Generation complete]"

# === REGULAR (NON-STREAMING) ENDPOINT FOR COMPARISON ===

@chute.cord(
public_api_path="/process-batch",
method="POST",
input_schema=StreamingTextInput,
output_content_type="application/json"
)
async def process_text_batch(self, data: StreamingTextInput) -> dict:
"""Non-streaming version that returns all results at once."""

    start_time = time.time()
    results = {}

    # Sentiment analysis
    if data.include_sentiment:
        inputs = self.sentiment_tokenizer(
            data.text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        predicted_class = predictions.argmax().item()
        confidence = predictions[0][predicted_class].item()

        results["sentiment"] = {
            "label": labels[predicted_class],
            "confidence": confidence
        }

    # Summarization
    if data.include_summary and len(data.text) > 100:
        summary_result = self.summarizer(data.text, max_length=130, min_length=30)
        results["summary"] = summary_result[0]['summary_text']

    # Entities
    if data.include_entities:
        doc = self.nlp(data.text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        results["entities"] = entities

    results["processing_time"] = time.time() - start_time
    return results

````

## Testing the Streaming API

### Using curl

```bash
# Test streaming processing
curl -X POST https://myuser-streaming-processor.chutes.ai/process-stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I love using this amazing new technology! It has completely transformed how I work. The artificial intelligence capabilities are impressive and the user interface is intuitive. However, there are still some areas that could be improved.",
    "include_sentiment": true,
    "include_summary": true,
    "include_entities": true,
    "chunk_size": 100
  }' \
  --no-buffer
```

### Using Python

```python
import asyncio
import aiohttp
import json

async def stream_text_processing():
    """Test the streaming text processing endpoint."""

    payload = {
        "text": """
        Artificial intelligence is rapidly transforming industries across the globe.
        Companies like Google, Microsoft, and OpenAI are leading the charge with
        innovative models and applications. The technology is being used in healthcare,
        finance, education, and many other sectors. While the potential is enormous,
        there are also important ethical considerations that need to be addressed.
        """,
        "include_sentiment": True,
        "include_summary": True,
        "include_entities": True,
        "chunk_size": 150
    }

    async with aiohttp.ClientSession() as session:
        url = "https://myuser-streaming-processor.chutes.ai/process-stream"

        async with session.post(url, json=payload) as response:
            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode())
                        print(f"[{data['status']}] {data.get('message', '')}")

                        # Handle specific result types
                        if data['status'] == 'sentiment_complete':
                            print(f"Overall sentiment: {data['overall_sentiment']}")
                        elif data['status'] == 'summarization_complete':
                            print(f"Summary: {data['summary']}")
                        elif data['status'] == 'entities_complete':
                            print(f"Found {data['total_entities']} entities")

                    except json.JSONDecodeError:
                        continue

# Run the test
asyncio.run(stream_text_processing())
```

### Using JavaScript/Browser

```javascript
async function streamTextProcessing() {
	const response = await fetch('/process-stream', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({
			text: 'Your text here...',
			include_sentiment: true,
			include_summary: true,
			chunk_size: 200
		})
	});

	const reader = response.body.getReader();
	const decoder = new TextDecoder();

	while (true) {
		const { done, value } = await reader.read();
		if (done) break;

		const lines = decoder.decode(value).split('\n');
		for (const line of lines) {
			if (line.trim()) {
				try {
					const data = JSON.parse(line);
					console.log(`[${data.status}]`, data.message || '');

					// Update UI based on status
					updateProgressUI(data);
				} catch (e) {
					// Skip invalid JSON
				}
			}
		}
	}
}

function updateProgressUI(data) {
	const statusDiv = document.getElementById('status');
	const resultsDiv = document.getElementById('results');

	statusDiv.textContent = data.message || data.status;

	if (data.status === 'sentiment_complete') {
		resultsDiv.innerHTML += `<p>Sentiment: ${data.overall_sentiment}</p>`;
	} else if (data.status === 'summarization_complete') {
		resultsDiv.innerHTML += `<p>Summary: ${data.summary}</p>`;
	}
}
```

## Key Streaming Concepts

### 1. **AsyncGenerator Pattern**

```python
async def my_stream() -> AsyncGenerator[dict, None]:
    for i in range(10):
        yield {"step": i, "data": f"Processing item {i}"}
        await asyncio.sleep(0.1)  # Simulate work
```

### 2. **Progress Tracking**

```python
total_items = len(items)
for i, item in enumerate(items):
    # Process item
    result = await process_item(item)

    # Yield progress
    yield {
        "status": "processing",
        "progress": (i + 1) / total_items,
        "current_item": i + 1,
        "total_items": total_items,
        "result": result
    }
```

### 3. **Error Handling in Streams**

```python
try:
    result = await risky_operation()
    yield {"status": "success", "result": result}
except Exception as e:
    yield {"status": "error", "error": str(e)}
    # Continue with other operations if possible
```

### 4. **Multiple Content Types**

```python
# JSON streaming
@chute.cord(stream=True, output_content_type="application/json")
async def json_stream(self):
    yield {"message": "JSON data"}

# Plain text streaming
@chute.cord(stream=True, output_content_type="text/plain")
async def text_stream(self):
    yield "Plain text data\n"
```

## Performance Considerations

### 1. **Chunk Size Optimization**

```python
# Too small: many HTTP chunks, overhead
chunk_size = 10

# Too large: delayed responses, memory usage
chunk_size = 10000

# Just right: balance responsiveness and efficiency
chunk_size = 200
```

### 2. **Async Processing**

```python
# Good: Non-blocking delays
await asyncio.sleep(0.1)

# Bad: Blocking operations (use sparingly)
time.sleep(0.1)
```

### 3. **Memory Management**

```python
# Process in chunks to avoid memory issues
async def process_large_text(text: str):
    chunk_size = 1000
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        result = await process_chunk(chunk)
        yield {"chunk": i // chunk_size, "result": result}
        # Chunk is automatically garbage collected
```

## Use Cases for Streaming

### 1. **Long-Running AI Tasks**

- Model training progress
- Large text processing
- Image/video generation

### 2. **Real-Time Analysis**

- Live sentiment monitoring
- Stream processing
- Progressive enhancement

### 3. **User Experience**

- Show progress to users
- Provide intermediate results
- Reduce perceived latency

## Next Steps

- **[Batch Processing](/docs/examples/batch-processing)** - Handle multiple inputs efficiently
- **[Multi-Model Analysis](/docs/examples/multi-model-analysis)** - Combine different AI models
- **[Custom Images Guide](/docs/guides/custom-images)** - Advanced Docker setups
- **[Performance Optimization](/docs/guides/performance)** - Speed up your chutes
