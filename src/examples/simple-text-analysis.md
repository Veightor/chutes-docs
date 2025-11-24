# Simple Text Analysis Chute

This example shows how to build a basic text analysis service using transformers and custom API endpoints. Perfect for getting started with custom Chutes.

## What We'll Build

A simple text sentiment analysis service that:

- 📊 **Analyzes sentiment** using a pre-trained model
- 🔍 **Validates input** with Pydantic schemas
- 🚀 **Provides REST API** for easy integration
- 📦 **Uses custom Docker image** with optimized dependencies

## Complete Example

### `sentiment_analyzer.py`

````python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydantic import BaseModel, Field
from fastapi import HTTPException

from chutes.chute import Chute, NodeSelector
from chutes.image import Image

# === INPUT/OUTPUT SCHEMAS ===

class TextInput(BaseModel):
    text: str = Field(..., min_length=5, max_length=1000, description="Text to analyze")

    class Config:
        schema_extra = {
            "example": {
                "text": "I love using this new AI service!"
            }
        }

class SentimentResult(BaseModel):
    text: str
    sentiment: str  # POSITIVE, NEGATIVE, NEUTRAL
    confidence: float
    processing_time: float

# === CUSTOM IMAGE ===

image = (
    Image(username="myuser", name="sentiment-analyzer", tag="1.0")
    .from_base("nvidia/cuda:12.2-runtime-ubuntu22.04")
    .with_python("3.11")
    .run_command("pip install torch==2.1.0 transformers==4.30.0 accelerate==0.20.0")
    .with_env("TRANSFORMERS_CACHE", "/app/models")
    .run_command("mkdir -p /app/models")
)

# === CHUTE DEFINITION ===

chute = Chute(
    username="myuser",
    name="sentiment-analyzer",
    image=image,
    tagline="Simple sentiment analysis with transformers",
    readme="""
# Sentiment Analyzer

A simple sentiment analysis service using DistilBERT.

## Usage

Send a POST request to `/analyze`:

```bash
curl -X POST https://myuser-sentiment-analyzer.chutes.ai/analyze \\
  -H "Content-Type: application/json" \\
  -d '{"text": "I love this product!"}'
```

## Response

```json
{
	"text": "I love this product!",
	"sentiment": "POSITIVE",
	"confidence": 0.99,
	"processing_time": 0.05
}
```

    """,
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=8
    )

)

# === MODEL LOADING ===

@chute.on_startup()
async def load_model(self):
"""Load the sentiment analysis model on startup."""
print("Loading sentiment analysis model...")

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    # Load tokenizer and model
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Move to GPU if available
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model.to(self.device)
    self.model.eval()  # Set to evaluation mode

    print(f"Model loaded on device: {self.device}")

# === API ENDPOINTS ===

@chute.cord(
public_api_path="/analyze",
method="POST",
input_schema=TextInput,
output_content_type="application/json"
)
async def analyze_sentiment(self, data: TextInput) -> SentimentResult:
"""Analyze the sentiment of the input text."""
import time

    start_time = time.time()

    try:
        # Tokenize input
        inputs = self.tokenizer(
            data.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Get results
        labels = ["NEGATIVE", "POSITIVE"]  # DistilBERT SST-2 labels
        predicted_class = predictions.argmax(dim=-1).item()
        confidence = predictions[0][predicted_class].item()

        processing_time = time.time() - start_time

        return SentimentResult(
            text=data.text,
            sentiment=labels[predicted_class],
            confidence=confidence,
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@chute.cord(
public_api_path="/health",
method="GET",
output_content_type="application/json"
)
async def health_check(self) -> dict:
"""Simple health check endpoint."""
return {
"status": "healthy",
"model_loaded": hasattr(self, 'model'),
"device": getattr(self, 'device', 'unknown')
}

# Test the chute locally (optional)

if **name** == "**main**":
import asyncio

    async def test_locally():
        # Simulate startup
        await load_model(chute)

        # Test analysis
        test_input = TextInput(text="I love this new AI service!")
        result = await analyze_sentiment(chute, test_input)
        print(f"Result: {result}")

    asyncio.run(test_locally())

````

## Step-by-Step Breakdown

### 1. Define Input/Output Schemas

```python
class TextInput(BaseModel):
    text: str = Field(..., min_length=5, max_length=1000)
```

- **Validation**: Ensures text is between 5-1000 characters
- **Documentation**: Provides clear API documentation
- **Type Safety**: Automatic parsing and validation

### 2. Build Custom Image

```python
image = (
    Image(username="myuser", name="sentiment-analyzer", tag="1.0")
    .from_base("nvidia/cuda:12.2-runtime-ubuntu22.04")
    .with_python("3.11")
    .run_command("pip install torch==2.1.0 transformers==4.30.0")
)
```

- **Base Image**: CUDA-enabled Ubuntu for GPU support
- **Dependencies**: Only what we need for sentiment analysis
- **Optimization**: Runtime image (smaller than devel)

### 3. Model Loading

```python
@chute.on_startup()
async def load_model(self):
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
    self.model.to(self.device)
```

- **Startup Hook**: Load model once when chute starts
- **GPU Support**: Automatically use GPU if available
- **State Management**: Store in chute instance

### 4. API Endpoint

```python
@chute.cord(public_api_path="/analyze", input_schema=TextInput)
async def analyze_sentiment(self, data: TextInput) -> SentimentResult:
    # Process the input
    return SentimentResult(...)
```

- **Path Mapping**: Creates `/analyze` endpoint
- **Input Validation**: Automatic validation using schema
- **Typed Response**: Structured output with SentimentResult

## Building and Deploying

### 1. Build the Image

```bash
chutes build sentiment_analyzer:chute --wait
```

### 2. Deploy the Chute

```bash
chutes deploy sentiment_analyzer:chute
```

### 3. Test Your Deployment

```bash
curl -X POST https://myuser-sentiment-analyzer.chutes.ai/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'
```

Expected response:

```json
{
	"text": "This is amazing!",
	"sentiment": "POSITIVE",
	"confidence": 0.99,
	"processing_time": 0.05
}
```

## Testing Different Texts

```python
import requests

texts = [
    "I love this product!",           # Should be POSITIVE
    "This is terrible.",              # Should be NEGATIVE
    "It's okay, nothing special.",    # Could be NEGATIVE or POSITIVE
    "Amazing technology!",            # Should be POSITIVE
    "Poor quality."                   # Should be NEGATIVE
]

for text in texts:
    response = requests.post(
        "https://myuser-sentiment-analyzer.chutes.ai/analyze",
        json={"text": text}
    )
    result = response.json()
    print(f"'{text}' -> {result['sentiment']} ({result['confidence']:.2f})")
```

## Key Concepts Learned

### 1. **Custom Images**

- How to build optimized Docker environments
- Installing Python packages efficiently
- Setting environment variables

### 2. **Model Management**

- Loading models at startup (not per request)
- GPU detection and utilization
- Memory optimization

### 3. **API Design**

- Input validation with Pydantic
- Structured responses
- Error handling

### 4. **Performance**

- Model reuse across requests
- Efficient tokenization
- GPU acceleration

## Next Steps

Now that you understand the basics, try:

- **[Streaming Responses](/docs/examples/streaming-responses)** - Real-time analysis
- **[Batch Processing](/docs/examples/batch-processing)** - Process multiple texts
- **[Multi-Model Setup](/docs/examples/multi-model-analysis)** - Combine multiple models
- **[Custom Image Building](/docs/guides/custom-images)** - Advanced Docker

## Common Issues & Solutions

**Model not loading?**

- Check GPU requirements in NodeSelector
- Verify model name is correct
- Ensure sufficient VRAM

**Slow responses?**

- Model loads on first request (normal)
- Consider warming up with health check
- Check GPU utilization

**Out of memory?**

- Reduce max_length in tokenizer
- Use smaller model variant
- Increase VRAM requirements
