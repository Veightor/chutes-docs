# Cords (API Endpoints)

**Cords** are the way you define HTTP API endpoints in your Chutes. Think of them as FastAPI routes, but with additional features for AI workloads like streaming, input validation, and automatic scaling.

## What is a Cord?

A Cord is a decorated function that becomes an HTTP API endpoint. The name comes from "parachute cord" - the connection between your chute and the outside world.

```python
@chute.cord(public_api_path="/predict")
async def predict(self, text: str) -> dict:
    result = await self.model.predict(text)
    return {"prediction": result}
```

This creates an endpoint accessible at `https://your-username-your-chute.chutes.ai/predict`.

## Basic Cord Definition

### Simple Cord

```python
from chutes.chute import Chute

chute = Chute(username="myuser", name="my-chute", image="my-image")

@chute.cord(public_api_path="/hello")
async def say_hello(self, name: str) -> dict:
    return {"message": f"Hello, {name}!"}
```

### With Input Validation

```python
from pydantic import BaseModel, Field

class GreetingInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    language: str = Field("en", regex="^(en|es|fr|de)$")

@chute.cord(
    public_api_path="/greet",
    input_schema=GreetingInput
)
async def greet(self, data: GreetingInput) -> dict:
    greetings = {
        "en": f"Hello, {data.name}!",
        "es": f"¡Hola, {data.name}!",
        "fr": f"Bonjour, {data.name}!",
        "de": f"Hallo, {data.name}!"
    }
    return {"greeting": greetings[data.language]}
```

## Cord Parameters

### Required Parameters

#### `public_api_path: str`

The URL path where your endpoint will be accessible.

```python
@chute.cord(public_api_path="/predict")  # https://user-chute.chutes.ai/predict
@chute.cord(public_api_path="/api/v1/generate")  # https://user-chute.chutes.ai/api/v1/generate
```

### Optional Parameters

#### `method: str = "POST"`

HTTP method for the endpoint.

```python
@chute.cord(public_api_path="/status", method="GET")
async def get_status(self) -> dict:
    return {"status": "healthy"}

@chute.cord(public_api_path="/update", method="PUT")
async def update_config(self, config: dict) -> dict:
    return {"updated": True}
```

#### `input_schema: BaseModel = None`

Pydantic model for automatic input validation and API documentation.

```python
from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    text: str = Field(..., description="Input text to analyze")
    max_length: int = Field(100, ge=1, le=1000, description="Maximum output length")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")

@chute.cord(
    public_api_path="/predict",
    input_schema=PredictionInput
)
async def predict(self, data: PredictionInput) -> dict:
    # Automatic validation and type conversion
    return await self.model.generate(
        data.text,
        max_length=data.max_length,
        temperature=data.temperature
    )
```

#### `minimal_input_schema: BaseModel = None`

Simplified input schema for easier testing and basic usage.

```python
class FullInput(BaseModel):
    text: str
    max_length: int = Field(100, ge=1, le=1000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)

class SimpleInput(BaseModel):
    text: str  # Only required field

@chute.cord(
    public_api_path="/generate",
    input_schema=FullInput,
    minimal_input_schema=SimpleInput  # For simpler API calls
)
async def generate(self, data: FullInput) -> dict:
    return await self.model.generate(data.text, **data.dict(exclude={'text'}))
```

#### `output_content_type: str = None`

Specify the content type of the response.

```python
@chute.cord(
    public_api_path="/generate-image",
    output_content_type="image/jpeg"
)
async def generate_image(self, prompt: str) -> Response:
    image_bytes = await self.model.generate_image(prompt)
    return Response(content=image_bytes, media_type="image/jpeg")

@chute.cord(
    public_api_path="/generate-audio",
    output_content_type="audio/wav"
)
async def generate_audio(self, text: str) -> Response:
    audio_bytes = await self.tts_model.synthesize(text)
    return Response(content=audio_bytes, media_type="audio/wav")
```

#### `stream: bool = False`

Enable streaming responses for real-time output.

```python
@chute.cord(
    public_api_path="/stream-generate",
    stream=True
)
async def stream_generate(self, prompt: str):
    # Yield tokens as they're generated
    async for token in self.model.generate_stream(prompt):
        yield {"token": token, "done": False}
    yield {"token": "", "done": True}
```

#### `passthrough: bool = False`

Proxy requests to another service running in the same container.

```python
@chute.cord(
    public_api_path="/v1/chat/completions",
    passthrough=True,
    passthrough_path="/v1/chat/completions",
    passthrough_port=8000
)
async def chat_completions(self, data):
    # Automatically forwards to localhost:8000/v1/chat/completions
    return data
```

## Function Signatures

### Self Parameter

All cord functions must take `self` as the first parameter, which provides access to the chute instance.

```python
@chute.cord(public_api_path="/predict")
async def predict(self, text: str) -> dict:
    # Access chute instance data
    result = await self.model.predict(text)
    self.request_count += 1
    return {"result": result, "count": self.request_count}
```

### Input Parameters

#### Direct Parameters

```python
@chute.cord(public_api_path="/simple")
async def simple_endpoint(self, text: str, temperature: float = 0.7) -> dict:
    return {"text": text, "temperature": temperature}
```

#### Pydantic Model Input

```python
@chute.cord(public_api_path="/validated", input_schema=MyInput)
async def validated_endpoint(self, data: MyInput) -> dict:
    return {"processed": data.text}
```

### Return Types

#### JSON Response (Default)

```python
@chute.cord(public_api_path="/json")
async def json_response(self, text: str) -> dict:
    return {"result": "processed"}  # Automatically serialized to JSON
```

#### Custom Response Objects

```python
from fastapi import Response

@chute.cord(public_api_path="/custom")
async def custom_response(self, data: str) -> Response:
    return Response(
        content="Custom content",
        media_type="text/plain",
        headers={"X-Custom-Header": "value"}
    )
```

#### Streaming Responses

```python
@chute.cord(public_api_path="/stream", stream=True)
async def streaming_response(self, prompt: str):
    for i in range(10):
        yield {"chunk": i, "data": f"Generated text {i}"}
```

## Advanced Features

### Error Handling

```python
from fastapi import HTTPException

@chute.cord(public_api_path="/predict")
async def predict(self, text: str) -> dict:
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        result = await self.model.predict(text)
        return {"prediction": result}
    except Exception as e:
        # Log the error
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
```

### Request Context

```python
from fastapi import Request

@chute.cord(public_api_path="/context")
async def with_context(self, request: Request, text: str) -> dict:
    # Access request metadata
    client_ip = request.client.host
    user_agent = request.headers.get("user-agent")

    return {
        "result": await self.model.predict(text),
        "metadata": {
            "client_ip": client_ip,
            "user_agent": user_agent
        }
    }
```

### File Uploads

```python
from fastapi import UploadFile, File

@chute.cord(public_api_path="/upload")
async def upload_file(self, file: UploadFile = File(...)) -> dict:
    contents = await file.read()

    # Process the uploaded file
    result = await self.process_file(contents, file.content_type)

    return {
        "filename": file.filename,
        "size": len(contents),
        "result": result
    }
```

### Response Headers

```python
from fastapi import Response

@chute.cord(public_api_path="/with-headers")
async def with_headers(self, text: str) -> dict:
    result = await self.model.predict(text)

    # Add custom headers (if returning Response object)
    response = Response(
        content=json.dumps({"result": result}),
        media_type="application/json"
    )
    response.headers["X-Processing-Time"] = "123ms"
    response.headers["X-Model-Version"] = self.model_version

    return response
```

## Streaming in Detail

### Text Streaming

```python
@chute.cord(public_api_path="/stream-text", stream=True)
async def stream_text(self, prompt: str):
    async for token in self.model.generate_stream(prompt):
        yield {
            "choices": [{
                "delta": {"content": token},
                "index": 0,
                "finish_reason": None
            }]
        }

    # Signal completion
    yield {
        "choices": [{
            "delta": {},
            "index": 0,
            "finish_reason": "stop"
        }]
    }
```

### Binary Streaming

```python
@chute.cord(
    public_api_path="/stream-audio",
    stream=True,
    output_content_type="audio/wav"
)
async def stream_audio(self, text: str):
    async for audio_chunk in self.tts_model.synthesize_stream(text):
        yield audio_chunk
```

### Server-Sent Events

```python
@chute.cord(
    public_api_path="/events",
    stream=True,
    output_content_type="text/event-stream"
)
async def server_sent_events(self, prompt: str):
    async for event in self.model.generate_events(prompt):
        yield f"data: {json.dumps(event)}\n\n"
```

## Best Practices

### 1. Input Validation

```python
from pydantic import BaseModel, Field, validator

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    language: str = Field("en", regex="^[a-z]{2}$")

    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()

@chute.cord(input_schema=TextInput)
async def process_text(self, data: TextInput) -> dict:
    # Input is guaranteed to be valid
    return await self.model.process(data.text, data.language)
```

### 2. Error Handling

```python
@chute.cord(public_api_path="/robust")
async def robust_endpoint(self, text: str) -> dict:
    try:
        # Validate input
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text is required")

        if len(text) > 10000:
            raise HTTPException(status_code=413, detail="Text too long")

        # Process request
        result = await self.model.predict(text)

        return {"result": result, "status": "success"}

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log unexpected errors
        logger.exception(f"Unexpected error in robust_endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### 3. Performance Optimization

```python
@chute.cord(public_api_path="/optimized")
async def optimized_endpoint(self, texts: list[str]) -> dict:
    # Batch processing for efficiency
    if len(texts) > 100:
        raise HTTPException(status_code=413, detail="Too many texts")

    # Process in batches
    results = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = await self.model.predict_batch(batch)
        results.extend(batch_results)

    return {"results": results}
```

### 4. Resource Management

```python
@chute.cord(public_api_path="/resource-managed")
async def resource_managed_endpoint(self, file_data: bytes) -> dict:
    temp_file = None
    try:
        # Create temporary resources
        temp_file = await self.create_temp_file(file_data)

        # Process
        result = await self.model.process_file(temp_file)

        return {"result": result}

    finally:
        # Always clean up
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
```

## Common Patterns

### Authentication

```python
from fastapi import Depends, HTTPException
import jwt

async def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, "secret", algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@chute.cord(public_api_path="/secure")
async def secure_endpoint(self, text: str, user=Depends(verify_token)) -> dict:
    return {
        "result": await self.model.predict(text),
        "user": user["username"]
    }
```

### Rate Limiting

```python
import time
from collections import defaultdict

# Simple in-memory rate limiter
request_counts = defaultdict(list)

@chute.cord(public_api_path="/rate-limited")
async def rate_limited_endpoint(self, request: Request, text: str) -> dict:
    client_ip = request.client.host
    current_time = time.time()

    # Clean old requests (older than 1 minute)
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip]
        if current_time - req_time < 60
    ]

    # Check rate limit (max 10 requests per minute)
    if len(request_counts[client_ip]) >= 10:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Record this request
    request_counts[client_ip].append(current_time)

    return await self.model.predict(text)
```

### Caching

```python
import hashlib
import json

@chute.on_startup()
async def setup_cache(self):
    self.cache = {}

@chute.cord(public_api_path="/cached")
async def cached_endpoint(self, text: str, temperature: float = 0.7) -> dict:
    # Create cache key
    cache_key = hashlib.md5(
        json.dumps({"text": text, "temperature": temperature}).encode()
    ).hexdigest()

    # Check cache
    if cache_key in self.cache:
        return {"result": self.cache[cache_key], "cached": True}

    # Compute result
    result = await self.model.predict(text, temperature=temperature)

    # Store in cache
    self.cache[cache_key] = result

    return {"result": result, "cached": False}
```

## Testing Cords

### Unit Testing

```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_predict_endpoint():
    async with AsyncClient(app=chute, base_url="http://test") as client:
        response = await client.post(
            "/predict",
            json={"text": "Hello world"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
```

### Local Testing

```python
if __name__ == "__main__":
    # Test locally before deployment
    import uvicorn
    uvicorn.run(chute, host="0.0.0.0", port=8000)
```

## Next Steps

- **[Jobs (Background Tasks)](/docs/core-concepts/jobs)** - Learn about long-running tasks
- **[Input/Output Schemas](/docs/guides/schemas)** - Deep dive into validation
- **[Streaming Responses](/docs/guides/streaming)** - Advanced streaming patterns
- **[Error Handling](/docs/guides/error-handling)** - Robust error management
