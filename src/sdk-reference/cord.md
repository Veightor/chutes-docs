# Cord Decorator API Reference

The `@chute.cord()` decorator is used to create HTTP API endpoints in Chutes applications. This reference covers all parameters, patterns, and best practices for building robust API endpoints.

## Decorator Signature

```python
@chute.cord(
    public_api_path: str,
    method: str = "POST",
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Dict] = None,
    minimal_input_schema: Optional[Type[BaseModel]] = None,
    output_content_type: str = "application/json",
    stream: bool = False,
    path: str = None,
    passthrough_path: str = None,
    passthrough: bool = False,
    passthrough_port: int = None,
    public_api_method: str = "POST",  # Internal parameter
    provision_timeout: int = 180,
    **session_kwargs
)
```

## Parameters

### Required Parameters

#### `public_api_path: str`

The URL path where the endpoint will be accessible.

**Format Rules:**

- Must start with `/`
- Can include path parameters with `{parameter_name}` syntax
- Should follow REST conventions
- Case-sensitive

**Examples:**

```python
# Simple path
@chute.cord(public_api_path="/generate")

# Path with parameter
@chute.cord(public_api_path="/users/{user_id}")

# Nested resource
@chute.cord(public_api_path="/models/{model_id}/generate")

# Multiple parameters
@chute.cord(public_api_path="/users/{user_id}/files/{file_id}")
```

**Best Practices:**

- Use kebab-case for multi-word paths: `/generate-text`
- Use nouns for resources: `/users`, `/models`
- Use verbs for actions: `/generate`, `/process`
- Keep paths concise but descriptive

### Optional Parameters

#### `method: str = "POST"`

The HTTP method for the endpoint.

**Supported Methods:**

- `GET` - Retrieve data
- `POST` - Create or process data
- `PUT` - Update existing data
- `DELETE` - Remove data
- `PATCH` - Partial updates
- `HEAD` - Get headers only
- `OPTIONS` - Get allowed methods

**Examples:**

```python
# GET for data retrieval
@chute.cord(public_api_path="/models", method="GET")
async def list_models(self):
    return {"models": ["gpt-3.5", "gpt-4"]}

# POST for data processing
@chute.cord(public_api_path="/generate", method="POST")
async def generate_text(self, prompt: str):
    return await self.model.generate(prompt)

# PUT for updates
@chute.cord(public_api_path="/config", method="PUT")
async def update_config(self, config: dict):
    self.config.update(config)
    return {"status": "updated"}

# DELETE for removal
@chute.cord(public_api_path="/cache", method="DELETE")
async def clear_cache(self):
    self.cache.clear()
    return {"status": "cache cleared"}
```

**Method Selection Guidelines:**

- Use `GET` for read-only operations
- Use `POST` for AI generation/processing
- Use `PUT` for complete resource updates
- Use `PATCH` for partial updates
- Use `DELETE` for resource removal

#### `input_schema: Optional[Type[BaseModel]] = None`

Pydantic model for input validation and documentation.

**Benefits:**

- Automatic input validation
- Auto-generated API documentation
- Type safety
- Error handling

**Basic Example:**

```python
from pydantic import BaseModel, Field

class TextGenerationInput(BaseModel):
    prompt: str = Field(..., description="Text prompt for generation")
    max_tokens: int = Field(100, ge=1, le=2000, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")

@chute.cord(
    public_api_path="/generate",
    method="POST",
    input_schema=TextGenerationInput
)
async def generate_text(self, input_data: TextGenerationInput):
    return await self.model.generate(
        input_data.prompt,
        max_tokens=input_data.max_tokens,
        temperature=input_data.temperature
    )
```

**Advanced Schema Example:**

```python
from typing import Optional, List, Literal
from pydantic import validator

class AdvancedGenerationInput(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    model_type: Literal["gpt-3.5", "gpt-4", "claude"] = Field("gpt-3.5")
    max_tokens: int = Field(100, ge=1, le=4000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    stop_sequences: Optional[List[str]] = Field(None, max_items=4)
    seed: Optional[int] = Field(None, ge=0)

    @validator('stop_sequences')
    def validate_stop_sequences(cls, v):
        if v and any(len(seq) > 10 for seq in v):
            raise ValueError('Stop sequences must be 10 characters or less')
        return v

    class Config:
        schema_extra = {
            "example": {
                "prompt": "Write a haiku about AI",
                "model_type": "gpt-4",
                "max_tokens": 50,
                "temperature": 0.8,
                "stop_sequences": ["\n\n"]
            }
        }

@chute.cord(
    public_api_path="/advanced_generate",
    input_schema=AdvancedGenerationInput
)
async def advanced_generate(self, params: AdvancedGenerationInput):
    return await self.model.generate(
        prompt=params.prompt,
        model=params.model_type,
        max_tokens=params.max_tokens,
        temperature=params.temperature,
        top_p=params.top_p,
        stop=params.stop_sequences,
        seed=params.seed
    )
```

#### `output_schema: Optional[Type[BaseModel]] = None`

Pydantic model for output validation and documentation.

**Example:**

```python
class GenerationOutput(BaseModel):
    generated_text: str = Field(..., description="The generated text")
    model_used: str = Field(..., description="Model that generated the text")
    tokens_used: int = Field(..., ge=0, description="Number of tokens consumed")
    processing_time: float = Field(..., gt=0, description="Processing time in seconds")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")

@chute.cord(
    public_api_path="/generate",
    input_schema=TextGenerationInput,
    output_schema=GenerationOutput
)
async def generate_with_metadata(self, input_data: TextGenerationInput) -> GenerationOutput:
    start_time = time.time()

    result = await self.model.generate(
        input_data.prompt,
        max_tokens=input_data.max_tokens,
        temperature=input_data.temperature
    )

    processing_time = time.time() - start_time

    return GenerationOutput(
        generated_text=result,
        model_used="gpt-3.5-turbo",
        tokens_used=len(result.split()),
        processing_time=processing_time,
        metadata={"timestamp": datetime.now().isoformat()}
    )
```

#### `minimal_input_schema: Optional[Type[BaseModel]] = None`

Simplified schema for basic API documentation and testing.

**Use Case:**
When you have a complex input schema but want to provide a simplified version for quick testing or documentation.

**Example:**

```python
class FullGenerationInput(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    seed: Optional[int] = None

class SimpleGenerationInput(BaseModel):
    prompt: str = Field(..., description="Just the prompt for quick testing")

@chute.cord(
    public_api_path="/generate",
    input_schema=FullGenerationInput,
    minimal_input_schema=SimpleGenerationInput
)
async def generate_flexible(self, input_data: FullGenerationInput):
    return await self.model.generate(**input_data.dict())
```

#### `output_content_type: str = "application/json"`

The MIME type of the response content.

**Common Content Types:**

- `application/json` - JSON responses (default)
- `text/plain` - Plain text
- `text/html` - HTML content
- `image/jpeg`, `image/png` - Images
- `audio/wav`, `audio/mpeg` - Audio files
- `video/mp4` - Video files
- `application/pdf` - PDF documents
- `text/event-stream` - Server-sent events

**JSON Response (Default):**

```python
@chute.cord(
    public_api_path="/generate",
    output_content_type="application/json"  # Optional, this is default
)
async def generate_json(self, prompt: str):
    result = await self.model.generate(prompt)
    return {"generated_text": result}
```

**Plain Text Response:**

```python
@chute.cord(
    public_api_path="/generate_text",
    output_content_type="text/plain"
)
async def generate_plain_text(self, prompt: str):
    return await self.model.generate(prompt)
```

**Image Response:**

```python
from fastapi import Response

@chute.cord(
    public_api_path="/generate_image",
    output_content_type="image/png"
)
async def generate_image(self, prompt: str) -> Response:
    image_data = await self.image_model.generate(prompt)

    return Response(
        content=image_data,
        media_type="image/png",
        headers={
            "Content-Disposition": "inline; filename=generated.png"
        }
    )
```

**Audio Response:**

```python
@chute.cord(
    public_api_path="/text_to_speech",
    output_content_type="audio/wav"
)
async def text_to_speech(self, text: str) -> Response:
    audio_data = await self.tts_model.synthesize(text)

    return Response(
        content=audio_data,
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=speech.wav"
        }
    )
```

#### `stream: bool = False`

Enable streaming responses for real-time data transmission.

**When to Use Streaming:**

- Long-running text generation
- Real-time progress updates
- Large data processing
- Interactive applications

**Basic Streaming Example:**

```python
from fastapi.responses import StreamingResponse
import json
import asyncio

@chute.cord(
    public_api_path="/stream_generate",
    stream=True
)
async def stream_text_generation(self, prompt: str):
    async def generate_stream():
        async for token in self.model.stream_generate(prompt):
            data = {"token": token, "finished": False}
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0.01)  # Small delay to prevent overwhelming

        # Send completion signal
        final_data = {"token": "", "finished": True}
        yield f"data: {json.dumps(final_data)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
```

**Progress Streaming Example:**

```python
@chute.cord(
    public_api_path="/process_large_file",
    stream=True
)
async def process_large_file(self, file_url: str):
    async def process_with_progress():
        total_steps = 100

        for step in range(total_steps):
            # Simulate processing
            await asyncio.sleep(0.1)

            progress_data = {
                "step": step + 1,
                "total_steps": total_steps,
                "progress_percent": ((step + 1) / total_steps) * 100,
                "status": "processing" if step < total_steps - 1 else "completed"
            }

            yield f"data: {json.dumps(progress_data)}\n\n"

    return StreamingResponse(
        process_with_progress(),
        media_type="text/event-stream"
    )
```

## Function Signature Patterns

### Simple Functions

```python
# Basic function with primitive parameters
@chute.cord(public_api_path="/simple")
async def simple_endpoint(self, text: str, number: int = 10):
    return {"text": text, "number": number}

# Function with optional parameters
@chute.cord(public_api_path="/optional")
async def optional_params(
    self,
    required_param: str,
    optional_param: Optional[str] = None,
    default_param: int = 100
):
    return {
        "required": required_param,
        "optional": optional_param,
        "default": default_param
    }
```

### Schema-Based Functions

```python
# Function with input schema
@chute.cord(
    public_api_path="/with_schema",
    input_schema=MyInputSchema
)
async def schema_endpoint(self, input_data: MyInputSchema):
    return await self.process(input_data)

# Function with both input and output schemas
@chute.cord(
    public_api_path="/full_schema",
    input_schema=MyInputSchema,
    output_schema=MyOutputSchema
)
async def full_schema_endpoint(self, input_data: MyInputSchema) -> MyOutputSchema:
    result = await self.process(input_data)
    return MyOutputSchema(**result)
```

### Path Parameter Functions

```python
# Single path parameter
@chute.cord(public_api_path="/users/{user_id}")
async def get_user(self, user_id: str):
    return {"user_id": user_id}

# Multiple path parameters
@chute.cord(public_api_path="/users/{user_id}/files/{file_id}")
async def get_user_file(self, user_id: str, file_id: str):
    return {"user_id": user_id, "file_id": file_id}

# Path parameters with body
@chute.cord(
    public_api_path="/users/{user_id}/generate",
    input_schema=GenerationInput
)
async def generate_for_user(self, user_id: str, params: GenerationInput):
    # Use user_id for personalization
    personalized_prompt = f"For user {user_id}: {params.prompt}"
    return await self.model.generate(personalized_prompt)
```

### Query Parameter Functions

```python
from fastapi import Query

@chute.cord(public_api_path="/search", method="GET")
async def search_endpoint(
    self,
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Number of results"),
    offset: int = Query(0, ge=0, description="Result offset"),
    sort_by: str = Query("relevance", description="Sort criteria")
):
    return {
        "query": query,
        "limit": limit,
        "offset": offset,
        "sort_by": sort_by,
        "results": []  # Your search results here
    }
```

### File Upload Functions

```python
from fastapi import File, UploadFile

@chute.cord(public_api_path="/upload", method="POST")
async def upload_file(self, file: UploadFile = File(...)):
    # Read file content
    content = await file.read()

    # Process file
    result = await self.process_file(content, file.filename)

    return {
        "filename": file.filename,
        "size": len(content),
        "content_type": file.content_type,
        "result": result
    }

# Multiple file upload
@chute.cord(public_api_path="/upload_multiple", method="POST")
async def upload_multiple_files(self, files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        content = await file.read()
        result = await self.process_file(content, file.filename)
        results.append({
            "filename": file.filename,
            "size": len(content),
            "result": result
        })

    return {"files_processed": len(results), "results": results}
```

## Advanced Patterns

### Error Handling

```python
from fastapi import HTTPException

@chute.cord(public_api_path="/generate_with_errors")
async def generate_with_error_handling(self, prompt: str):
    try:
        # Validate input
        if not prompt.strip():
            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty"
            )

        if len(prompt) > 10000:
            raise HTTPException(
                status_code=400,
                detail="Prompt too long (max 10,000 characters)"
            )

        # Process
        result = await self.model.generate(prompt)

        # Validate output
        if not result:
            raise HTTPException(
                status_code=500,
                detail="Model failed to generate output"
            )

        return {"generated_text": result}

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        # Log the error
        self.logger.error(f"Generation failed: {e}")

        # Return user-friendly error
        raise HTTPException(
            status_code=500,
            detail="Internal server error during text generation"
        )
```

### Authentication and Authorization

```python
from fastapi import Depends, Request, HTTPException

async def verify_api_key(request: Request):
    """Dependency for API key verification."""
    api_key = request.headers.get("X-API-Key")

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required"
        )

    # Validate API key (implement your logic)
    if not is_valid_api_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return api_key

@chute.cord(public_api_path="/protected_generate")
async def protected_generate(
    self,
    prompt: str,
    api_key: str = Depends(verify_api_key)
):
    """Protected endpoint requiring API key."""

    # Log usage for the API key
    self.logger.info(f"Generation request from API key: {api_key[:8]}...")

    result = await self.model.generate(prompt)
    return {"generated_text": result}
```

### Rate Limiting

```python
import time
from collections import defaultdict
from fastapi import Request, HTTPException

class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        self.window_size = 60  # 1 minute
        self.max_requests = 100

    def is_allowed(self, identifier: str) -> bool:
        now = time.time()
        window_start = now - self.window_size

        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if req_time > window_start
        ]

        # Check limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False

        # Add current request
        self.requests[identifier].append(now)
        return True

# Initialize rate limiter
rate_limiter = RateLimiter()

async def check_rate_limit(request: Request):
    """Rate limiting dependency."""
    identifier = request.client.host  # Use IP address

    if not rate_limiter.is_allowed(identifier):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )

@chute.cord(public_api_path="/rate_limited_generate")
async def rate_limited_generate(
    self,
    prompt: str,
    rate_limit_check: None = Depends(check_rate_limit)
):
    return await self.model.generate(prompt)
```

### Response Customization

```python
from fastapi import Response
from datetime import datetime

@chute.cord(public_api_path="/custom_response")
async def custom_response_endpoint(self, prompt: str):
    result = await self.model.generate(prompt)

    # Create custom response with headers
    response_data = {
        "generated_text": result,
        "timestamp": datetime.now().isoformat(),
        "model_version": "1.0.0"
    }

    response = Response(
        content=json.dumps(response_data),
        media_type="application/json",
        headers={
            "X-Generation-Time": str(time.time()),
            "X-Model-Version": "1.0.0",
            "Cache-Control": "no-cache"
        }
    )

    return response
```

### Conditional Processing

```python
@chute.cord(public_api_path="/conditional_generate")
async def conditional_generate(
    self,
    prompt: str,
    use_cache: bool = True,
    model_version: str = "default"
):
    # Check cache if enabled
    if use_cache:
        cached_result = await self.get_from_cache(prompt)
        if cached_result:
            return {
                "generated_text": cached_result,
                "from_cache": True
            }

    # Select model version
    if model_version == "fast":
        result = await self.fast_model.generate(prompt)
    elif model_version == "accurate":
        result = await self.accurate_model.generate(prompt)
    else:
        result = await self.default_model.generate(prompt)

    # Cache result if caching is enabled
    if use_cache:
        await self.save_to_cache(prompt, result)

    return {
        "generated_text": result,
        "from_cache": False,
        "model_version": model_version
    }
```

## Testing Endpoints

### Unit Testing

```python
import pytest
from fastapi.testclient import TestClient

def test_basic_endpoint():
    """Test basic endpoint functionality."""
    client = TestClient(chute.app)

    response = client.post(
        "/generate",
        json={"prompt": "Test prompt"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "generated_text" in data
    assert isinstance(data["generated_text"], str)

def test_input_validation():
    """Test input validation."""
    client = TestClient(chute.app)

    # Test missing required field
    response = client.post("/generate", json={})
    assert response.status_code == 422

    # Test invalid data type
    response = client.post("/generate", json={"prompt": 123})
    assert response.status_code == 422

    # Test valid input
    response = client.post("/generate", json={"prompt": "Valid prompt"})
    assert response.status_code == 200

def test_error_handling():
    """Test error handling."""
    client = TestClient(chute.app)

    # Test empty prompt
    response = client.post("/generate", json={"prompt": ""})
    assert response.status_code == 400

    # Test prompt too long
    long_prompt = "x" * 20000
    response = client.post("/generate", json={"prompt": long_prompt})
    assert response.status_code == 400
```

### Integration Testing

```python
import asyncio

async def test_streaming_endpoint():
    """Test streaming endpoint."""
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/stream_generate",
            json={"prompt": "Test streaming"}
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream"

            tokens = []
            async for chunk in response.aiter_text():
                if chunk.startswith("data: "):
                    data = json.loads(chunk[6:])
                    if data.get("token"):
                        tokens.append(data["token"])
                    if data.get("finished"):
                        break

            assert len(tokens) > 0
```

## Best Practices

### API Design

1. **Use Clear, Descriptive Paths**

   ```python
   # Good
   @chute.cord(public_api_path="/generate_text")
   @chute.cord(public_api_path="/analyze_sentiment")

   # Avoid
   @chute.cord(public_api_path="/api")
   @chute.cord(public_api_path="/process")
   ```

2. **Choose Appropriate HTTP Methods**

   ```python
   # GET for data retrieval
   @chute.cord(public_api_path="/models", method="GET")

   # POST for data processing
   @chute.cord(public_api_path="/generate", method="POST")

   # PUT for full updates
   @chute.cord(public_api_path="/config", method="PUT")
   ```

3. **Use Input Schemas for Validation**

   ```python
   # Always define input schemas for complex inputs
   @chute.cord(
       public_api_path="/generate",
       input_schema=GenerationInput
   )
   ```

4. **Provide Clear Error Messages**

   ```python
   if not prompt.strip():
       raise HTTPException(
           status_code=400,
           detail="Prompt cannot be empty"
       )
   ```

5. **Use Consistent Response Formats**

   ```python
   # Consistent success format
   return {
       "status": "success",
       "data": result,
       "metadata": {...}
   }

   # Consistent error format
   raise HTTPException(
       status_code=400,
       detail={
           "error": "Validation failed",
           "details": {...}
       }
   )
   ```

This comprehensive guide covers all aspects of the `@chute.cord()` decorator. For more examples and patterns, see the [Examples](../examples/) section.
