# Chute API Reference

The `Chute` class is the core component of the Chutes framework, representing a deployable AI application unit. This reference covers all methods, properties, and configuration options.

## Class Definition

```python
from chutes.chute import Chute

chute = Chute(
    username: str,
    name: str,
    image: str | Image,
    tagline: str = "",
    readme: str = "",
    standard_template: str = None,
    revision: str = None,
    node_selector: NodeSelector = None,
    concurrency: int = 1,
    **kwargs
)
```

## Constructor Parameters

### Required Parameters

#### `username: str`

The username or organization name for the chute deployment.

**Example:**

```python
chute = Chute(username="mycompany", name="ai-service", image="python:3.11")
```

**Rules:**

- Must be lowercase alphanumeric with hyphens
- Cannot start or end with hyphen
- Maximum 63 characters
- Must be unique within the platform

#### `name: str`

The name of the chute application.

**Example:**

```python
chute = Chute(username="mycompany", name="text-generator", image="python:3.11")
```

**Rules:**

- Must be lowercase alphanumeric with hyphens
- Cannot start or end with hyphen
- Maximum 63 characters
- Must be unique within the username namespace

#### `image: str | Image`

Docker image for the chute runtime environment (required).

**Example:**

```python
# Using a string reference
chute = Chute(
    username="mycompany",
    name="text-generator",
    image="python:3.11"
)

# Using a custom Image object
from chutes.image import Image
custom_image = Image(username="mycompany", name="custom-ai", tag="1.0")
chute = Chute(
    username="mycompany",
    name="text-generator",
    image=custom_image
)
```

### Optional Parameters

#### `tagline: str = ""`

A brief description of what the chute does.

**Example:**

```python
chute = Chute(
    username="mycompany",
    name="text-generator",
    image="python:3.11",
    tagline="Advanced text generation with GPT models"
)
```

**Best Practices:**

- Keep under 100 characters
- Use present tense
- Be descriptive but concise

#### `readme: str = ""`

Detailed documentation for the chute in Markdown format.

**Example:**

```python
readme = """
# Text Generation API

This chute provides advanced text generation capabilities using state-of-the-art language models.

## Features
- Multiple model support
- Customizable parameters
- High-performance inference
- Real-time streaming

## Usage
Send a POST request to `/generate` with your prompt.
"""

chute = Chute(
    username="mycompany",
    name="text-generator",
    image="python:3.11",
    readme=readme
)
```

#### `standard_template: str = None`

Reference to a standard template to use as a base for the chute.

**Example:**

```python
chute = Chute(
    username="mycompany",
    name="text-generator",
    image="python:3.11",
    standard_template="vllm"
)
```

#### `revision: str = None`

Specific revision or version identifier for the chute.

**Example:**

```python
chute = Chute(
    username="mycompany",
    name="text-generator",
    image="python:3.11",
    revision="v1.2.0"
)
```

#### `node_selector: NodeSelector = None`

Hardware requirements and preferences for the chute.

**Example:**

```python
from chutes.chute import NodeSelector

node_selector = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=24,
    include=["h100", "a100"],
    exclude=["t4"]
)

chute = Chute(
    username="mycompany",
    name="text-generator",
    image="python:3.11",
    node_selector=node_selector
)
```

#### `concurrency: int = 1`

Maximum number of concurrent requests the chute can handle.

**Example:**

```python
chute = Chute(
    username="mycompany",
    name="text-generator",
    concurrency=8  # Handle up to 8 concurrent requests
)
```

**Guidelines:**

- Higher concurrency requires more memory
- Consider model size and GPU memory
- Typical values: 1-16 for most applications

#### `**kwargs`

Additional keyword arguments passed to the underlying FastAPI application.

**Example:**

```python
chute = Chute(
    username="mycompany",
    name="text-generator",
    image="python:3.11",
    title="My AI API",  # FastAPI title
    description="Custom AI service",  # FastAPI description
    version="1.0.0"  # API version
)
```

## Decorators and Methods

### Lifecycle Decorators

#### `@chute.on_startup()`

Decorator for functions to run during chute startup.

**Signature:**

```python
@chute.on_startup()
async def initialization_function(self) -> None:
    """Function to run on startup."""
    pass
```

**Example:**

```python
@chute.on_startup()
async def load_model(self):
    """Load the AI model during startup."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
    self.model = AutoModelForCausalLM.from_pretrained("gpt2")

    print("Model loaded successfully")
```

**Use Cases:**

- Load AI models
- Initialize databases
- Set up caches
- Configure services
- Load configuration

**Best Practices:**

- Use async functions for I/O operations
- Add error handling
- Log initialization steps
- Keep startup time reasonable

#### `@chute.on_shutdown()`

Decorator for functions to run during chute shutdown.

**Signature:**

```python
@chute.on_shutdown()
async def cleanup_function(self) -> None:
    """Function to run on shutdown."""
    pass
```

**Example:**

```python
@chute.on_shutdown()
async def cleanup_resources(self):
    """Clean up resources during shutdown."""
    if hasattr(self, 'model'):
        del self.model

    if hasattr(self, 'database_connection'):
        await self.database_connection.close()

    print("Resources cleaned up")
```

**Use Cases:**

- Free memory (unload models)
- Close database connections
- Save state
- Clean temporary files
- Log shutdown events

### API Endpoint Decorators

#### `@chute.cord()`

Decorator to create HTTP API endpoints.

**Signature:**

```python
@chute.cord(
    public_api_path: str,
    method: str = "POST",
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    minimal_input_schema: Optional[Type[BaseModel]] = None,
    output_content_type: str = "application/json",
    stream: bool = False,
    public_api_method: Optional[str] = None  # Deprecated: use 'method'
)
```

**Parameters:**

- **`public_api_path: str`** - URL path for the endpoint (required)
- **`method: str = "POST"`** - HTTP method (GET, POST, PUT, DELETE, etc.)
- **`input_schema: Optional[Type[BaseModel]]`** - Pydantic schema for input validation
- **`output_schema: Optional[Type[BaseModel]]`** - Pydantic schema for output validation
- **`minimal_input_schema: Optional[Type[BaseModel]]`** - Simplified schema for documentation
- **`output_content_type: str`** - Response content type
- **`stream: bool = False`** - Enable streaming responses

**Basic Example:**

```python
@chute.cord(public_api_path="/generate", method="POST")
async def generate_text(self, prompt: str) -> str:
    """Generate text from a prompt."""
    return await self.model.generate(prompt)
```

**Advanced Example with Schemas:**

```python
from pydantic import BaseModel, Field

class GenerationInput(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(100, ge=1, le=1000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)

class GenerationOutput(BaseModel):
    generated_text: str = Field(..., description="Generated text")
    tokens_used: int = Field(..., description="Number of tokens used")

@chute.cord(
    public_api_path="/generate",
    method="POST",
    input_schema=GenerationInput,
    output_schema=GenerationOutput
)
async def generate_text(self, params: GenerationInput) -> GenerationOutput:
    """Generate text with parameters."""
    result = await self.model.generate(
        params.prompt,
        max_tokens=params.max_tokens,
        temperature=params.temperature
    )

    return GenerationOutput(
        generated_text=result,
        tokens_used=len(result.split())
    )
```

**Non-JSON Responses:**

```python
from fastapi import Response

@chute.cord(
    public_api_path="/generate_image",
    method="POST",
    output_content_type="image/png"
)
async def generate_image(self, prompt: str) -> Response:
    """Generate an image and return as PNG."""
    image_data = await self.model.generate_image(prompt)

    return Response(
        content=image_data,
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=generated.png"}
    )
```

**Streaming Example:**

```python
from fastapi.responses import StreamingResponse

@chute.cord(
    public_api_path="/stream_generate",
    method="POST",
    stream=True
)
async def stream_generate(self, prompt: str):
    """Stream text generation token by token."""

    async def generate_tokens():
        async for token in self.model.stream_generate(prompt):
            yield f"data: {json.dumps({'token': token})}\n\n"

    return StreamingResponse(
        generate_tokens(),
        media_type="text/event-stream"
    )
```

#### `@chute.job()`

Decorator to create background job handlers.

**Signature:**

```python
@chute.job(
    name: Optional[str] = None,
    schedule: Optional[str] = None,
    retry_count: int = 3,
    timeout: Optional[int] = None
)
```

**Parameters:**

- **`name: Optional[str]`** - Job name (defaults to function name)
- **`schedule: Optional[str]`** - Cron schedule for recurring jobs
- **`retry_count: int = 3`** - Number of retry attempts on failure
- **`timeout: Optional[int]`** - Job timeout in seconds

**Basic Job Example:**

```python
@chute.job(name="data_processing")
async def process_data(self, data: dict) -> dict:
    """Process data in the background."""

    # Simulate processing
    await asyncio.sleep(5)

    return {
        "status": "completed",
        "processed_items": len(data.get("items", []))
    }
```

**Scheduled Job Example:**

```python
@chute.job(
    name="daily_cleanup",
    schedule="0 2 * * *"  # Run daily at 2 AM
)
async def daily_cleanup(self):
    """Daily cleanup job."""

    # Clean up temporary files
    temp_files_cleaned = await self.cleanup_temp_files()

    # Clear old cache entries
    cache_entries_cleared = await self.clear_old_cache()

    return {
        "temp_files_cleaned": temp_files_cleaned,
        "cache_entries_cleared": cache_entries_cleared
    }
```

**Job with Retry:**

```python
@chute.job(
    name="external_api_call",
    retry_count=5,
    timeout=30
)
async def call_external_api(self, endpoint: str, data: dict):
    """Call external API with retry logic."""

    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, json=data) as response:
            if response.status != 200:
                raise Exception(f"API call failed: {response.status}")

            return await response.json()
```

#### `@chute.websocket()`

Decorator to create WebSocket endpoints.

**Signature:**

```python
@chute.websocket(path: str)
```

**Example:**

```python
from fastapi import WebSocket

@chute.websocket("/chat")
async def websocket_chat(self, websocket: WebSocket):
    """WebSocket endpoint for real-time chat."""

    await websocket.accept()

    try:
        while True:
            # Receive message
            message = await websocket.receive_text()

            # Process with AI model
            response = await self.model.generate(message)

            # Send response
            await websocket.send_text(response)

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
```

## Properties and Attributes

### Runtime Properties

#### `chute.app`

The underlying FastAPI application instance.

**Type:** `fastapi.FastAPI`

**Example:**

```python
# Access the FastAPI app directly
@chute.on_startup()
async def configure_app(self):
    # Add custom middleware
    from fastapi.middleware.cors import CORSMiddleware

    self.app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )

    # Add custom route
    @self.app.get("/custom")
    async def custom_endpoint():
        return {"message": "Custom endpoint"}
```

#### `chute.logger`

Built-in logger for the chute.

**Type:** `logging.Logger`

**Example:**

```python
@chute.on_startup()
async def setup_logging(self):
    self.logger.info("Chute startup initiated")

@chute.cord(public_api_path="/generate")
async def generate(self, prompt: str):
    self.logger.info(f"Generating text for prompt: {prompt[:50]}...")

    try:
        result = await self.model.generate(prompt)
        self.logger.info("Text generation completed successfully")
        return result
    except Exception as e:
        self.logger.error(f"Text generation failed: {e}")
        raise
```

### Configuration Properties

#### `chute.config`

Access to chute configuration.

**Example:**

```python
@chute.on_startup()
async def log_config(self):
    self.logger.info(f"Chute name: {self.config.name}")
    self.logger.info(f"Concurrency: {self.config.concurrency}")
    self.logger.info(f"Username: {self.config.username}")
```

## Method Reference

### Utility Methods

#### `chute.run_local()`

Run the chute locally for development and testing.

**Signature:**

```python
def run_local(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = True,
    log_level: str = "info"
) -> None
```

**Example:**

```python
if __name__ == "__main__":
    chute.run_local(
        host="127.0.0.1",
        port=8080,
        reload=True,
        log_level="debug"
    )
```

**Parameters:**

- **`host: str`** - Host to bind to
- **`port: int`** - Port to listen on
- **`reload: bool`** - Enable auto-reload on code changes
- **`log_level: str`** - Logging level (debug, info, warning, error)

#### `chute.health_check()`

Built-in health check method.

**Signature:**

```python
async def health_check(self) -> dict
```

**Example:**

```python
# Access health check programmatically
health_status = await chute.health_check()
print(health_status)
# Output: {"status": "healthy", "timestamp": "2024-01-01T12:00:00Z"}
```

**Custom Health Checks:**

```python
@chute.on_startup()
async def setup_custom_health_check(self):
    """Override default health check."""

    async def custom_health_check():
        try:
            # Check model availability
            test_result = await self.model.generate("test", max_tokens=1)

            # Check external dependencies
            # ... additional checks ...

            return {
                "status": "healthy",
                "model_loaded": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    # Replace default health check
    self.health_check = custom_health_check
```

## Advanced Usage Patterns

### State Management

```python
@chute.on_startup()
async def initialize_state(self):
    """Initialize application state."""

    # Model state
    self.model = None
    self.model_loaded = False

    # Cache state
    self.cache = {}
    self.cache_hits = 0
    self.cache_misses = 0

    # Request tracking
    self.request_count = 0
    self.error_count = 0

@chute.cord(public_api_path="/generate")
async def generate_with_tracking(self, prompt: str):
    """Generate text with state tracking."""

    self.request_count += 1

    try:
        # Check cache
        cache_key = hash(prompt)
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        # Generate new result
        result = await self.model.generate(prompt)

        # Update cache
        self.cache[cache_key] = result
        self.cache_misses += 1

        return result

    except Exception as e:
        self.error_count += 1
        raise

@chute.cord(public_api_path="/stats", method="GET")
async def get_stats(self):
    """Get application statistics."""
    return {
        "requests": self.request_count,
        "errors": self.error_count,
        "cache_hits": self.cache_hits,
        "cache_misses": self.cache_misses,
        "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
    }
```

### Dependency Injection

```python
from typing import Protocol

class ModelInterface(Protocol):
    async def generate(self, prompt: str) -> str: ...

class CacheInterface(Protocol):
    async def get(self, key: str) -> Optional[str]: ...
    async def set(self, key: str, value: str) -> None: ...

@chute.on_startup()
async def setup_dependencies(self):
    """Set up dependency injection."""

    # Create implementations
    from my_models import GPTModel
    from my_cache import RedisCache

    self.model: ModelInterface = GPTModel()
    self.cache: CacheInterface = RedisCache()

    # Initialize
    await self.model.load()
    await self.cache.connect()

@chute.cord(public_api_path="/generate")
async def generate_with_deps(self, prompt: str):
    """Generate using injected dependencies."""

    # Check cache
    cached = await self.cache.get(prompt)
    if cached:
        return cached

    # Generate
    result = await self.model.generate(prompt)

    # Cache result
    await self.cache.set(prompt, result)

    return result
```

### Error Handling

```python
from fastapi import HTTPException

@chute.on_startup()
async def setup_error_handling(self):
    """Set up centralized error handling."""

    @self.app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        self.logger.error(f"Unhandled exception: {exc}")

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "request_id": getattr(request.state, "request_id", None)
            }
        )

@chute.cord(public_api_path="/generate")
async def generate_with_error_handling(self, prompt: str):
    """Generate with comprehensive error handling."""

    try:
        # Validate input
        if not prompt or len(prompt.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty"
            )

        if len(prompt) > 10000:
            raise HTTPException(
                status_code=400,
                detail="Prompt too long (max 10,000 characters)"
            )

        # Generate
        result = await self.model.generate(prompt)

        if not result:
            raise HTTPException(
                status_code=500,
                detail="Model failed to generate output"
            )

        return {"generated_text": result}

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        self.logger.error(f"Generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Text generation failed"
        )
```

## Best Practices

### Performance Optimization

```python
import asyncio
from functools import lru_cache

@chute.on_startup()
async def optimize_performance(self):
    """Set up performance optimizations."""

    # Connection pooling
    self.session = aiohttp.ClientSession()

    # Model batching
    self.batch_queue = asyncio.Queue()
    self.batch_size = 8

    # Start batch processor
    asyncio.create_task(self.process_batches())

@lru_cache(maxsize=1000)
def expensive_computation(self, input_data: str) -> str:
    """Cache expensive computations."""
    # Expensive operation here
    return result

async def process_batches(self):
    """Process requests in batches for efficiency."""
    while True:
        batch = []

        # Collect batch
        try:
            for _ in range(self.batch_size):
                item = await asyncio.wait_for(
                    self.batch_queue.get(),
                    timeout=0.1
                )
                batch.append(item)
        except asyncio.TimeoutError:
            pass

        if batch:
            # Process batch
            results = await self.model.generate_batch([item['prompt'] for item in batch])

            # Return results
            for item, result in zip(batch, results):
                item['future'].set_result(result)

@chute.cord(public_api_path="/generate_batched")
async def generate_batched(self, prompt: str):
    """Generate using batched processing."""

    future = asyncio.Future()

    await self.batch_queue.put({
        'prompt': prompt,
        'future': future
    })

    result = await future
    return {"generated_text": result}
```

### Security Best Practices

```python
import hashlib
import hmac
from fastapi import Request, HTTPException

@chute.on_startup()
async def setup_security(self):
    """Set up security measures."""

    # API key validation
    self.valid_api_keys = {"your-secret-api-key"}

    # Rate limiting
    self.rate_limits = {}
    self.rate_limit_window = 60  # seconds
    self.rate_limit_max = 100   # requests per window

async def validate_api_key(self, request: Request):
    """Validate API key from request."""

    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key not in self.valid_api_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )

async def check_rate_limit(self, request: Request):
    """Check rate limiting."""

    client_ip = request.client.host
    now = time.time()

    # Clean old entries
    self.rate_limits = {
        ip: times for ip, times in self.rate_limits.items()
        if any(t > now - self.rate_limit_window for t in times)
    }

    # Check current IP
    if client_ip not in self.rate_limits:
        self.rate_limits[client_ip] = []

    recent_requests = [
        t for t in self.rate_limits[client_ip]
        if t > now - self.rate_limit_window
    ]

    if len(recent_requests) >= self.rate_limit_max:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )

    # Record current request
    self.rate_limits[client_ip].append(now)

@chute.cord(public_api_path="/secure_generate")
async def secure_generate(self, prompt: str, request: Request):
    """Secure generation endpoint."""

    # Security checks
    await self.validate_api_key(request)
    await self.check_rate_limit(request)

    # Input sanitization
    prompt = prompt.strip()
    if len(prompt) > 5000:
        raise HTTPException(400, "Prompt too long")

    # Generate
    result = await self.model.generate(prompt)

    return {"generated_text": result}
```

## Common Patterns and Examples

### Multi-Model Chute

```python
@chute.on_startup()
async def load_multiple_models(self):
    """Load multiple models for different tasks."""

    self.text_model = TextModel()
    self.image_model = ImageModel()
    self.embedding_model = EmbeddingModel()

    await asyncio.gather(
        self.text_model.load(),
        self.image_model.load(),
        self.embedding_model.load()
    )

@chute.cord(public_api_path="/generate_text")
async def generate_text(self, prompt: str):
    return await self.text_model.generate(prompt)

@chute.cord(public_api_path="/generate_image")
async def generate_image(self, prompt: str):
    return await self.image_model.generate(prompt)

@chute.cord(public_api_path="/embed_text")
async def embed_text(self, text: str):
    return await self.embedding_model.embed(text)
```

### Template-Based Chute

```python
from chutes.templates import build_vllm_chute

# Use template as base
chute = build_vllm_chute(
    username="mycompany",
    name="custom-llm",
    model_name="gpt2"
)

# Add custom functionality
@chute.cord(public_api_path="/custom_generate")
async def custom_generate(self, prompt: str, style: str = "formal"):
    """Custom generation with style control."""

    style_prompts = {
        "formal": "Please respond in a formal tone: ",
        "casual": "Please respond casually: ",
        "technical": "Please provide a technical explanation: "
    }

    styled_prompt = style_prompts.get(style, "") + prompt

    result = await self.generate(styled_prompt)
    return {"generated_text": result, "style": style}
```

This comprehensive API reference covers all aspects of the Chute class. For specific implementation examples, see the [Examples](../examples/) section and [Templates Guide](../templates/).
