# Understanding Chutes

A **Chute** is the fundamental building block of the Chutes platform. Think of it as a complete AI application that can be deployed to GPU-accelerated infrastructure with just a few lines of code.

## What is a Chute?

A Chute is essentially a **FastAPI application** with superpowers for AI workloads. It provides:

- 🚀 **Serverless deployment** to GPU clusters
- 🔌 **Simple decorator-based API** definition
- 🏗️ **Custom Docker image** building
- ⚡ **Hardware resource** specification
- 📊 **Automatic scaling** based on demand
- 💰 **Pay-per-use** billing

## Basic Chute Structure

```python
from chutes.chute import Chute, NodeSelector
from chutes.image import Image

# Define your custom image (optional)
image = (
    Image(username="myuser", name="my-ai-app", tag="1.0")
    .from_base("nvidia/cuda:12.2-devel-ubuntu22.04")
    .with_python("3.11")
    .run_command("pip install torch transformers")
)

# Create your chute
chute = Chute(
    username="myuser",
    name="my-ai-app",
    image=image,  # or use a string like "my-custom-image:latest"
    tagline="My awesome AI application",
    readme="# My AI App\nThis app does amazing things!",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16
    ),
    concurrency=4
)

# Add startup initialization
@chute.on_startup()
async def initialize_model(self):
    import torch
    from transformers import AutoModel, AutoTokenizer

    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model = AutoModel.from_pretrained("bert-base-uncased")
    self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Move model to GPU
    self.model.to(self.device)

# Define API endpoints
@chute.cord(public_api_path="/predict")
async def predict(self, text: str) -> dict:
    inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
    with torch.no_grad():
        outputs = self.model(**inputs)
    return {"prediction": outputs.last_hidden_state.mean().item()}
```

## Chute Constructor Parameters

### Required Parameters

#### `username: str`

Your Chutes platform username. This is used for:

- Image naming and organization
- URL generation (`username-chute-name.chutes.ai`)
- Access control and billing

```python
chute = Chute(username="myuser", ...)  # Required
```

#### `name: str`

The name of your chute. Must be:

- Alphanumeric with hyphens/underscores
- Unique within your account
- Used in the public URL

```python
chute = Chute(name="my-awesome-app", ...)  # Required
```

#### `image: str | Image`

The Docker image to use. Can be:

- A string reference to an existing image: `"nvidia/cuda:12.2-runtime-ubuntu22.04"`
- A custom `Image` object with build instructions
- A pre-built template image: `"chutes/vllm:latest"`

```python
# Using a string reference
chute = Chute(image="nvidia/cuda:12.2-runtime-ubuntu22.04", ...)

# Using a custom Image object
from chutes.image import Image
custom_image = Image(username="myuser", name="my-image", tag="1.0")
chute = Chute(image=custom_image, ...)
```

### Optional Parameters

#### `tagline: str = ""`

A short description displayed in the Chutes dashboard and API listings.

```python
chute = Chute(tagline="Fast text generation with custom models", ...)
```

#### `readme: str = ""`

Markdown documentation for your chute. Supports full markdown syntax.

````python
chute = Chute(
    readme="""
    # My AI Application

    This chute provides text generation capabilities using a fine-tuned model.

    ## Usage
    ```bash
    curl -X POST https://myuser-myapp.chutes.ai/generate \\
         -d '{"prompt": "Hello world"}'
    ```

    ## Features
    - Fast inference
    - Streaming support
    - Custom fine-tuning
    """,
    ...
)
````

#### `node_selector: NodeSelector = None`

Hardware requirements for your chute. If not specified, uses default settings.

```python
from chutes.chute import NodeSelector

chute = Chute(
    node_selector=NodeSelector(
        gpu_count=2,
        min_vram_gb_per_gpu=24,
        include=["a100", "h100"],  # Preferred GPU types
        exclude=["k80", "p100"]    # Avoid older GPUs
    ),
    ...
)
```

#### `concurrency: int = 1`

Maximum number of simultaneous requests each instance can handle.

```python
# Handle up to 8 requests simultaneously
chute = Chute(concurrency=8, ...)
```

#### `revision: str = None`

Version control for your chute deployment.

```python
chute = Chute(revision="v1.2.0", ...)
```

#### `standard_template: str = None`

Used internally by template builders. Generally not set manually.

## Chute Methods

### Lifecycle Methods

#### `@chute.on_startup()`

Decorator for functions that run when your chute starts up. Use this for:

- Model loading and initialization
- Database connections
- Preprocessing setup

```python
@chute.on_startup()
async def load_model(self):
    # This runs once when the chute starts
    self.model = load_my_model()
    self.preprocessor = setup_preprocessing()
```

#### `@chute.on_shutdown()`

Decorator for cleanup functions that run when your chute shuts down.

```python
@chute.on_shutdown()
async def cleanup(self):
    # This runs when the chute is shutting down
    if hasattr(self, 'database'):
        await self.database.close()
```

### API Definition Methods

#### `@chute.cord(...)`

Define HTTP API endpoints. See [Cords Documentation](/docs/core-concepts/cords) for details.

```python
@chute.cord(
    public_api_path="/predict",
    method="POST",
    input_schema=MyInputSchema,
    output_content_type="application/json"
)
async def predict(self, data: MyInputSchema) -> dict:
    return {"result": "prediction"}
```

#### `@chute.job(...)`

Define background jobs or long-running tasks. See [Jobs Documentation](/docs/core-concepts/jobs) for details.

```python
@chute.job(timeout=3600, upload=True)
async def train_model(self, training_data: dict):
    # Long-running training job
    pass
```

## Chute Properties

### Read-Only Properties

```python
# Access chute metadata
print(chute.name)           # Chute name
print(chute.uid)            # Unique identifier
print(chute.username)       # Owner username
print(chute.tagline)        # Short description
print(chute.readme)         # Documentation
print(chute.node_selector)  # Hardware requirements
print(chute.image)          # Docker image reference
print(chute.cords)          # List of API endpoints
print(chute.jobs)           # List of background jobs
```

## Advanced Usage

### Custom Context Management

You can store data in the chute instance that persists across requests:

```python
@chute.on_startup()
async def setup(self):
    # This data persists for the lifetime of the chute instance
    self.cache = {}
    self.request_count = 0

@chute.cord(public_api_path="/cached-predict")
async def cached_predict(self, text: str) -> dict:
    # Access persistent data
    self.request_count += 1

    if text in self.cache:
        return self.cache[text]

    result = await expensive_computation(text)
    self.cache[text] = result
    return result
```

### Integration with FastAPI Features

Since Chute extends FastAPI, you can use FastAPI features directly:

```python
from fastapi import HTTPException, Depends

@chute.cord(public_api_path="/secure-endpoint")
async def secure_endpoint(self, data: str, api_key: str = Depends(validate_api_key)):
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return {"secure_data": process_data(data)}
```

### Environment Variables

Access environment variables in your chute:

```python
import os

@chute.on_startup()
async def configure(self):
    self.debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    self.model_path = os.getenv("MODEL_PATH", "/app/models/default")
```

## Best Practices

### 1. Resource Management

```python
@chute.on_startup()
async def initialize(self):
    # Pre-load models and resources
    self.model = load_model()  # Do this once, not per request

@chute.on_shutdown()
async def cleanup(self):
    # Clean up resources
    if hasattr(self, 'model'):
        del self.model
```

### 2. Error Handling

```python
@chute.cord(public_api_path="/predict")
async def predict(self, text: str) -> dict:
    try:
        result = await self.model.predict(text)
        return {"result": result}
    except Exception as e:
        # Log the error and return a user-friendly message
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
```

### 3. Input Validation

```python
from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)

@chute.cord(input_schema=PredictionInput)
async def predict(self, data: PredictionInput) -> dict:
    # Input is automatically validated
    return await self.model.generate(data.text, temperature=data.temperature)
```

### 4. Performance Optimization

```python
@chute.on_startup()
async def optimize(self):
    import torch

    # Optimize for inference
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True

    # Pre-compile models if possible
    self.model = torch.jit.script(self.model)
```

## Common Patterns

### Model Loading

```python
@chute.on_startup()
async def load_models(self):
    from transformers import AutoModel, AutoTokenizer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "bert-base-uncased"

    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name).to(device)
    self.device = device
```

### Batched Processing

```python
@chute.cord(public_api_path="/batch-predict")
async def batch_predict(self, texts: list[str]) -> list[dict]:
    # Process multiple inputs efficiently
    inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(self.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = self.model(**inputs)

    return [{"result": output.tolist()} for output in outputs.last_hidden_state]
```

### Streaming Responses

```python
@chute.cord(public_api_path="/stream", stream=True)
async def stream_generate(self, prompt: str):
    for token in self.model.generate_stream(prompt):
        yield {"token": token}
```

## Next Steps

- **[Cords (API Endpoints)](/docs/core-concepts/cords)** - Learn how to define custom API endpoints
- **[Jobs (Background Tasks)](/docs/core-concepts/jobs)** - Understand background job processing
- **[Images (Docker Containers)](/docs/core-concepts/images)** - Build custom Docker environments
- **[Node Selection](/docs/core-concepts/node-selection)** - Optimize hardware allocation
- **[Your First Custom Chute](/docs/getting-started/first-chute)** - Complete example walkthrough
